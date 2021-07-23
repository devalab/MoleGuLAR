import errno

import dgl
import gpytorch
import numpy as np
import torch
from tqdm import tqdm
from dgl.nn.pytorch.glob import AvgPooling
from dgllife.model import load_pretrained
from dgllife.utils import (PretrainAtomFeaturizer, PretrainBondFeaturizer,
                           mol_to_bigraph)
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.mlls import AddedLossTerm, DeepApproximateMLL, VariationalELBO
from gpytorch.models import GP, ApproximateGP
from gpytorch.models.deep_gps import DeepGP, DeepGPLayer
from gpytorch.variational import (CholeskyVariationalDistribution,
                                  VariationalStrategy)
from rdkit import Chem
from torch.nn import Linear
from torch.utils.data import DataLoader, TensorDataset


def graph_construction_and_featurization(smiles):
    """Construct graphs from SMILES and featurize them
    Parameters
    ----------
    smiles : list of str
        SMILES of molecules for embedding computation
    Returns
    -------
    list of DGLGraph
        List of graphs constructed and featurized
    list of bool
        Indicators for whether the SMILES string can be
        parsed by RDKit
    """
    graphs = []
    success = []
    for smi in smiles:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                success.append(False)
                continue
            g = mol_to_bigraph(mol, add_self_loop=True,
                               node_featurizer=PretrainAtomFeaturizer(),
                               edge_featurizer=PretrainBondFeaturizer(),
                               canonical_atom_order=False)
            graphs.append(g)
            success.append(True)
        except:
            success.append(False)

    return graphs, success

def collate(graphs):
    return dgl.batch(graphs)


class DeepGPHiddenLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=128, mean_type='constant'):
        if output_dims is None:
            inducing_points = torch.randn(num_inducing, input_dims)
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.randn(output_dims, num_inducing, input_dims)
            batch_shape = torch.Size([output_dims])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super(DeepGPHiddenLayer, self).__init__(variational_strategy, input_dims, output_dims)

        if mean_type == 'constant':
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        else:
            self.mean_module = LinearMean(input_dims)
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel())


    def forward(self, x):
        mean_x = self.mean_module(x) # self.linear_layer(x).squeeze(-1)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, *other_inputs, **kwargs):
        """
        Overriding __call__ isn't strictly necessary, but it lets us add concatenation based skip connections
        easily. For example, hidden_layer2(hidden_layer1_outputs, inputs) will pass the concatenation of the first
        hidden layer's outputs and the input data to hidden_layer2.
        """
        if len(other_inputs):
            if isinstance(x, gpytorch.distributions.MultitaskMultivariateNormal):
                x = x.rsample()

            processed_inputs = [
                inp.unsqueeze(0).expand(self.num_samples, *inp.shape)
                for inp in other_inputs
            ]

            x = torch.cat([x] + processed_inputs, dim=-1)

        return super().__call__(x, are_samples=bool(len(other_inputs)))

class DeepGPModel(DeepGP):
    def __init__(self, train_x_shape):

        hidden_layer = DeepGPHiddenLayer(
            input_dims=train_x_shape[-1],
            output_dims=20,
            mean_type='linear',
        )

        last_layer = DeepGPHiddenLayer(
            input_dims=hidden_layer.output_dims,
            output_dims=None,
            mean_type='constant',
        )

        super().__init__()
        self.hidden_layer = hidden_layer
        self.last_layer = last_layer
        self.likelihood = GaussianLikelihood()

    def forward(self, inputs):
        hidden_rep1 = self.hidden_layer(inputs)
        output = self.last_layer(hidden_rep1)
        return output
    
    def predict(self, test_loader):
        with torch.no_grad():
            mus = []
            variances = []
            lls = []
            gts = []
            for x_batch, y_batch in test_loader:
                # x_batch = x_batch.cuda()
                # y_batch = y_batch.cuda()
                preds = self.likelihood(self(x_batch))
                mus.append(preds.mean)
                variances.append(preds.stddev)
                gts.append(y_batch)
                #lls.append(self.likelihood.log_marginal(y_batch, self(x_batch)))

        return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1), torch.cat(gts, dim=-1)

class DeepGPPredictor():
    def __init__(self, model_path):
        self.model = DeepGPModel((None, 300))
        self.model.load_state_dict(torch.load(model_path))
        self.embeddings = load_pretrained('gin_supervised_infomax').to(torch.device('cpu'))
        self.embeddings.eval()
        self.readout = AvgPooling()

    def predict(self, smiles, use_tqdm=False, test=False):
        canonical_indices = []
        invalid_indices = []
        if use_tqdm:
            pbar = tqdm(range(len(smiles)))
        else:
            pbar = range(len(smiles))
        for i in pbar:
            sm = smiles[i]
            if use_tqdm:
                pbar.set_description("Calculating predictions...")
            try:
                sm = Chem.MolToSmiles(Chem.MolFromSmiles(sm))
                if len(sm) == 0:
                    invalid_indices.append(i)
                else:
                    canonical_indices.append(i)
            except:
                invalid_indices.append(i)
        canonical_smiles = [smiles[i] for i in canonical_indices]
        invalid_smiles = [smiles[i] for i in invalid_indices]
        if len(canonical_indices) == 0:
            return canonical_smiles, [], invalid_smiles
        mol_emb = []
        dataset, success = graph_construction_and_featurization(smiles)
        args = {
            'device' : torch.device('cpu')
        }
        data_loader = DataLoader(dataset, batch_size=len(smiles), shuffle=False, collate_fn=collate)
        for id, bg in enumerate(data_loader):
            nfeats = [bg.ndata.pop('atomic_number').to(args['device']),
                  bg.ndata.pop('chirality_type').to(args['device'])]
            efeats = [bg.edata.pop('bond_type').to(args['device']),
                  bg.edata.pop('bond_direction_type').to(args['device'])]
            with torch.no_grad():
                node_repr = self.embeddings(bg, nfeats, efeats)
            mol_emb.append(self.readout(bg, node_repr))
        mol_emb = torch.cat(mol_emb, dim=0).detach().cpu().numpy()
        dataset = TensorDataset(torch.Tensor(mol_emb), torch.Tensor(mol_emb))
        loader = DataLoader(dataset, batch_size=len(smiles), shuffle=False)
        vals, variances, gts = self.model.predict(loader)
        vals = vals.cpu().detach().numpy().mean(0).tolist()
        return canonical_smiles, vals, invalid_smiles

if __name__ == '__main__':
    pred = DeepGPPredictor('./PredictorModel.tar')
    print(pred.predict('C'))

        

