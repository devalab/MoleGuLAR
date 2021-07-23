import errno

import dgl
import numpy as np
import torch
from tqdm import tqdm
from dgl.nn.pytorch.glob import AvgPooling
from dgllife.model import load_pretrained
from dgllife.utils import (PretrainAtomFeaturizer, PretrainBondFeaturizer,
                           mol_to_bigraph)
from rdkit import Chem
from torch.nn import Linear
from torch.utils.data import DataLoader, TensorDataset

import joblib

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

class RFRPredictor():
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        # self.model.load_state_dict(torch.load(model_path))
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
        vals = []
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
        vals = self.model.predict(mol_emb)
        return canonical_smiles, vals, invalid_smiles

if __name__ == '__main__':
    pred = RFRPredictor('./RFRPredictor.pkl')
    print(pred.predict('C'))