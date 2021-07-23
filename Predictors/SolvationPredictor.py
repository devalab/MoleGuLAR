import dgl
import torch
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from dgllife.model import MPNNPredictor
from torch.utils.data import DataLoader
from rdkit import Chem
from tqdm import tqdm

def collate(graphs):
    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    return bg

class FreeSolvPredictor():
    def __init__(self, model_path):
        self.model = MPNNPredictor(74, 12)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))

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
        graphs = []
        for sm in smiles:
            graphs.append(smiles_to_bigraph(sm, edge_featurizer=CanonicalBondFeaturizer(), node_featurizer=CanonicalAtomFeaturizer()))

        loader = DataLoader(graphs, collate_fn=collate)
        scores = []
        for bg in loader:
            try:
                h = bg.ndata.pop('h')
                e = bg.edata.pop('e')
                scores.append(self.model(bg, h, e).item())
            except:
                scores.append(-3.8)
        vals = scores
        return canonical_smiles, vals, invalid_smiles

if __name__ == '__main__':
    predictor = FreeSolvPredictor('./SolvationPredictor.tar')
    print(predictor.predict(['c1ccccc1-c2ccccc2']))
    

