import dgl
import torch
from dgllife.utils import (PretrainAtomFeaturizer, PretrainBondFeaturizer,
                           mol_to_bigraph)
from rdkit import Chem
from torch.utils.data import DataLoader
from tqdm import tqdm

# from GINPredictor import Net, collate
import dgl
import errno
import numpy as np
import os
import pandas as pd
import torch

from dgl.nn.pytorch.glob import AvgPooling
from dgllife.model import load_pretrained
from dgllife.utils import mol_to_bigraph, PretrainAtomFeaturizer, PretrainBondFeaturizer
from rdkit import Chem
from torch.utils.data import DataLoader
from joblib import Parallel, delayed
from sklearn.metrics import mean_absolute_error

from tqdm import tqdm
from torch.utils.data import Dataset
from torch import nn

def collate(graphs):
    gs, labels = [], []
    for g in graphs:
        gs.append(g[0])
        labels.append(g[1])
    return dgl.batch(gs), torch.tensor(labels)


class dset(Dataset):
    def __init__(self, graphs, y):
        self.g = graphs
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.g[idx], self.y[idx]

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.gin = load_pretrained('gin_supervised_infomax')
        self.readout = AvgPooling()
        self.layer1 = nn.Linear(300, 128)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(128, 32)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(32, 1)
    
    def forward(self, bg, nfeats, efeats):
        node_repr = self.gin(bg, nfeats, efeats)
        node_repr = self.readout(bg, node_repr)
        ans = self.relu1(self.layer1(node_repr))
        ans = self.relu2(self.layer2(ans))
        return self.layer3(ans)


class Predictor(object):
    def __init__(self, path):
        self.args = {
            'device': torch.device('cpu')
        }
        self.net = Net()
        self.net.load_state_dict(torch.load(path, map_location='cpu'))
        self.net.eval()
    def graph_construction_and_featurization(self, smi):
        mol = Chem.MolFromSmiles(smi)
        g = mol_to_bigraph(mol, add_self_loop=True,
                           node_featurizer=PretrainAtomFeaturizer(),
                           edge_featurizer=PretrainBondFeaturizer(),
                           canonical_atom_order=False)
        return g
    
    def collate(self, graphs):
        return dgl.batch(graphs)
    
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
        for sm in canonical_smiles:
            g = self.graph_construction_and_featurization(sm)
            graphs.append(g)
        vals = []
        for bg in graphs:
            if bg == None:
                vals.append(0)
            else:
                dataloader = DataLoader([bg], collate_fn=self.collate, batch_size=1)
                for g in dataloader:
                    g = g.to(self.args['device'])
                    nfeats = [g.ndata.pop('atomic_number').to(self.args['device']),
                              g.ndata.pop('chirality_type').to(self.args['device'])]
                    efeats = [g.edata.pop('bond_type').to(self.args['device']),
                              g.edata.pop('bond_direction_type').to(self.args['device'])]
                    out = self.net(g, nfeats, efeats)
                    out = out.squeeze()
                    vals.append(out.item())
        return canonical_smiles, vals, invalid_smiles