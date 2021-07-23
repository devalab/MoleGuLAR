import numpy as np
from rdkit.Chem.Crippen import MolLogP
from rdkit import Chem
from rdkit.Chem.QED import qed
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from Predictors.SolvationPredictor import FreeSolvPredictor

def linear(smiles, predictor, invalid_reward=-5.0, threshold=0):
    threshold = threshold/100
    mol, prop, nan_smiles = predictor.predict([smiles])
    if len(nan_smiles) == 1:
        return invalid_reward
    return -1 * prop[0] - threshold

def exponential(smiles, predictor, invalid_reward=-5.0, threshold=0):
    threshold = threshold/100
    mol, prop, nan_smiles = predictor.predict([smiles])
    if len(nan_smiles) == 1:
        return invalid_reward
    return np.exp(-1 * prop[0]) - (1 + threshold)

def logarithmic(smiles, predictor, invalid_reward=-5.0, threshold=0):
    threshold = threshold/100
    mol, prop, nan_smiles = predictor.predict([smiles])
    if len(nan_smiles) == 1:
        return invalid_reward
    if prop[0] < threshold:
        return np.log(1 - prop[0])
    else:
        return -1 * np.log(1 + prop[0])

def squared(smiles, predictor, invalid_reward=-5.0, threshold=0):
    mol, prop, nan_smiles = predictor.predict([smiles])
    threshold = threshold/100
    if len(nan_smiles) == 1:
        return invalid_reward
    if prop[0] < threshold:
        return (prop[0] + threshold) ** 2
    else:
        return -1 * (prop[0] + threshold) ** 2


def SolvationReward(smiles, predictor, invalid_reward=-5, **kwargs):
    mol, prop, nan_smiles = predictor.predict([smiles])
    # prop = np.array(prop)
    # prop = ((prop + 25.47) / 29) * 8

    if len(nan_smiles) == 1:
        return invalid_reward
    scores = prop
    if 'solvation' in kwargs:
        threshold = kwargs['solvation']
    else:
        threshold = -10
    for i in range(len(scores)):
        if scores[i] > threshold - 0.5 and scores[i] < threshold:
            scores[i] = np.exp(15 * (scores[i] - (threshold - 0.5)))
        elif scores[i] >= threshold and scores[i] < (threshold + 0.5):
            scores[i] = np.exp(-15 * (scores[i] - (threshold + 0.5)))
        else:
            scores[i] = -10
    return scores[0]
 
def QEDReward(smiles, invalid_reward=-5.0, **kwargs):
    canonical_indices = []
    invalid_indices = []
    smiles = [smiles]
    pbar = range(len(smiles))
    for i in pbar:
        sm = smiles[i]
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
    
    if len(invalid_smiles) == 1:
        return invalid_reward
    mols = [Chem.MolFromSmiles(sm) for sm in canonical_smiles]
    arr = []
    for mol in mols:
        try:
            val = qed(mol)
        except:
            val = invalid_reward
        arr.append(val)
    scores = np.array(arr)
    if 'QED' in kwargs:
        threshold = kwargs['QED']
    else:
        #scores = np.array(arr)
        for i in range(len(scores)):
            if scores[i] != invalid_reward:
                scores[i] = np.exp(scores[i] * 8)
        return scores[0]
    for i in range(len(scores)):
        if scores[i] > threshold - 0.05 and scores[i] < threshold:
            scores[i] = np.exp(150 * (scores[i] - (threshold - 0.05)))
        elif scores[i] >= threshold and scores[i] < threshold + 0.05:
            scores[i] = np.exp(-150 * (scores[i] - (threshold + 0.05)))
        else:
            scores[i] = -10
    return scores[0]

def LogPReward(smiles, invalid_reward=-5.0, **kwargs):
    canonical_indices = []
    invalid_indices = []
    smiles = [smiles]
    pbar = range(len(smiles))
    for i in pbar:
        sm = smiles[i]
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
    
    if len(invalid_smiles) == 1:
        return invalid_reward
    mols = [Chem.MolFromSmiles(sm) for sm in canonical_smiles]
    scores = np.array([MolLogP(mol) for mol in mols])
    if 'LogP' in kwargs:
        threshold = kwargs['LogP']
    else:
        threshold = 2.5
    for i in range(len(scores)):
        if scores[i] > threshold - 1.5 and scores[i] < threshold:
            scores[i] = np.exp(5 * (scores[i] - (threshold - 1.5)))
        elif scores[i] >= threshold and scores[i] < threshold + 1.5:
            scores[i] = np.exp(-5 * (scores[i] - (threshold + 1.5)))
        else:
            scores[i] = -10
        # if scores[i] >= 1 and scores[i] <= 5:
        #     scores[i] = 10
        # else:
        #     scores[i] = -10
    return scores[0]
    #return (10 * np.exp(-((scores - 2)**2)/1.7) - 1)[0]

def TPSAReward(smiles, invalid_reward=-5.0, **kwargs):
    canonical_indices = []
    invalid_indices = []
    smiles = [smiles]
    pbar = range(len(smiles))
    for i in pbar:
        sm = smiles[i]
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
    
    if len(invalid_smiles) == 1:
        return invalid_reward
    mols = [Chem.MolFromSmiles(sm) for sm in canonical_smiles]
    scores = np.array([CalcTPSA(mol) for mol in mols])
    if 'TPSA' in kwargs:
        threshold = kwargs['TPSA']
    else:
        threshold = 100
    for i in range(len(scores)):
        if scores[i] > threshold - 5 and scores[i] < threshold:
            scores[i] = np.exp(1.5 * (scores[i] - (threshold - 5)))
        elif scores[i] >= threshold and scores[i] < threshold + 5:
            scores[i] = np.exp(-1.5 * (scores[i] - (threshold + 5)))
        else:
            scores[i] = -10
        # if scores[i] > threshold:
        #     scores[i] = -10
        # else:
        #     scores[i] = np.exp(scores[i] * (8/threshold))
    return scores[0]

class MultiReward():
    def __init__(self, func, use_docking=True, use_logP = True, use_qed=True, use_tpsa=True, use_solvation=True, **kwargs):
        self.func = func
        self.use_docking = use_docking
        self.use_logP = use_logP
        self.use_qed = use_qed
        self.use_tpsa = use_tpsa
        self.use_solvation = use_solvation
        self.solvation_predictor = FreeSolvPredictor('./Predictors/SolvationPredictor.tar')
        # self.threshold = threshold
        self.thresholds = kwargs
        # self.invalid_reward = invalid_reward
    
    def __call__(self, smiles, predictor, invalid_reward=-5.0, threshold=0):
        if self.use_docking == True:
            reward1 = self.func(smiles, predictor, invalid_reward, threshold)
        else:
            reward1 = 0
        if self.use_logP == True:
            reward2 = LogPReward(smiles, invalid_reward, **self.thresholds)
        else:
            reward2 = 0
        if self.use_qed == True:
            reward3 = QEDReward(smiles, invalid_reward, **self.thresholds)
        else:
            reward3 = 0
        if self.use_tpsa == True:
            reward4 = TPSAReward(smiles, invalid_reward, **self.thresholds)
        else:
            reward4 = 0
        if self.use_solvation == True:
            reward5 = SolvationReward(smiles, self.solvation_predictor, **self.thresholds)
        else:
            reward5 = 0
        # print("========================================")
        # print(reward1, reward2)
        return reward1 + reward2 + reward3 + reward4 + reward5
    def __str__(self):
        return f"{self.func}, {self.use_docking}, {self.use_logP}, {self.use_qed}, {self.use_tpsa}, {self.use_solvation}\n{self.thresholds}\n============="

if __name__ == "__main__":
    print(LogPReward("C"))
