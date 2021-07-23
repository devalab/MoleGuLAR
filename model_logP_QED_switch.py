import sys

sys.path.append('./release')

import argparse
import os
import pickle
import shutil

import numpy as np
import seaborn as sns
import torch
import wandb
from data import GeneratorData
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, rdmolfiles
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.QED import qed
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from reinforcement import Reinforcement
from stackRNN import StackAugmentedRNN
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ExponentialLR, StepLR
from tqdm import tqdm, trange
from utils import canonical_smiles

import rewards as rwds
from Predictors.GINPredictor import Predictor as GINPredictor
from Predictors.RFRPredictor import RFRPredictor
from Predictors.SolvationPredictor import FreeSolvPredictor

RDLogger.DisableLog('rdApp.info')

parser = argparse.ArgumentParser()
parser.add_argument("--reward_function", help="Reward Function linear/exponential/log/squared", default="linear")
parser.add_argument("--device", help="GPU/CPU", default="GPU")
parser.add_argument("--gen_data", default='./random.smi')
parser.add_argument("--num_iterations", default=100, type=int, help="Number of iterations")
parser.add_argument("--use_wandb", default='yes', help="Perform logging using wandb")
parser.add_argument("--use_checkpoint", default='yes', help="Load from a checkpoint")
parser.add_argument("--adaptive_reward", default='yes', help="Change reward with iterations")
parser.add_argument("--logP", default='no', help="Reward for LogP")
parser.add_argument("--qed", default='no', help="Reward for QED")
parser.add_argument("--tpsa", default='no', help="Reward for TPSA")
parser.add_argument("--solvation", default='no', help="Reward for Solvation")
parser.add_argument("--switch", default='no', help="switch reward function")
parser.add_argument("--predictor", default='dock', help='Choose prediction algorithm')
parser.add_argument("--protein", default='4BTK', help='4BTK/6LU7')
parser.add_argument("--remarks", default="")
parser.add_argument("--logP_threshold", default=2.5, type=float)
parser.add_argument("--tpsa_threshold", default=100, type=float)
parser.add_argument("--solvation_threshold", default=-10, type=float)
parser.add_argument("--qed_threshold", default=0.8, type=float)
parser.add_argument("--switch_frequency", default=35, type=int)



args = parser.parse_args()
switch_frequency = args.switch_frequency
thresholds = {
    'TPSA': args.tpsa_threshold,
    'LogP': args.logP_threshold,
    'solvation': args.solvation_threshold,
    'QED': args.qed_threshold
}

receptor = args.protein

if args.device == "CPU":
    device = torch.device('cpu')
    use_cuda = False
elif args.device == "GPU":
    if torch.cuda.is_available():
        use_cuda = True
        device = torch.device('cuda:0')
    else:
        print("Sorry! GPU not available Using CPU instead")
        device = torch.device('cpu')
        use_cuda = False
else:
    print("Invalid Device")
    quit(0)

OVERALL_INDEX = 0

if args.use_wandb == "yes":
    wandb.init(project=f"{args.reward_function}_{args.remarks}")
    wandb.config.update(args)

if args.reward_function == 'linear':
    get_reward = rwds.linear
if args.reward_function == 'exponential':
    get_reward = rwds.exponential
if args.reward_function == 'logarithmic':
    get_reward = rwds.logarithmic
if args.reward_function == 'squared':
    get_reward = rwds.squared

use_docking=True
use_qed = False
use_tpsa = False
use_solvation = False
use_logP = False
if args.logP == 'yes':
    use_logP = True
if args.qed == 'yes':
    use_qed = True
if args.tpsa == 'yes':
    use_tpsa = True
if args.solvation == 'yes':
    use_solvation = True

get_reward = rwds.MultiReward(get_reward, use_docking, use_logP, use_qed, use_tpsa, use_solvation, **thresholds)


if os.path.exists(f"./logs_{args.reward_function}_{args.remarks}") == False:
    os.mkdir(f"./logs_{args.reward_function}_{args.remarks}")
else:
    shutil.rmtree(f"./logs_{args.reward_function}_{args.remarks}")
    os.mkdir(f"./logs_{args.reward_function}_{args.remarks}")

if os.path.exists(f"./molecules_{args.reward_function}_{args.remarks}") == False:
    os.mkdir(f"./molecules_{args.reward_function}_{args.remarks}")
else:
    shutil.rmtree(f"./molecules_{args.reward_function}_{args.remarks}")
    os.mkdir(f"./molecules_{args.reward_function}_{args.remarks}")


if os.path.exists("./trajectories") == False:
    os.mkdir(f"./trajectories")
if os.path.exists("./rewards") == False:
    os.mkdir(f"./rewards")
if os.path.exists("./losses") == False:
    os.mkdir(f"./losses")
if os.path.exists("./models") == False:
    os.mkdir(f"./models")
if os.path.exists("./predictions") == False:
    os.mkdir("./predictions")


MODEL_NAME = f"./models/model_{args.reward_function}_{args.remarks}"
LOGS_DIR = f"./logs_{args.reward_function}_{args.remarks}"
MOL_DIR = f"./molecules_{args.reward_function}_{args.remarks}"

TRAJ_FILE = open(f"./trajectories/traj_{args.reward_function}_{args.remarks}", "w")
LOSS_FILE = f"./losses/{args.reward_function}_{args.remarks}"
REWARD_FILE = f"./rewards/{args.reward_function}_{args.remarks}"

gen_data_path = args.gen_data
tokens = ['<', '>', '#', '%', ')', '(', '+', '-', '/', '.', '1', '0', '3', '2', '5', '4', '7',
          '6', '9', '8', '=', 'A', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'P', 'S', '[', ']',
          '\\', 'c', 'e', 'i', 'l', 'o', 'n', 'p', 's', 'r', '\n']

gen_data = GeneratorData(training_data_path=gen_data_path, delimiter='\t',
                         cols_to_read=[0], keep_header=True, tokens=tokens)

def dock_and_get_score(smile, test=False):
    global MOL_DIR
    global LOGS_DIR
    global OVERALL_INDEX
    mol_dir = MOL_DIR
    log_dir = LOGS_DIR
    try:
        # path = "python2.5 ~/MGLTools-1.5.6/mgltools_x86_64Linux2_1.5.6/MGLToolsPckgs/AutoDockTools/Utilities24"
        path = "~/MGLTools-1.5.6/MGLToolsPckgs/AutoDockTools/Utilities24"
        mol = Chem.MolFromSmiles(smile)
        AllChem.EmbedMolecule(mol)
        if test == True:
            MOL_DIR = f"validation_mols_{args.reward_function}_{args.remarks}"
            LOGS_DIR = f"validation_log_{args.reward_function}_{args.remarks}"

            if os.path.exists(MOL_DIR):
                shutil.rmtree(MOL_DIR)
                os.mkdir(MOL_DIR)
            else:
                os.mkdir(MOL_DIR)

            if os.path.exists(LOGS_DIR):
                shutil.rmtree(LOGS_DIR)
                os.mkdir(LOGS_DIR)
            else:
                os.mkdir(LOGS_DIR)
            print(MOL_DIR, LOGS_DIR)
        rdmolfiles.MolToPDBFile(mol, f"{MOL_DIR}/{str(OVERALL_INDEX)}.pdb")

        os.system(f"{path}/prepare_ligand4.py -l {MOL_DIR}/{str(OVERALL_INDEX)}.pdb -o {MOL_DIR}/{str(OVERALL_INDEX)}.pdbqt > /dev/null 2>&1")
        os.system(f"{path}/prepare_receptor4.py -r {receptor}.pdb > /dev/null 2>&1")
        os.system(f"{path}/prepare_gpf4.py -i {receptor}_ref.gpf -l {MOL_DIR}/{str(OVERALL_INDEX)}.pdbqt -r {receptor}.pdbqt > /dev/null 2>&1")

        os.system(f"autogrid4 -p {receptor}.gpf > /dev/null 2>&1")
        os.system(f"~/AutoDock-GPU/bin/autodock_gpu_64wi -ffile {receptor}.maps.fld -lfile {MOL_DIR}/{str(OVERALL_INDEX)}.pdbqt -resnam {LOGS_DIR}/{str(OVERALL_INDEX)} -nrun 10 -devnum 1 > /dev/null 2>&1")

        cmd = f"cat {LOGS_DIR}/{str(OVERALL_INDEX)}.dlg | grep -i ranking | tr -s '\t' ' ' | cut -d ' ' -f 5 | head -n1"
        stream = os.popen(cmd)
        output = float(stream.read().strip())
        print(LOGS_DIR, OVERALL_INDEX)
        print(output, smile)
        OVERALL_INDEX += 1
        MOL_DIR = mol_dir
        LOGS_DIR = log_dir
        return output
    except Exception as e:
        MOL_DIR = mol_dir
        LOGS_DIR = log_dir
        print(smile)
        OVERALL_INDEX += 1
        print(f"Did Not Complete because of {e}")
        return 0

class Predictor(object):
    def __init__(self, path):
        super(Predictor, self).__init__()
        self.path = path

    def predict(self, smiles, test=False, use_tqdm=False):
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
        prediction = [dock_and_get_score(smiles[index], test) for index in canonical_indices]
        return canonical_smiles, prediction, invalid_smiles

def estimate_and_update(generator, predictor, n_to_generate):
    generated = []
    pbar = tqdm(range(n_to_generate))
    for i in pbar:
        pbar.set_description("Generating molecules...")
        generated.append(generator.evaluate(gen_data, predict_len=120)[1:-1])

    sanitized = canonical_smiles(generated, sanitize=False, throw_warning=False)[:-1]
    unique_smiles = list(np.unique(sanitized))[1:]
    smiles, prediction, nan_smiles = predictor.predict(unique_smiles, test=True, use_tqdm=True)

    return smiles, prediction

def simple_moving_average(previous_values, new_value, ma_window_size=10):
    value_ma = np.sum(previous_values[-(ma_window_size-1):]) + new_value
    value_ma = value_ma/(len(previous_values[-(ma_window_size-1):]) + 1)
    return value_ma

if args.predictor == 'dock':
    my_predictor = Predictor("")

if args.predictor != 'dock':
    if args.protein == '6LU7':
        print("Predictor models not supported for 6LU7")
        quit(0)
    elif args.predictor == 'rfr':
        my_predictor = RFRPredictor('./Predictors/RFRPredictor.pkl')
    elif args.predictor == 'gin':
        my_predictor = GINPredictor('./Predictors/GINPredictor.tar')

if args.use_checkpoint == "yes":
    if os.path.exists(f"./models/model_{args.reward_function}_{args.remarks}") == True:
        model_path = f"./models/model_{args.reward_function}_{args.remarks}"
    else:
        model_path = './checkpoints/generator/checkpoint_biggest_rnn'
else:
    model_path = './checkpoints/generator/checkpoint_biggest_rnn'


hidden_size = 1500
stack_width = 1500
stack_depth = 200
layer_type = 'GRU'
lr = 0.001
optimizer_instance = torch.optim.Adadelta
n_to_generate = 100
n_policy_replay = 10
n_policy = 15
n_iterations = args.num_iterations

generator = StackAugmentedRNN(input_size=gen_data.n_characters,
                                     hidden_size=hidden_size,
                                     output_size=gen_data.n_characters,
                                     layer_type=layer_type,
                                     n_layers=1, is_bidirectional=False, has_stack=True,
                                     stack_width=stack_width, stack_depth=stack_depth,
                                     use_cuda=use_cuda,
                                     optimizer_instance=optimizer_instance, lr=lr)
generator.load_model(model_path)

RL = Reinforcement(generator, my_predictor, get_reward)

rewards = []
rl_losses = []
preds = []
logp_iter = []
solvation_iter = []
qed_iter = []
tpsa_iter = []
solvation_predictor = FreeSolvPredictor('./Predictors/SolvationPredictor.tar')
PRED_FILE = f"./predictions/{args.reward_function}_{args.remarks}"

use_docking = False
use_logP = True
use_qed = False

use_arr = np.array([False, False, True])
for i in range(n_iterations):
    if args.switch == 'yes':
        if args.logP == 'yes' and args.qed == 'yes':
            if i % switch_frequency == 0:
                use_arr = np.roll(use_arr, 1)
                use_docking, use_logP, use_qed = use_arr
            get_reward = rwds.MultiReward(rwds.exponential, use_docking, use_logP, use_qed, use_tpsa, use_solvation, **thresholds) 
            print(get_reward)
        if args.logP == 'yes' and args.qed == 'no':
            if i % switch_frequency == 0:
                use_logP = not use_logP
                use_docking = not use_docking
            get_reward = rwds.MultiReward(rwds.exponential, use_docking, use_logP, use_qed, use_tpsa, use_solvation, **thresholds) 
            print(get_reward)
    for j in trange(n_policy, desc="Policy Gradient...."):
        if args.adaptive_reward == 'yes':
            cur_reward, cur_loss = RL.policy_gradient(gen_data,  get_reward, OVERALL_INDEX)
        else:
            cur_reward, cur_loss = RL.policy_gradient(gen_data, get_reward)
        rewards.append(simple_moving_average(rewards, cur_reward))
        rl_losses.append(simple_moving_average(rl_losses, cur_loss))
    
    smiles_cur, prediction_cur = estimate_and_update(RL.generator, my_predictor, n_to_generate)
    preds.append(sum(prediction_cur)/len(prediction_cur))
    logps = [MolLogP(Chem.MolFromSmiles(sm)) for sm in smiles_cur]
    tpsas = [CalcTPSA(Chem.MolFromSmiles(sm)) for sm in smiles_cur]
    qeds = []
    for sm in smiles_cur:
        try:
            qeds.append(qed(Chem.MolFromSmiles(sm)))
        except:
            pass
    _, solvations, _ = solvation_predictor.predict(smiles_cur)

    logp_iter.append(np.mean(logps))
    solvation_iter.append(np.mean(solvations))
    qed_iter.append(np.mean(qeds))
    tpsa_iter.append(np.mean(tpsas))
    print(f"BA: {preds[-1]}")
    print(f"LogP {logp_iter[-1]}")
    print(f"Hydration {solvation_iter[-1]}")
    print(f"TPSA {tpsa_iter[-1]}")
    print(f"QED {qed_iter[-1]}")
    RL.generator.save_model(f"{MODEL_NAME}")

    if args.use_wandb == 'yes':
        wandb.log({
            "loss" : rewards[-1],
            "reward" : rl_losses[-1],
            "predictions" : preds[-1],
            "logP" : sum(logps) / len(logps),
            "TPSA" : sum(tpsas) / len(tpsas),
            "QED" : sum(qeds) / len(qeds),
            "Solvation" : sum(solvations) / len(solvations)
        })
        wandb.save(MODEL_NAME)
    np.savetxt(LOSS_FILE, rl_losses)
    np.savetxt(REWARD_FILE, rewards)
    np.savetxt(PRED_FILE, preds)

TRAJ_FILE.close()
if args.use_wandb == 'yes':
    wandb.save(MODEL_NAME)
