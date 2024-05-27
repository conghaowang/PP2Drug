from rdkit import Chem
import numpy as np
import pandas as pd
from collections import defaultdict
import sys
import pickle
import os
import argparse
sys.path.append('../')
sys.path.append('../data_processing/')

from openbabel import pybel
import matplotlib
import matplotlib.pylab as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import seaborn as sns
import random
import torch
from tqdm import tqdm

from data_processing.ligand import Ligand
from data_processing.utils import sample_probability, PP_TYPE_MAPPING
from script_utils import load_qm9_data
from utils import extract_pp, pp_match


def load_generated_mols(generated_path):
    mols = []
    gen_map = defaultdict(list)
    for file in tqdm(os.listdir(generated_path)):
        smi = file.split('.')[0]
        m = Chem.MolFromMolFile(os.path.join(generated_path, file))
        mols.append(m)
        gen_map[smi] = m    # use the latest generated molecule if there are multiple
    return mols, gen_map


def compute_matching_scores(generated_path, pp_info, threshold=1.5):
    score_dict = {}
    match_dict = {}
    for file in tqdm(os.listdir(generated_path)):
        smi = file.split('.')[0]
        mol_path = os.path.join(generated_path, smi+'.sdf')
        rdmol = Chem.MolFromMolFile(mol_path, sanitize=False)
        pbmol = next(pybel.readfile("sdf", mol_path))
        try:
            ligand = Ligand(pbmol, rdmol, atom_positions=None, conformer_axis=None, filtering=False)
        except:
            print('ligand init failed')
            continue
        pp_atom_indices, pp_positions, pp_types, pp_index = extract_pp(ligand)
        
        ref_pp_info = pp_info[smi]
        if ['pp_types', 'pp_positions'] not in list(ref_pp_info.keys()):
            # print(ref_pp_info)
            continue
        if isinstance(ref_pp_info['pp_types'], list):
            # print(ref_pp_info)
            ref_pp_info = {k:v[-1] for k, v in ref_pp_info.items()}     # To address the iterated list issue, for now we use the last element (which is appended to the list at last). TODO: fix this in the data processing script
        
        match = pp_match(pp_types, pp_positions, ref_pp_info, threshold=threshold)
        match_dict[smi] = match
        score = np.mean(match)
        score_dict[smi] = score

    return match_dict, score_dict


def save_matching_scores(match_dict, score_dict, save_path):
    with open(save_path + '_matches.pkl', 'wb') as f:
        pickle.dump(match_dict, f)
        # pickle.dump(score_dict, f)
    score_df = pd.DataFrame.from_dict(score_dict, orient='index', columns=['score'])
    score_df.to_csv(save_path + '_scores.csv')


def plot_matching_scores(score_dict, save_path):
    scores = np.array([v for v in score_dict.values()])
    scores = np.clip(scores, 0, 1)
    ax = sns.histplot(scores, bins=15)
    # ax.bar_label('{:.2f} %'.format(ax.containers[0]))
    ax.set_title('Pharmacophore Matching Scores')
    ax.set_xlabel('Matching Score')
    ax.set_ylabel('Frequency')
    plt.savefig(save_path + '_score_dist.png', dpi=300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', type=str, default='../lightning_logs/vp_bridge_egnn_QM9Dataset_2024-05-22_19_48_30.573332/reconstructed_mols', help='Path to the generated molecules')
    args = parser.parse_args()

    # if args.path.endswith('aromatic_mode'):
    #     basic_mode = False
    # else:
    #     basic_mode = True

    # mols, gen_map = load_generated_mols(args.path)
    # print(f'Loaded {len(mols)} generated molecules')

    with open('../../data/qm9/metadata/pp_info.pkl', 'rb') as f:
        pp_info = pickle.load(f)
    # pp_info = {k:v for k, v in pp_info.items() if k in gen_map.keys()}

    match_dict, score_dict = compute_matching_scores(args.path, pp_info)
    save_matching_scores(match_dict, score_dict, args.path)
    plot_matching_scores(score_dict, args.path)