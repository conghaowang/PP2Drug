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
from utils_eval import extract_pp, extract_all_pp, pp_match, save_matching_scores, plot_matching_scores


def load_generated_mols(generated_path):
    mols = []
    gen_map = defaultdict(list)
    for file in tqdm(os.listdir(generated_path)):
        filename = file.split('.')[0]
        m = Chem.MolFromMolFile(os.path.join(generated_path, file))
        mols.append(m)
        gen_map[filename] = m    # use the latest generated molecule if there are multiple
    return mols, gen_map

def compute_matching_scores(generated_path, ref_pp_info, threshold=1.5):
    score_dict = {}
    match_dict = {}
    for file in tqdm(os.listdir(generated_path)):
        filename = file.split('.')[0]
        mol_path = os.path.join(generated_path, file)
        rdmol = Chem.MolFromMolFile(mol_path, sanitize=True)
        pbmol = next(pybel.readfile("sdf", mol_path))
        try:
            rdmol = Chem.AddHs(rdmol)
            ligand = Ligand(pbmol, rdmol, atom_positions=None, conformer_axis=None, filtering=False)
        except:
            print('ligand init failed')
            continue
        pp_atom_indices, pp_positions, pp_types, pp_index = extract_all_pp(ligand)
        # print(pp_types)
        
        # if not all(k in list(ref_pp_info.keys()) for k in ['pp_types', 'pp_positions']):
        #     print(ref_pp_info)
        #     continue
        # if isinstance(ref_pp_info['pp_types'], list):
        #     # print(ref_pp_info)
        #     ref_pp_info = {k:v[-1] for k, v in ref_pp_info.items()}     # To address the iterated list issue, for now we use the last element (which is appended to the list at last). TODO: fix this in the data processing script
        
        match = pp_match(pp_types, pp_positions, ref_pp_info, threshold=threshold)
        match_dict[filename] = match
        score = np.mean(match)
        score_dict[filename] = score

    return match_dict, score_dict


def compute_center(combined_pos, Gt_mask):
    GT_mask = ~Gt_mask
    center = torch.mean(combined_pos[GT_mask], dim=0)
    assert center.size(-1) == 3
    return center


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', '-r', type=str, default='structure_based/', help='Path to the generated molecules')
    parser.add_argument('--aromatic', '-a', action='store_true', help='Whether the data is aromatic')
    parser.add_argument('--ligand_name', '-l', type=str, required=True, help='Name of the ligand')
    args = parser.parse_args()
    ligand_name = args.ligand_name

    # folder_name = ligand_name[ligand_name.rfind('rec')+4:ligand_name.rfind('rec')+8]
    folder_name = ligand_name
    generated_path = os.path.join(args.root_path, folder_name, 'aromatic' if args.aromatic else 'basic')
    test_data = torch.load(os.path.join(args.root_path, folder_name, ligand_name + '_aromatic' + '.pt' if args.aromatic else ligand_name + '.pt'))
    center = compute_center(test_data['target_pos'], test_data['Gt_mask'])

    with open(os.path.join(args.root_path, folder_name, 'pp_info.pkl'), 'rb') as f:
        pp_info = pickle.load(f)

    # print(pp_info['pp_positions'], center)
    pp_info['pp_positions'] = pp_info['pp_positions'] - center
    # print(pp_info['pp_positions'])

    match_dict, score_dict = compute_matching_scores(generated_path, pp_info, threshold=1.5)
    save_matching_scores(match_dict, score_dict, os.path.join(args.root_path, folder_name, 'aromatic' if args.aromatic else 'basic'))
    plot_matching_scores(score_dict, os.path.join(args.root_path, folder_name, 'aromatic' if args.aromatic else 'basic'))