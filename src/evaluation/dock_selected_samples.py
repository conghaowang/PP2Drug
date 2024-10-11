import os
import argparse
import rdkit
from rdkit import Chem
from tqdm import tqdm
import pickle
import pandas as pd
import numpy as np
import re
from collections import defaultdict
import subprocess
from utils_eval import build_pdb_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', '-r', type=str, default='../lightning_logs/vp_bridge_egnn_CombinedSparseGraphDataset_2024-08-19_21_05_04.140916/ligand_based', help='Path of root folder')
    parser.add_argument('--aromatic', '-a', action='store_true', help='Use aromatic atoms')
    parser.add_argument('--no_optimization', '-no_opt', action='store_false', help='Do not optimize the ligand before docking')
    parser.add_argument('--gpu', '-g', type=str, default=0, help='GPU to use for docking')
    args = parser.parse_args()
    optimization = args.no_optimization

    raw_data_path = '../../data/cleaned_crossdocked_data/raw'
    pdb_dict, pdb_rev_dict = build_pdb_dict(raw_data_path)

    fn = 'reconstructed_mols' + '_aromatic_mode' if args.aromatic else 'reconstructed_mols'
    # we don't consider optimization for the pp matching evaluation
    match_fn = fn + '_matches.pkl'
    score_fn = fn + '_scores.csv'
    match_file = os.path.join(args.root, match_fn)
    score_file = os.path.join(args.root, score_fn)
    gen_path = os.path.join(args.root, fn + '_optimized' if optimization else fn)

    with open(match_file, 'rb') as f:
        match_dict = pickle.load(f)

    scores = pd.read_csv(score_file, index_col=0)

    all_lig = match_dict.keys()
    lig_names = [lig[lig.rfind('rec')+9:lig.rfind('rec')+12] for lig in all_lig]
    lig_names = set(lig_names)
    lig_dict = defaultdict(lambda : [])
    for lig in tqdm(all_lig):
        lig_name = lig[lig.rfind('rec')+9:lig.rfind('rec')+12]
        lig_dict[lig_name].append(lig)
    rev_lig_dict = {lig:lig_name for lig_name, ligs in lig_dict.items() for lig in ligs }

    avg_score_df = pd.read_csv(os.path.join(args.root, fn+'_avg_score_by_lig.csv'), index_col=0, header=0)
    filtered_df = avg_score_df[avg_score_df['score'] >= 0.5]

    log_folder = 'logs_aromatic' if args.aromatic else 'logs'
    out_folder = 'output_aromatic' if args.aromatic else 'output'
    log_path = os.path.join(args.root, 'selected', log_folder)
    out_path = os.path.join(args.root, 'selected', out_folder)
    log_path += '_optimized' if optimization else ''
    out_path += '_optimized' if optimization else ''
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(out_path, exist_ok=True)

    for lig_name in tqdm(filtered_df.index):
        ligands = lig_dict[lig_name]
        for ligand in ligands:
            pattern = r"(\w+_[A-Z]_rec)"
            match = re.search(pattern, ligand)
            if match:
                pr_pdb = match.group(1)
            else:
                print(f"Cannot identify the receptor from the file name of ligand {ligand}")
            ligand_fn = ligand[ligand.rfind('rec')+4:ligand.rfind('lig')+3]
            pdb_folder = pdb_rev_dict[ligand + '.sdf']
            protein_file = os.path.join(raw_data_path, pdb_folder, pr_pdb + '.pdb')
            autobox_ligand_file = os.path.join(raw_data_path, pdb_folder, ligand_fn + '.pdb')
            ligand_file = os.path.join(gen_path, ligand + '.sdf')

            if os.path.exists(ligand_file) == False or os.path.exists(protein_file) == False or os.path.exists(autobox_ligand_file) == False:
                print(f"Cannot find the ligand file, protein file or original ligand file for ligand {ligand}")
                continue

            # log_path = '../../docking_res/logs/' + bridge_type
            # out_path = '../../docking_res/output/' + bridge_type

            log_file = os.path.join(log_path, ligand + '.log')
            out_file = os.path.join(out_path, ligand + '.sdf')

            if os.path.exists(out_file) and os.path.exists(log_file):
                if os.path.getsize(out_file) > 0 and os.path.getsize(log_file) > 0:
                    # print(f"Docking result for ligand {ligand_name} already exists")
                    continue
            subprocess.run(['gnina', '-r', protein_file, '-l', ligand_file, '--autobox_ligand', autobox_ligand_file, 
                            '-o', out_file, '--exhaustiveness', '16', '--log', log_file, '--device', args.gpu, '-q'], 
                            stdout=subprocess.DEVNULL)