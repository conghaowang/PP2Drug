import os
import argparse
import rdkit
from rdkit import Chem
from tqdm import tqdm
import re
import subprocess

def build_pdb_dict(raw_data_path):
    pdb_dict = {}
    for directory in os.listdir(raw_data_path):
        dir_path = os.path.join(raw_data_path, directory)
        if os.path.isdir(dir_path):
            # Get the list of files in the directory
            files = os.listdir(dir_path)

            # Add the directory and its files to the dictionary
            pdb_dict[directory] = files
    pdb_rev_dict = {v:k for k, files in pdb_dict.items() for v in files}
    return pdb_dict, pdb_rev_dict


def molecule_docking(bridge_type, res_path, gen_files, pdb_rev_dict):
    for gen_file in tqdm(gen_files):
        ligand_name = gen_file.split('.')[0]
        pattern = r"(\w+_[A-Z]_rec)"
        match = re.search(pattern, ligand_name)
        if match:
            pr_pdb = match.group(1)
        else:
            print(f"Cannot identify the receptor from the file name of ligand {ligand_name}")
            continue

        ligand_file = os.path.join(res_path, gen_file)
        pdb_folder = pdb_rev_dict[gen_file]
        protein_file = os.path.join(raw_data_path, pdb_folder, pr_pdb + '.pdb')
        autobox_ligand_file = os.path.join(raw_data_path, pdb_folder, gen_file)

        if os.path.exists(ligand_file) == False or os.path.exists(protein_file) == False or os.path.exists(autobox_ligand_file) == False:
            print(f"Cannot find the ligand file, protein file or original ligand file for ligand {ligand_name}")
            continue

        log_path = '../../docking_res/logs/' + bridge_type
        out_path = '../../docking_res/output/' + bridge_type
        os.makedirs(log_path, exist_ok=True)
        os.makedirs(out_path, exist_ok=True)

        log_file = os.path.join(log_path, ligand_name + '.log')
        out_file = os.path.join(out_path, ligand_name + '.sdf')
        subprocess.run(['gnina', '-r', protein_file, '-l', ligand_file, '--autobox_ligand', autobox_ligand_file, '-o', out_file, '--exhaustiveness', '16', '--log', log_file, '-q'], stdout=subprocess.DEVNULL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bridge', '-b', type=str, default='vp', help='Type of bridge to use')
    args = parser.parse_args()
    bridge_type = args.bridge

    res_path = '../../generation_results/' + bridge_type
    raw_data_path = '../../data/cleaned_crossdocked_data/raw'

    gen_files = os.listdir(res_path)
    pdb_dict, pdb_rev_dict = build_pdb_dict(raw_data_path)

    molecule_docking(bridge_type, res_path, gen_files, pdb_rev_dict)