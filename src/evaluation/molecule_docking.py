import os
import argparse
import rdkit
from rdkit import Chem
from tqdm import tqdm
import re
import subprocess
from utils import build_pdb_dict


def molecule_docking(bridge_type, res_path, gen_files, pdb_rev_dict, root):
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

        # log_path = '../../docking_res/logs/' + bridge_type
        # out_path = '../../docking_res/output/' + bridge_type

        log_path = os.path.join(root, 'logs')
        out_path = os.path.join(root, 'output')
        os.makedirs(log_path, exist_ok=True)
        os.makedirs(out_path, exist_ok=True)

        log_file = os.path.join(log_path, ligand_name + '.log')
        out_file = os.path.join(out_path, ligand_name + '.sdf')
        subprocess.run(['gnina', '-r', protein_file, '-l', ligand_file, '--autobox_ligand', autobox_ligand_file, '-o', out_file, '--exhaustiveness', '16', '--log', log_file, '-q'], stdout=subprocess.DEVNULL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bridge', '-b', type=str, default='vp', help='Type of bridge to use')
    parser.add_argument('--root', '-r', type=str, required=True, help='Path of lightning logs')
    args = parser.parse_args()
    bridge_type = args.bridge

    # res_path = '../../generation_results/' + bridge_type
    res_path = os.path.join(args.root, bridge_type)
    raw_data_path = '../../data/cleaned_crossdocked_data/raw'

    gen_files = os.listdir(res_path)
    pdb_dict, pdb_rev_dict = build_pdb_dict(raw_data_path)

    molecule_docking(bridge_type, res_path, gen_files, pdb_rev_dict, root=args.root)