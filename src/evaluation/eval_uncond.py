from rdkit import Chem
from rdkit.Chem import RDConfig
from rdkit.Chem.QED import qed
import pickle
import argparse
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
from utils_eval import build_pdb_dict

def get_mols(gen_path):
    mols = {}
    for file in os.listdir(gen_path):
        m = Chem.MolFromMolFile(os.path.join(gen_path, file))
        mols[file] = m
    return mols

def assess_uniqueness(mol_dict):
    gen_smis = [Chem.MolToSmiles(m) for m in mol_dict.values()]
    unique_smis = set(gen_smis)
    return len(unique_smis) / len(gen_smis)


def assess_novelty(mol_dict, raw_path, pdb_rev_dict):
    num_novel = 0
    for file, m in mol_dict.items():
        smi = Chem.MolToSmiles(m)
        ref_mol_file = os.path.join(raw_path, pdb_rev_dict[file], file)
        ref_mol = Chem.MolFromMolFile(ref_mol_file)
        ref_smi = Chem.MolToSmiles(ref_mol)
        if smi != ref_smi:
            num_novel += 1
            
    return num_novel / len(mol_dict)


def assess_qed(mol_dict):
    return {file: qed(m) for file, m in mol_dict.items()}


def assess_sa(mol_dict):
    return {file: sascorer.calculateScore(m) for file, m in mol_dict.items()}


def save_sa_qed(sa_dict, qed_dict, gen_path):
    with open(gen_path + '_SA_QED.pkl', 'wb') as f:
        pickle.dump({'SA': sa_dict, 'QED': qed_dict}, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', '-r', type=str, default='../lightning_logs/uncond_vp_bridge_egnn_CombinedUnconditionalDataset_2024-09-05_00_09_16.277074/ten_thousand', help='Path to the generated molecules')
    parser.add_argument('--aromatic', '-a', action='store_true', help='Use aromatic atoms')
    parser.add_argument('--no_optimization', '-no_opt', action='store_false', help='Do not optimize the ligand before docking')
    args = parser.parse_args()

    raw_data_path = '../../data/cleaned_crossdocked_data/raw'
    pdb_dict, pdb_rev_dict = build_pdb_dict(raw_data_path)

    gen_path = os.path.join(args.root, 'reconstructed_mols')
    gen_path += '_aromatic_mode' if args.aromatic else ''
    gen_path += '_optimized' if args.no_optimization else ''

    mol_dict = get_mols(gen_path)
    num_samples = 1e4

    validity = len(mol_dict) / num_samples
    print(f'Validity: {validity}')
    uniqueness = assess_uniqueness(mol_dict)
    print(f'Uniqueness: {uniqueness}')
    novelty = assess_novelty(mol_dict, raw_data_path, pdb_rev_dict)
    print(f'Novelty: {novelty}')

    qed_dict = assess_qed(mol_dict)
    sa_dict = assess_sa(mol_dict)
    save_sa_qed(sa_dict, qed_dict, gen_path)