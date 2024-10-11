import numpy as np
import pandas as pd
import os, sys
sys.path.append('../evaluation/')
sys.path.append('..')
from tqdm import tqdm
from rdkit import Chem
from openbabel import pybel
from collections import defaultdict
from sklearn.cluster import DBSCAN, HDBSCAN
import pickle
from data_processing.paired_data import CombinedSparseGraphDataset, CombinedSparseGraphDataset
from data_processing.ligand import Ligand
from data_processing.utils import ATOM_TYPE_MAPPING, PP_TYPE_MAPPING
from utils_eval import build_pdb_dict


# this is different from the dataset class mtd
def cluster_non_pp(pos, atom_in_pp):
    pos = np.array(pos)
    # print(pos)
    non_pp_atom_positions = []
    non_pp_atom_pos_dict = []

    for i in range(pos.shape[0]):
        if i not in atom_in_pp:
            # dist_i = np.zeros(num_pp)
            # for j in range(num_pp):
            #     dist_i[j] = np.linalg.norm(atom_positions[i] - pp_positions[j])
            non_pp_atom_positions.append(pos[i])
            non_pp_atom_pos_dict.append({'id':i, 'pos':pos[i]})

    non_pp_atom_positions = np.array(non_pp_atom_positions)
    # print(non_pp_atom_positions)
    if non_pp_atom_positions.shape[0] == 1:
        return {0: [non_pp_atom_pos_dict[0]['id']]}

    clustering_model = HDBSCAN(min_cluster_size=2)
    clustering = clustering_model.fit(non_pp_atom_positions)
    non_pp_atom_labels = clustering.labels_
    max_label = np.max(non_pp_atom_labels)

    for i in range(len(non_pp_atom_labels)):
        if non_pp_atom_labels[i] == -1:
            non_pp_atom_labels[i] = max_label + 1
            max_label += 1

    non_pp_groups = np.unique(non_pp_atom_labels)
    # non_pp_group_center_positions = torch.zeros((len(non_pp_groups), 3))
    non_pp_atom_indices = {label: [] for label in non_pp_groups}

    for group in non_pp_groups:
        # nodes: the index in the non_pp_atom_positions matrix
        nodes = np.where(non_pp_atom_labels==group)[0]
        # print(nodes)

        # atoms: the index in the original ligand
        atoms = []
        for node in nodes:
            # print(node)
            atoms.append(non_pp_atom_pos_dict[int(node)]['id'])
        # print(atoms)
        non_pp_atom_indices[group] = atoms
        
        # positions = non_pp_atom_positions[nodes]
        # print(positions.size())
        # center_pos = np.mean(positions, axis=0)
        # print(center_pos)
        # non_pp_group_center_positions[group] = center_pos

    return non_pp_atom_indices


if __name__ == '__main__':

    raw_path = '../../data/cleaned_crossdocked_data/raw'

    num_appearance = {k: defaultdict(lambda : 0) for k in PP_TYPE_MAPPING.keys()}

    for folder in tqdm(os.listdir(raw_path)):
        folder_path = os.path.join(raw_path, folder)
        if not os.path.isdir(folder_path):
            continue
        for fn in tqdm(os.listdir(folder_path)):
            if fn.split('.')[-1] != 'sdf':
                continue
            file = os.path.join(folder_path, fn)

            rdmol = Chem.MolFromMolFile(file, removeHs=False, sanitize=True)
            pbmol = next(pybel.readfile("sdf", file))

            try:
                rdmol = Chem.AddHs(rdmol)
                ligand = Ligand(pbmol, rdmol, atom_positions=None, conformer_axis=None, filtering=False)
                rdmol = ligand.rdmol_noH
            except Exception as e:
                print(f'Ligand {file} init failed')
                print(e)
                continue

            atom_in_pp = []
            for pp_node in ligand.graph.nodes:
                atom_indices = list([pp_node.atom_indices]) if type(pp_node.atom_indices)==int else list(sorted(pp_node.atom_indices))
                atom_in_pp += atom_indices
                # positions = pp_node.positions.squeeze()
                # index = pp_node.index
                num_nodes = len(atom_indices)

                for pp_type in pp_node.types:
                    # print(pp_type, num_nodes)
                    num_appearance[pp_type][num_nodes] += 1

            conformer = rdmol.GetConformer()
            atom_positions = conformer.GetPositions()
            num_nodes = rdmol.GetNumAtoms()
            # print(atom_positions)
            
            if len(atom_in_pp) < num_nodes:
                non_pp_atom_indices = cluster_non_pp(atom_positions, atom_in_pp)
                for group, atom_indices in non_pp_atom_indices.items():
                    num_nodes = len(atom_indices)
                    num_appearance['Linker'][num_nodes] += 1

    prob_appearance = {}
    for k, d in num_appearance.items():
        total_counts = sum(d.values())
        # print(k, total_counts)
        prob_appearance[k] = {num: count/total_counts for num, count in d.items()}

    print(num_appearance)
    print(prob_appearance)
    num_appearance_dict = {k: dict(v) for k, v in num_appearance.items()}
    prob_appearance_dict = {k: dict(v) for k, v in prob_appearance.items()}

    with open('../../data/cleaned_crossdocked_data/metadata/num_appearance.pkl', 'wb') as f:
        pickle.dump(num_appearance_dict, f)
    with open('../../data/cleaned_crossdocked_data/metadata/prob_appearance.pkl', 'wb') as f:
        pickle.dump(prob_appearance_dict, f)