from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data import download_url, extract_zip
import torch
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from openbabel import pybel
import os
from collections import defaultdict
import pickle
import sys
sys.path.append('../')
from typing import List
from data_processing.ligand import Ligand
from data_processing.utils import PP_TYPE_MAPPING, ATOM_FAMILIES
from data_processing.paired_data import PharmacophoreDataset, CombinedSparseGraphDataset, load_dataset

# QM9 dataset contains less types of atoms than the cross-docked dataset
# ATOM_TYPE_MAPPING = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
# MAP_ATOM_TYPE_AROMATIC_TO_INDEX = {
#     (1, False): 0,  # H
#     (6, False): 1,  # C
#     (6, True): 2,   # C.ar
#     (7, False): 3,  # N
#     (7, True): 4,   # N.ar
#     (8, False): 5,  # O
#     (8, True): 6,   # O.ar
#     (9, False): 7,  # F
# }

ATOM_TYPE_MAPPING = {'C': 1, 'N': 2, 'O': 3, 'F': 4}
MAP_ATOM_TYPE_AROMATIC_TO_INDEX = {
    (6, False): 0, # C
    (6, True): 1,  # C.ar
    (7, False): 2, # N
    (7, True): 3,  # N.ar
    (8, False): 4, # O
    (8, True): 5,  # O.ar
    (9, False): 6  # F
}

class QM9Dataset(CombinedSparseGraphDataset):
    def __init__(self, root, split='all', transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        self._split = split
        self.raw_url = ('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip')
        self.raw_url2 = 'https://ndownloader.figshare.com/files/3195404'
        self.processed_url = 'https://data.pyg.org/datasets/qm9_v3.zip'
        super(QM9Dataset, self).__init__(root, split, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self) -> List[str]:
        try:
            import rdkit  # noqa
            return ['gdb9.sdf', 'gdb9.sdf.csv', 'uncharacterized.txt']
        except ImportError:
            return ['qm9_v3.pt']

    @property
    def processed_file_names(self) -> List[str]:
        return [f'data.pt']

    def download(self):
        try:
            import rdkit  # noqa
            file_path = download_url(self.raw_url, self.raw_dir)
            extract_zip(file_path, self.raw_dir)
            os.unlink(file_path)

            file_path = download_url(self.raw_url2, self.raw_dir)
            os.rename(os.path.join(self.raw_dir, '3195404'),
                      os.path.join(self.raw_dir, 'uncharacterized.txt'))
        except ImportError:
            path = download_url(self.processed_url, self.raw_dir)
            extract_zip(path, self.raw_dir)
            os.unlink(path)

    def process(self):
        with open(self.raw_paths[2], 'r') as f:
            skip = [int(x.split()[0]) - 1 for x in f.read().split('\n')[9:-2]]

        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False, sanitize=True)
        pbmols = list(pybel.readfile("sdf", self.raw_paths[0]))
        
        data_list = []
        skip_num = 0
        pp_info = defaultdict(dict)
        for i, rdmol in enumerate(tqdm(suppl)):
            if i in skip:
                skip_num += 1
                continue

            if rdmol is None:
                skip_num += 1
                continue

            smiles = Chem.MolToSmiles(rdmol)
            pbmol = pbmols[i]
            try:
                rdmol = Chem.AddHs(rdmol)
                ligand = Ligand(pbmol, rdmol, atom_positions=None, conformer_axis=None)
            except Exception as e:
                print(smiles, 'Ligand init failed')
                print(e)
                continue
            num_feat_class = max(len(PP_TYPE_MAPPING.keys()), len(MAP_ATOM_TYPE_AROMATIC_TO_INDEX.keys()))
            try:
                _, x, atomic_numbers, pos, num_nodes = self.extract_atom_features(rdmol, num_feat_class)
                # edge_mask = self.make_edge_mask(num_nodes)
            except KeyError:  # some elements are not considered, skip such ligands
                continue
            try:
                pp_atom_indices, pp_positions, pp_types, pp_index = self.extract_pp(ligand, num_feat_class)
                assert pp_positions.size(1) == 3
            except Exception as e:
                print(smiles, 'extract pp failed')
                print(e)
                continue

            CoM_tensor = self.compute_CoM(pos, atomic_numbers)

            target_x, target_pos, node_pp_index = self.compute_target(x, pos, pp_atom_indices, pp_positions, pp_types, pp_index, CoM_tensor)
            if len(pp_info[smiles].keys()) == 0:
                pp_info[smiles].update({
                    'pp_atom_indices': [pp_atom_indices],
                    'pp_positions': [pp_positions],
                    'pp_types': [pp_types],
                    'pp_index': [pp_index],
                    'node_pp_index': [node_pp_index]
                })
            else:
                pp_info[smiles]['pp_atom_indices'].append(pp_atom_indices)
                pp_info[smiles]['pp_positions'].append(pp_positions)
                pp_info[smiles]['pp_types'].append(pp_types)
                pp_info[smiles]['pp_index'].append(pp_index)
                pp_info[smiles]['node_pp_index'].append(node_pp_index)
            x_ctr, pos_ctr, Gt_mask = self.combine_target(x, pos, target_x, target_pos)
            target_x_ctr, target_pos_ctr, _ = self.combine_target(target_x, target_pos, target_x, target_pos)
            edge_mask_ctr = self.make_edge_mask(num_nodes * 2)
            node_mask_ctr = torch.ones([1, num_nodes * 2], dtype=torch.bool)

            data = Data(x=x_ctr, pos=pos_ctr, target_x=target_x_ctr, target_pos=target_pos_ctr, Gt_mask=Gt_mask, ligand_name=smiles)
            data_list.append(data)

        self.save(data_list, self.processed_paths[0])
        self.save_pp_info(pp_info)


if __name__ == '__main__':
    module = QM9Dataset
    root = '../../data/qm9'
    split = 'all'
    dataset = load_dataset(module, root, split)