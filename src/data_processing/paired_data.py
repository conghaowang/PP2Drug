from rdkit import Chem
from openbabel import pybel
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.utils import to_dense_batch
import torch
import numpy as np
import glob
import os
import random
import pickle
from tqdm import tqdm
from collections import defaultdict
import sys
sys.path.append('../')
from data_processing.ligand import Ligand
from data_processing.utils import ATOM_TYPE_MAPPING, PP_TYPE_MAPPING, ATOM_FAMILIES, MAP_ATOM_TYPE_AROMATIC_TO_INDEX
# from script_utils import load_dataset


class PharmacophoreData(Data):
    def __init__(self, x=None, pos=None, target_x=None, target_pos=None, **kwargs):
        super(PharmacophoreData, self).__init__(x=x, pos=pos, target_x=target_x, target_pos=target_pos, **kwargs)
        self.__set_properties__(x, pos, target_x, target_pos)
        # print(self.pos)
        # self.CoM2zero()
        # self.compute_target()

    def __set_properties__(self, x, pos, target_x, target_pos):
        self.x = x
        self.pos = pos
        # self.atomic_numbers = atomic_numbers
        # self.pp_atom_indices = pp_atom_indices
        # self.pp_positions = pp_positions
        # self.pp_types = pp_types
        # self.pp_index = pp_index

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'pos':
            return self.pos.size(0)
        # elif key == 'pp_positions':
        #     return self.pp_positions.size(0)
        # elif key == 'pp_types':
        #     return self.pp_types.size(0)
        # elif key == 'pp_index':
        #     return self.pp_index.size(0)
        # elif key == 'pp_atom_indices':
        #     return len(self.pp_atom_indices)
        elif key == 'target_pos':
            return self.target_pos.size(0)
        else:
            return super(PharmacophoreData, self).__inc__(key, value)

    def compute_CoM(self):
        # compute the center of mass of the ligand
        periodic_table = Chem.GetPeriodicTable()
        # print(self.pos)
        sum_pos = torch.zeros(self.pos.size(1))
        sum_mass = 0
        for i, atomic_number in enumerate(self.atomic_numbers):
            mass = periodic_table.GetAtomicWeight(int(atomic_number))
            sum_pos += mass * self.pos[i]
            sum_mass += mass
        CoM = sum_pos / sum_mass
        return CoM

    def CoM2zero(self):
        # translate the ligand so that its center of mass is at the origin
        self.CoM = self.compute_CoM()
        self.pos -= self.CoM
        self.pp_positions -= self.CoM

    def compute_target(self):
        # compute the target of the diffusion bridge, which is each atom's feat/pos destination regarding its pharmacophore membership
        self.target_x = torch.zeros(self.x.size(0), self.x.size(1))
        self.target_pos = torch.zeros(self.pos.size(0), self.pos.size(1))
        atom_in_pp = []
        # print(self.pp_atom_indices)
        for atom_indices in self.pp_atom_indices:
            atom_in_pp += atom_indices
        for i in range(self.x.size(0)):
            if i not in atom_in_pp:  # if the atom is not in any pharmacophore, we set its target type to Linker:0 and target position to 0
                self.target_x[i] = torch.nn.functional.one_hot(torch.tensor([0]), num_classes=self.pp_types.size(1)).to(torch.float)
                self.target_pos[i] = torch.zeros(self.pos.size(1))
            else:  # if the atom is in a pharmacophore, we set its target type to the pharmacophore type and target position to the pharmacophore position
                for j, atom_indices in enumerate(self.pp_atom_indices):
                    if i in atom_indices:
                        self.target_x[i] = self.pp_types[j]
                        self.target_pos[i] = self.pp_positions[j]
                        break


class PharmacophoreDataset(InMemoryDataset):
    def __init__(self, root, split='train', transform=None, pre_transform=None, pre_filter=None, aromatic=False):
        self.root = root
        self._split = split
        self._max_N = 86
        self.aromatic = aromatic
        super(PharmacophoreDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        all_docked = os.listdir(os.path.join(self.root, 'raw'))
        random.seed(2024)
        random.shuffle(all_docked)
        if self._split == 'train':
            selected_docked = all_docked[:int(len(all_docked)*0.8)]
        elif self._split == 'valid':
            selected_docked = all_docked[int(len(all_docked)*0.8):int(len(all_docked)*0.9)]
        elif self._split == 'test':
            selected_docked = all_docked[int(len(all_docked)*0.9):]
        elif self._split == 'all':
            selected_docked = all_docked
        else:
            raise ValueError('split must be "train" or "test" or "all"')

        raw_files = []
        for docked in selected_docked:
            raw_files += glob.glob(os.path.join(self.root, 'raw', docked, '*.sdf'))
        raw_file_names = [os.path.join(raw_file.split('/')[-2], raw_file.split('/')[-1]) for raw_file in raw_files]
        return raw_file_names

    @property
    def processed_file_names(self):
        if self._split == 'train':
            return ['train.pt']
        elif self._split == 'test':
            return ['test.pt']
        elif self._split == 'valid':
            return ['valid.pt']
        elif self._split == 'all':
            return ['data.pt']
        else: 
            raise ValueError('split must be "train", "valid", "test" or "all"')

    def download(self):
        pass

    # def len(self):
    #     return len(self.raw_file_names)
    
    # def get(self, idx):
    #     return self.data[idx]

    def get_max_N(self):
        print('Calculating the maximum number of atoms in the dataset...')
        all_docked = os.listdir(os.path.join(self.root, 'raw'))
        all_files = []
        for docked in all_docked:
            all_files += glob.glob(os.path.join(self.root, 'raw', docked, '*.sdf'))
        max_N = 0
        for file in tqdm(all_files):
            rdmol = Chem.MolFromMolFile(file, sanitize=False)
            N = rdmol.GetNumAtoms()
            if N > max_N:
                max_N = N

        print('The maximum number of atoms in the dataset is:', max_N)
        return max_N

    def save_pp_info(self, pp_info, split='all'):
        if split == 'train':
            filename = 'train_pp_info.pkl'
        elif split == 'valid':
            filename = 'valid_pp_info.pkl'
        elif split == 'test':
            filename = 'test_pp_info.pkl'
        elif split == 'all':
            filename = 'pp_info.pkl'
        else:
            raise ValueError('split must be "train" or "test" or "all"')
        os.makedirs(os.path.join(self.root, 'metadata'), exist_ok=True)
        with open(os.path.join(self.root, 'metadata', filename), 'wb') as f:
            pickle.dump(pp_info, f)

    def process(self):
        data_list = []
        # print(self.raw_file_names)
        # print(self.raw_paths)
        # max_N = self.get_max_N()
        max_N = self._max_N
        pp_info = defaultdict(dict)
        for raw_path in tqdm(self.raw_paths):
            filename = raw_path.split('/')[-1].split('.')[0]
            # print(raw_path)
            rdmol = Chem.MolFromMolFile(raw_path, sanitize=False)
            pbmol = next(pybel.readfile("sdf", raw_path))
            # try:
            #     pbmol.removeh()
            #     rdmol = Chem.RemoveHs(rdmol)
            # except Exception as e:
            #     print(raw_path, 'remove Hs failed')
            #     print(e)
            #     continue
            try:
                ligand = Ligand(pbmol, rdmol, atom_positions=None, conformer_axis=None)
            except Exception as e:
                print(raw_path, 'Ligand init failed')
                print(e)
                continue
            num_feat_class = max(len(PP_TYPE_MAPPING.keys()), len(MAP_ATOM_TYPE_AROMATIC_TO_INDEX.keys()))
            try:
                _, x, atomic_numbers, pos, num_nodes = self.extract_atom_features(rdmol, num_feat_class, aromatic=self.aromatic)
                edge_mask = self.make_edge_mask(num_nodes, self._max_N)
            except KeyError:  # some elements are not considered, skip such ligands
                continue
            try:
                pp_atom_indices, pp_positions, pp_types, pp_index = self.extract_pp(ligand, num_feat_class)
                assert pp_positions.size(1) == 3
            except Exception as e:
                print(raw_path, 'extract pp failed')
                print(e)
                continue
            # print(raw_path, pos.size())
            
            # should we move CoM to zero during data 
            CoM_tensor = self.compute_CoM(pos, atomic_numbers)
            # pos = self.CoM2zero(pos, CoM)
            # pp_positions = self.CoM2zero(pp_positions, CoM)
            target_x, target_pos, node_pp_index = self.compute_target(x, pos, pp_atom_indices, pp_positions, pp_types, pp_index, CoM_tensor)
            pp_info[filename].update({
                'pp_atom_indices': pp_atom_indices,
                'pp_positions': pp_positions,
                'pp_types': pp_types,
                'pp_index': pp_index,
                'node_pp_index': node_pp_index
            })

            x, node_mask = to_dense_batch(x, max_num_nodes=max_N)
            pos, _ = to_dense_batch(pos, max_num_nodes=max_N)
            target_x, _ = to_dense_batch(target_x, max_num_nodes=max_N)
            target_pos, _ = to_dense_batch(target_pos, max_num_nodes=max_N)
            node_pp_index, _ = to_dense_batch(node_pp_index, max_num_nodes=max_N)
            
            data = Data(x=x, pos=pos, target_x=target_x, target_pos=target_pos, CoM=CoM_tensor, node_mask=node_mask, edge_mask=edge_mask, num_nodes=num_nodes, node_pp_index=node_pp_index, ligand_name=filename)
            data_list.append(data)
        # data, slices = self.collate(data_list)
        # torch.save((data, slices), self.processed_paths[0])
        self.save(data_list, self.processed_paths[0])
        self.save_pp_info(pp_info)

    def extract_atom_features(self, mol, num_class, aromatic=False):
        '''
            calculate:
                h: encoding of atom type 
                types: atomic number
                atom_positions: 3D coordinates of atoms
                aromatic_features: whether the atom is aromatic (for reconstruction usage)
                types_with_aromatic: encoding of types + aromatic feature
        '''
        num_nodes = mol.GetNumAtoms()
        atom_type_mapping = ATOM_TYPE_MAPPING   # {'C': 0, 'N': 1, 'O': 2, 'F': 3, 'P': 4, 'S': 5, 'Cl': 6} not atomic number
        # aromatic_features = [1 for atom in mol.GetAtoms() if atom.GetIsAromatic() else 0]
        aromatic_features = [atom.GetIsAromatic() for atom in mol.GetAtoms()]
        h = [atom_type_mapping[atom.GetSymbol()] for atom in mol.GetAtoms()]
        types = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        conformer = mol.GetConformer()
        atom_positions = conformer.GetPositions()
        types_with_aromatic = [(type, aromatic) for type, aromatic in zip(types, aromatic_features)]
        types_with_aromatic_mapped = [MAP_ATOM_TYPE_AROMATIC_TO_INDEX[type_with_aromatic] for type_with_aromatic in types_with_aromatic]

        num_nodes_tensor = torch.tensor(num_nodes, dtype=torch.long)
        h_tensor = torch.tensor(np.array(h), dtype=torch.long)
        types_tensor = torch.tensor(np.array(types), dtype=torch.float)
        # one_hot_h_tensor = torch.nn.functional.one_hot(h_tensor, num_classes=len(atom_type_mapping.keys())).to(torch.float)
        one_hot_h_tensor = torch.nn.functional.one_hot(h_tensor, num_classes=num_class).to(torch.float)
        types_with_aromatic_tensor = torch.tensor(np.array(types_with_aromatic_mapped), dtype=torch.long)
        if aromatic:
            one_hot_types_with_aromatic_tensor = torch.nn.functional.one_hot(types_with_aromatic_tensor, num_classes=num_class).to(torch.float)
        else:
            one_hot_types_with_aromatic_tensor = None
        
        atom_positions_tensor = torch.tensor(np.array(atom_positions), dtype=torch.float)

        return one_hot_h_tensor, one_hot_types_with_aromatic_tensor, types_tensor, atom_positions_tensor, num_nodes_tensor
    
    def make_edge_mask(self, N, max_N=86):
        adj = torch.ones((N, N), dtype=torch.bool)
        adj[range(N), range(N)] = True      # I wrote False all the time?????
        edge_mask = torch.zeros((max_N, max_N), dtype=torch.bool)
        edge_mask[:N, :N] = adj
        edge_mask = edge_mask.view(1, max_N * max_N).float()
        edge_mask = edge_mask.bool()
        return edge_mask

    def extract_pp(self, ligand, num_class):
        pp_type_mapping = PP_TYPE_MAPPING

        atom_indice_list = []
        position_list = []
        pp_type_list = []
        pp_index_list = []
        
        for pp_node in ligand.graph.nodes:
            atom_indices = list([pp_node.atom_indices]) if type(pp_node.atom_indices)==int else list(sorted(pp_node.atom_indices))
            positions = pp_node.positions.squeeze()
            index = pp_node.index
            # types = [one_hot_encoding[type] for type in pp_node.types]
            types = pp_type_mapping[pp_node.types[0]]  # we can't have multiple types for one pharmacophore, so we just take the first one

            atom_indice_list.append(atom_indices)
            position_list.append(positions)
            pp_index_list.append(index)
            pp_type_list.append(types)

        # atom_indices_tensor = torch.tensor(atom_indice_list, dtype=torch.long)
        positions_tensor = torch.tensor(np.array(position_list), dtype=torch.float)
        # one_hot_pp_tensor = torch.nn.functional.one_hot(torch.tensor(pp_type_list, dtype=torch.long), num_classes=len(pp_type_mapping.keys())).to(torch.float)
        one_hot_pp_tensor = torch.nn.functional.one_hot(torch.tensor(np.array(pp_type_list), dtype=torch.long), num_classes=num_class).to(torch.float)
        pp_index_tensor = torch.tensor(np.array(pp_index_list), dtype=torch.long)

        return atom_indice_list, positions_tensor, one_hot_pp_tensor, pp_index_tensor

    def compute_CoM(self, pos, atomic_numbers):
        # compute the center of mass of the ligand
        periodic_table = Chem.GetPeriodicTable()
        # print(self.pos)
        sum_pos = torch.zeros(1, pos.size(1))
        sum_mass = 0
        for i, atomic_number in enumerate(atomic_numbers):
            mass = periodic_table.GetAtomicWeight(int(atomic_number))
            sum_pos += mass * pos[i]
            sum_mass += mass
        CoM = sum_pos / sum_mass
        return CoM
    
    def compute_pp_center(self, pp_positions):
        return torch.mean(pp_positions, dim=0)
    
    def CoM2zero(self, pos, CoM):
        # translate the ligand so that its center of mass is at the origin
        pos -= CoM
        return pos
    
    def compute_target(self, x, pos, pp_atom_indices, pp_positions, pp_types, pp_index, center_tensor, noise_std=0.01):
        '''
            Compute the target of the diffusion bridge, which is each atom's feat/pos destination regarding its pharmacophore membership
            Should we include a bit noise when initializing the target pos?
            TODO: We don't move the CoM to zero during data preparation now, should initiate the target pos as CoM rather than zeros!!! But then how do we init the target pos during sampling stage?
        '''

        target_x = torch.zeros(x.size(0), x.size(1))
        target_pos = torch.zeros(pos.size(0), pos.size(1))
        node_pp_index = torch.zeros(x.size(0), dtype=torch.long)
        atom_in_pp = []
        # print(self.pp_atom_indices)
        for atom_indices in pp_atom_indices:
            atom_in_pp += atom_indices
        for i in range(x.size(0)):
            if i not in atom_in_pp:  # if the atom is not in any pharmacophore, we set its target type to Linker:0 and target position to CoM plus a bit noise
                target_x[i] = torch.nn.functional.one_hot(torch.tensor([0]), num_classes=pp_types.size(1)).to(torch.float)
                # target_pos[i] = torch.zeros(pos.size(1))
                target_pos[i] = center_tensor + torch.randn_like(center_tensor) # * noise_std
                node_pp_index[i] = -1
            else:  # if the atom is in a pharmacophore, we set its target type to the pharmacophore type and target position to the pharmacophore position
                for j, atom_indices in enumerate(pp_atom_indices):
                    if i in atom_indices:
                        target_x[i] = pp_types[j]
                        target_pos[i] = pp_positions[j] + torch.randn_like(pp_positions[j]) # * noise_std
                        node_pp_index[i] = j    # = pp_index[j]
                        break
        
        # if args.augment_noise > 0:
        #     # Add noise eps ~ N(0, augment_noise) around points.
        #     eps = sample_center_gravity_zero_gaussian_with_mask(x.size(), x.device, node_mask)
        #     x = x + eps * args.augment_noise

        return target_x, target_pos, node_pp_index
    

class CombinedGraphDataset(PharmacophoreDataset):
    def __init__(self, root, split='train', transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        self._split = split
        self._max_N = 86 * 2
        super(CombinedGraphDataset, self).__init__(root, split, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        if self._split == 'train':
            return ['train_combined_graph.pt']
        elif self._split == 'test':
            return ['test_combined_graph.pt']
        elif self._split == 'valid':
            return ['valid_combined_graph.pt']
        elif self._split == 'all':
            return ['data_combined_graph.pt']
        else: 
            raise ValueError('split must be "train", "valid", "test" or "all"')
        
    def process(self):
        data_list = []
        # print(self.raw_file_names)
        # print(self.raw_paths)
        # max_N = self.get_max_N()
        max_N = self._max_N
        for raw_path in tqdm(self.raw_paths):
            filename = raw_path.split('/')[-1].split('.')[0]
            # print(raw_path)
            rdmol = Chem.MolFromMolFile(raw_path, sanitize=False)
            pbmol = next(pybel.readfile("sdf", raw_path))
            try:
                ligand = Ligand(pbmol, rdmol, atom_positions=None, conformer_axis=None)
            except Exception as e:
                print(raw_path, 'Ligand init failed')
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
                print(raw_path, 'extract pp failed')
                print(e)
                continue
            # print(raw_path, pos.size())

            # should we move CoM to zero during data 
            CoM_tensor = self.compute_CoM(pos, atomic_numbers)
            # pos = self.CoM2zero(pos, CoM)
            # pp_positions = self.CoM2zero(pp_positions, CoM)
            target_x, target_pos, node_pp_index = self.compute_target(x, pos, pp_atom_indices, pp_positions, pp_types, pp_index, CoM_tensor)
            x_ctr, pos_ctr, Gt_mask = self.combine_target(x, pos, target_x, target_pos, max_N)
            target_x_ctr, target_pos_ctr, _ = self.combine_target(target_x, target_pos, target_x, target_pos, max_N)
            edge_mask_ctr = self.make_edge_mask(num_nodes * 2, max_N)


            x_ctr, node_mask_ctr = to_dense_batch(x_ctr, max_num_nodes=max_N)
            target_x_ctr, _ = to_dense_batch(target_x_ctr, max_num_nodes=max_N)
            pos_ctr, _ = to_dense_batch(pos_ctr, max_num_nodes=max_N)
            target_pos_ctr, _ = to_dense_batch(target_pos_ctr, max_num_nodes=max_N)
            node_pp_index, _ = to_dense_batch(node_pp_index, max_num_nodes=max_N)
            
            data = Data(x=x_ctr, pos=pos_ctr, original_x=x, original_pos=pos, target_x=target_x_ctr, target_pos=target_pos_ctr, CoM=CoM_tensor, node_mask=node_mask_ctr, Gt_mask=Gt_mask, edge_mask=edge_mask_ctr, num_nodes=num_nodes, node_pp_index=node_pp_index, ligand_name=filename)
            data_list.append(data)
        # data, slices = self.collate(data_list)
        # torch.save((data, slices), self.processed_paths[0])
        self.save(data_list, self.processed_paths[0])

    def combine_target(self, x, pos, target_x, target_pos, max_N_combined):
        '''
            combine the target of the diffusion bridge with the node features and positions into a single graph
            the target is the destination of the diffusion bridge
            Input:
                x: node features
                pos: node positions
                target_x: target node features
                target_pos: target node positions
                max_N_combined: the maximum number of nodes in the combined graph (2 * max_N)
            Output:
                x: node features
                pos: node positions
                Gt_mask: mask for the nodes in Gt (excluding the nodes in GT)
        '''
        
        N = x.size(0)
        x = torch.cat((x, target_x), dim=0)
        pos = torch.cat((pos, target_pos), dim=0)
        Gt_mask = torch.zeros([max_N_combined], dtype=torch.bool)
        Gt_mask[:N] = True
        return x, pos, Gt_mask


class CombinedSparseGraphDataset(PharmacophoreDataset):
    def __init__(self, root, split='train', transform=None, pre_transform=None, pre_filter=None, aromatic=False):
        self.root = root
        self._split = split
        self.aromatic = aromatic
        # self._max_N = 86 * 2
        super(CombinedSparseGraphDataset, self).__init__(root, split, transform, pre_transform, pre_filter, aromatic=aromatic)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        assert self._split in ['train', 'valid', 'test', 'all']
        if self.aromatic:
            processed_file_name = f'{self._split}_aromatic_combined_sparse_graph.pt'
        else:
            processed_file_name = f'{self._split}_combined_sparse_graph.pt'

        return [processed_file_name]

        # if self._split == 'train':
        #     return ['train_combined_sparse_graph.pt']
        # elif self._split == 'test':
        #     return ['test_combined_sparse_graph.pt']
        # elif self._split == 'valid':
        #     return ['valid_combined_sparse_graph.pt']
        # elif self._split == 'all':
        #     return ['data_combined_sparse_graph.pt']
        # else: 
        #     raise ValueError('split must be "train", "valid", "test" or "all"')
        
    def process(self):
        data_list = []
        # print(self.raw_file_names)
        # print(self.raw_paths)
        # max_N = self.get_max_N()
        # max_N = self._max_N
        pp_info = defaultdict(dict)
        for raw_path in tqdm(self.raw_paths):
            filename = raw_path.split('/')[-1].split('.')[0]
            # print(raw_path)
            rdmol = Chem.MolFromMolFile(raw_path, removeHs=False, sanitize=True)
            pbmol = next(pybel.readfile("sdf", raw_path))
            # rdmol = Chem.AddHs(rdmol)
            try:
                rdmol = Chem.AddHs(rdmol)
                ligand = Ligand(pbmol, rdmol, atom_positions=None, conformer_axis=None)
                rdmol = ligand.rdmol_noH
            except Exception as e:
                print(f'Ligand {raw_path} init failed')
                print(e)
                continue
            # print('Ligand init success')
            # for atom in rdmol.GetAtoms():
            #     print(atom.GetSymbol())

            if self.aromatic:
                num_feat_class = max(len(PP_TYPE_MAPPING.keys()), len(MAP_ATOM_TYPE_AROMATIC_TO_INDEX.keys()))
            else:
                num_feat_class = max(len(PP_TYPE_MAPPING.keys()), len(ATOM_TYPE_MAPPING.keys()))
            try:
                x, x_aromatic, atomic_numbers, pos, num_nodes = self.extract_atom_features(rdmol, num_feat_class, aromatic=self.aromatic)
                # edge_mask = self.make_edge_mask(num_nodes)
            except KeyError as e:  # some elements are not considered, skip such ligands
                print(f'Ligand {raw_path} contains rare elements: {e}')
                continue
            try:
                pp_atom_indices, pp_positions, pp_types, pp_index = self.extract_pp(ligand, num_feat_class)
                assert pp_positions.size(1) == 3
            except Exception as e:
                print(raw_path, 'extract pp failed')
                print(e)
                continue

            # CoM_tensor = self.compute_CoM(pos, atomic_numbers)
            pp_center_tensor = self.compute_pp_center(pp_positions)
            # print(pp_center_tensor.size())
            assert pp_center_tensor.size(0) == 3

            if self.aromatic:
                feat = x_aromatic
            else:
                feat = x
            target_x, target_pos, node_pp_index = self.compute_target(feat, pos, pp_atom_indices, pp_positions, pp_types, pp_index, pp_center_tensor)
            x_ctr, pos_ctr, Gt_mask = self.combine_target(feat, pos, target_x, target_pos)
            target_x_ctr, target_pos_ctr, _ = self.combine_target(target_x, target_pos, target_x, target_pos)
            edge_mask_ctr = self.make_edge_mask(num_nodes * 2)
            node_mask_ctr = torch.ones([1, num_nodes * 2], dtype=torch.bool)

            if len(pp_info[filename].keys()) == 0:
                pp_info[filename].update({
                    'pp_atom_indices': [pp_atom_indices],
                    'pp_positions': [pp_positions],
                    'pp_types': [pp_types],
                    'pp_index': [pp_index],
                    'node_pp_index': [node_pp_index]
                })
            else:
                pp_info[filename]['pp_atom_indices'].append(pp_atom_indices)
                pp_info[filename]['pp_positions'].append(pp_positions)
                pp_info[filename]['pp_types'].append(pp_types)
                pp_info[filename]['pp_index'].append(pp_index)
                pp_info[filename]['node_pp_index'].append(node_pp_index)

            data = Data(x=x_ctr, pos=pos_ctr, target_x=target_x_ctr, target_pos=target_pos_ctr, Gt_mask=Gt_mask, ligand_name=filename)
            data_list.append(data)
        # data, slices = self.collate(data_list)
        # torch.save((data, slices), self.processed_paths[0])
        # self.data, self.slices = self.collate(data_list)
        # for data in data_list:
        #     print(data)
        self.save(data_list, self.processed_paths[0])
        # self.save()
        self.save_pp_info(pp_info, self._split)

    def combine_target(self, x, pos, target_x, target_pos):
        N = x.size(0)
        x = torch.cat((x, target_x), dim=0)
        pos = torch.cat((pos, target_pos), dim=0)
        Gt_mask = torch.zeros([N * 2], dtype=torch.bool)
        Gt_mask[:N] = True
        return x, pos, Gt_mask

    def make_edge_mask(self, N):
        edge_mask = torch.ones((N, N), dtype=torch.bool)
        edge_mask = edge_mask.view(1, N * N)
        return edge_mask


def load_dataset(module, root, split, aromatic=False):
    dataset = module(root=root, split=split, aromatic=aromatic)
    return dataset


if __name__ == '__main__':
    # train_dataset = PharmacophoreDataset(root='../../data/cleaned_crossdocked_data', split='train')
    # valid_dataset = PharmacophoreDataset(root='../../data/cleaned_crossdocked_data', split='valid')
    # test_dataset = PharmacophoreDataset(root='../../data/cleaned_crossdocked_data', split='test')

    # train_dataset = CombinedGraphDataset(root='../../data/cleaned_crossdocked_data', split='train')
    # valid_dataset = CombinedGraphDataset(root='../../data/cleaned_crossdocked_data', split='valid')
    # test_dataset = CombinedGraphDataset(root='../../data/cleaned_crossdocked_data', split='test')

    aromatic = sys.argv[1] == 'aromatic'
    print(f'aromatic: {aromatic}')

    module = CombinedSparseGraphDataset # CombinedGraphDataset # PharmacophoreDataset
    root = '../../data/cleaned_crossdocked_data'
    train_dataset = load_dataset(module, root, split='train', aromatic=aromatic)
    valid_dataset = load_dataset(module, root, split='valid', aromatic=aromatic)
    test_dataset = load_dataset(module, root, split='test', aromatic=aromatic)

    # to test on a few samples
    # root = '../../data/small_dataset'
    # dataset = load_dataset(module, root, split='all')