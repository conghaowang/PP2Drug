from rdkit import Chem
from openbabel import pybel
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.utils import to_dense_batch
import torch
import numpy as np
import glob
import os
import random
from tqdm import tqdm
from ligand import Ligand
from utils import ATOM_TYPE_MAPPING, PP_TYPE_MAPPING, ATOM_FAMILIES, MAP_ATOM_TYPE_AROMATIC_TO_INDEX


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
    def __init__(self, root, split='train', transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        self._split = split
        self._max_N = 86
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
            raise ValueError('split must be "train" or "test" or "all"')

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
               _, x, atomic_numbers, pos = self.extract_atom_features(rdmol, num_feat_class)
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
            target_x, target_pos = self.compute_target(x, pos, pp_atom_indices, pp_positions, pp_types)

            x, node_mask = to_dense_batch(x, max_num_nodes=max_N)
            pos, _ = to_dense_batch(pos, max_num_nodes=max_N)
            target_x, _ = to_dense_batch(target_x, max_num_nodes=max_N)
            target_pos, _ = to_dense_batch(target_pos, max_num_nodes=max_N)
            
            data = Data(x=x, pos=pos, target_x=target_x, target_pos=target_pos, CoM=CoM_tensor, node_mask=node_mask, ligand_name=filename)
            data_list.append(data)
        # data, slices = self.collate(data_list)
        # torch.save((data, slices), self.processed_paths[0])
        self.save(data_list, self.processed_paths[0])

    def extract_atom_features(self, mol, num_class):
        '''
            calculate:
                h: encoding of atom type 
                types: atomic number
                atom_positions: 3D coordinates of atoms
                aromatic_features: whether the atom is aromatic (for reconstruction usage)
                types_with_aromatic: encoding of types + aromatic feature
        '''
        atom_type_mapping = ATOM_TYPE_MAPPING   # {'C': 0, 'N': 1, 'O': 2, 'F': 3, 'P': 4, 'S': 5, 'Cl': 6} not atomic number
        # aromatic_features = [1 for atom in mol.GetAtoms() if atom.GetIsAromatic() else 0]
        aromatic_features = [atom.GetIsAromatic() for atom in mol.GetAtoms()]
        h = [atom_type_mapping[atom.GetSymbol()] for atom in mol.GetAtoms()]
        types = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        conformer = mol.GetConformer()
        atom_positions = conformer.GetPositions()
        types_with_aromatic = [(type, aromatic) for type, aromatic in zip(types, aromatic_features)]
        types_with_aromatic_mapped = [MAP_ATOM_TYPE_AROMATIC_TO_INDEX[type_with_aromatic] for type_with_aromatic in types_with_aromatic]

        h_tensor = torch.tensor(np.array(h), dtype=torch.long)
        types_tensor = torch.tensor(np.array(types), dtype=torch.float)
        one_hot_h_tensor = torch.nn.functional.one_hot(h_tensor, num_classes=len(atom_type_mapping.keys())).to(torch.float)
        types_with_aromatic_tensor = torch.tensor(np.array(types_with_aromatic_mapped), dtype=torch.long)
        one_hot_types_with_aromatic_tensor = torch.nn.functional.one_hot(types_with_aromatic_tensor, num_classes=num_class).to(torch.float)
        atom_positions_tensor = torch.tensor(np.array(atom_positions), dtype=torch.float)

        return one_hot_h_tensor, one_hot_types_with_aromatic_tensor, types_tensor, atom_positions_tensor

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
    
    def CoM2zero(self, pos, CoM):
        # translate the ligand so that its center of mass is at the origin
        pos -= CoM
        return pos
    
    def compute_target(self, x, pos, pp_atom_indices, pp_positions, pp_types):
        '''
            Compute the target of the diffusion bridge, which is each atom's feat/pos destination regarding its pharmacophore membership
            Should we include a bit noise when initializing the target pos?
        '''

        target_x = torch.zeros(x.size(0), x.size(1))
        target_pos = torch.zeros(pos.size(0), pos.size(1))
        atom_in_pp = []
        # print(self.pp_atom_indices)
        for atom_indices in pp_atom_indices:
            atom_in_pp += atom_indices
        for i in range(x.size(0)):
            if i not in atom_in_pp:  # if the atom is not in any pharmacophore, we set its target type to Linker:0 and target position to 0
                target_x[i] = torch.nn.functional.one_hot(torch.tensor([0]), num_classes=pp_types.size(1)).to(torch.float)
                target_pos[i] = torch.zeros(pos.size(1))
            else:  # if the atom is in a pharmacophore, we set its target type to the pharmacophore type and target position to the pharmacophore position
                for j, atom_indices in enumerate(pp_atom_indices):
                    if i in atom_indices:
                        target_x[i] = pp_types[j]
                        target_pos[i] = pp_positions[j]
                        break
        
        return target_x, target_pos
    

if __name__ == '__main__':
    train_dataset = PharmacophoreDataset(root='../../data/cleaned_crossdocked_data', split='train')
    valid_dataset = PharmacophoreDataset(root='../../data/cleaned_crossdocked_data', split='valid')
    test_dataset = PharmacophoreDataset(root='../../data/cleaned_crossdocked_data', split='test')