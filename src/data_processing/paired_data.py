from rdkit import Chem
from openbabel import pybel
from torch_geometric.data import Data, Dataset, InMemoryDataset
import torch
import glob
import os
import random
from tqdm import tqdm
from ligand import Ligand


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
        super(PharmacophoreDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        all_docked = os.listdir(os.path.join(self.root, 'raw'))
        random.seed(2024)
        random.shuffle(all_docked)
        if self._split == 'train':
            selected_docked = all_docked[:int(len(all_docked)*0.8)]
        elif self._split == 'test':
            selected_docked = all_docked[int(len(all_docked)*0.8):]
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

    def process(self):
        data_list = []
        # print(self.raw_file_names)
        # print(self.raw_paths)
        for raw_path in tqdm(self.raw_paths):
            rdmol = Chem.MolFromMolFile(raw_path, sanitize=False)
            pbmol = next(pybel.readfile("sdf", raw_path))
            ligand = Ligand(pbmol, rdmol, atom_positions=None, conformer_axis=None)
            try:
               x, atomic_numbers, pos = self.extract_atom_features(rdmol)
            except KeyError:  # some elements are not considered, skip such ligands
                continue
            try:
                pp_atom_indices, pp_positions, pp_types, pp_index = self.extract_pp(ligand)
                assert pp_positions.size(1) == 3
            except Exception as e:
                print(raw_path, 'failed')
                print(e)
                continue
            # print(raw_path, pos.size())

            # should we move CoM to zero during data 
            CoM = self.compute_CoM(pos, atomic_numbers)
            pos = self.CoM2zero(pos, CoM)
            pp_positions = self.CoM2zero(pp_positions, CoM)
            target_x, target_pos = self.compute_target(x, pos, pp_atom_indices, pp_positions, pp_types)
            data = Data(x=x, pos=pos, target_x=target_x, target_pos=target_pos)
            data_list.append(data)
        # data, slices = self.collate(data_list)
        # torch.save((data, slices), self.processed_paths[0])
        self.save(data_list, self.processed_paths[0])

    def extract_atom_features(self, mol):
        one_hot_encoding = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'P': 5, 'S': 6, 'Cl': 7}    # 8 classes
        one_hot_x = [one_hot_encoding[atom.GetSymbol()] for atom in mol.GetAtoms()]
        types = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        conformer = mol.GetConformer()
        atom_positions = conformer.GetPositions()

        one_hot_x_tensor = torch.tensor(one_hot_x, dtype=torch.long)
        types_tensor = torch.tensor(types, dtype=torch.float)
        one_hot_x_tensor = torch.nn.functional.one_hot(one_hot_x_tensor, num_classes=len(one_hot_encoding.keys())).to(torch.float)
        atom_positions_tensor = torch.tensor(atom_positions, dtype=torch.float)

        return one_hot_x_tensor, types_tensor, atom_positions_tensor

    def extract_pp(self, ligand):
        one_hot_encoding = {
            'Linker': 0,
            'Hydrophobic': 1,
            'Aromatic': 2,
            'Cation': 3,
            'Anion': 4,
            'HBond_donor': 5,
            'HBond_acceptor': 6,
            'Halogen': 7
        }

        atom_indice_list = []
        position_list = []
        one_hot_pp_list = []
        pp_index_list = []
        
        for pp_node in ligand.graph.nodes:
            atom_indices = list([pp_node.atom_indices]) if type(pp_node.atom_indices)==int else list(sorted(pp_node.atom_indices))
            positions = pp_node.positions.squeeze()
            index = pp_node.index
            # types = [one_hot_encoding[type] for type in pp_node.types]
            types = one_hot_encoding[pp_node.types[0]]  # we can't have multiple types for one pharmacophore, so we just take the first one

            atom_indice_list.append(atom_indices)
            position_list.append(positions)
            pp_index_list.append(index)
            one_hot_pp_list.append(types)

        # atom_indices_tensor = torch.tensor(atom_indice_list, dtype=torch.long)
        positions_tensor = torch.tensor(position_list, dtype=torch.float)
        one_hot_pp_tensor = torch.nn.functional.one_hot(torch.tensor(one_hot_pp_list, dtype=torch.long), num_classes=len(one_hot_encoding.keys())).to(torch.float)
        pp_index_tensor = torch.tensor(pp_index_list, dtype=torch.long)

        return atom_indice_list, positions_tensor, one_hot_pp_tensor, pp_index_tensor

    def compute_CoM(self, pos, atomic_numbers):
        # compute the center of mass of the ligand
        periodic_table = Chem.GetPeriodicTable()
        # print(self.pos)
        sum_pos = torch.zeros(pos.size(1))
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
        # compute the target of the diffusion bridge, which is each atom's feat/pos destination regarding its pharmacophore membership
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
    test_dataset = PharmacophoreDataset(root='../../data/cleaned_crossdocked_data', split='test')