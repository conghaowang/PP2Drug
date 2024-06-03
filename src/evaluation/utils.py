import os 
import numpy as np
import sys
sys.path.append('../')
from data_processing.utils import PP_TYPE_MAPPING


def extract_pp(ligand):
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

    '''
    # atom_indices_tensor = torch.tensor(atom_indice_list, dtype=torch.long)
    positions_tensor = torch.tensor(np.array(position_list), dtype=torch.float)
    # one_hot_pp_tensor = torch.nn.functional.one_hot(torch.tensor(pp_type_list, dtype=torch.long), num_classes=len(pp_type_mapping.keys())).to(torch.float)
    one_hot_pp_tensor = torch.nn.functional.one_hot(torch.tensor(np.array(pp_type_list), dtype=torch.long), num_classes=num_class).to(torch.float)
    pp_index_tensor = torch.tensor(np.array(pp_index_list), dtype=torch.long)
    '''
    
    positions_array = np.array(position_list)
    pp_type_array = np.array(pp_type_list, dtype=int)
    pp_index_array = np.array(pp_index_list, dtype=int)

    return atom_indice_list, positions_array, pp_type_array, pp_index_array


def center2zero(x, mean_dim=0):
    # if x == None:
    #     return None
    mean = np.mean(x, axis=mean_dim, keepdims=True)
    assert mean.shape[-1] == 3
    x = x - mean
    return x


def pp_match(pp_types, pp_positions, ref_pp_info, threshold=1.5):
    pps = pp_types
    ref_pps = np.argmax(ref_pp_info['pp_types'].numpy(), axis=-1)
    ref_pp_positions = ref_pp_info['pp_positions'].numpy()
    ref_pp_positions = center2zero(ref_pp_positions, mean_dim=0)
    match = np.zeros_like(ref_pps)
    for i, ref_pp in enumerate(ref_pps):
        ref_pos = ref_pp_positions[i]
        for j, pp in enumerate(pps):
            if pp == ref_pp:
                pos = pp_positions[j]
                dist = np.sqrt(np.sum((pos - ref_pos) ** 2))
                if dist < threshold:
                    match[i] += 1
    return match


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