import os
import torch
from torch_geometric.data import Data
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import sys
sys.path.append('../')
from data_processing.utils import ATOM_TYPE_MAPPING, PP_TYPE_MAPPING, ATOM_FAMILIES, MAP_ATOM_TYPE_AROMATIC_TO_INDEX
from data_processing.paired_data import CombinedSparseGraphDataset


def group_by(mol, ligand, level='pp'):
    import py3Dmol
    from rdkit.Chem.Draw import IPythonConsole
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import rdMolDraw2D
    from IPython.display import SVG, Image
    IPythonConsole.drawOptions.addAtomIndices = True
    IPythonConsole.ipython_useSVG = True
    IPythonConsole.molSize = 600, 600

    my_cmap = matplotlib.colormaps['coolwarm']
#     my_cmap = cm.get_cmap('coolwarm')
    if level == 'cluster':
        n_group = len(ligand.graph.node_clusters)
    elif level == 'pp':
        n = 0
        for cluster in ligand.graph.node_clusters:
            n += cluster.positions.shape[1]
        n_group = n
        # pp_id = 0
        
    my_norm = Normalize(vmin=0, vmax=n_group)
    atommap, bondmap = {}, {}
    for i in range(len(ligand.graph.node_clusters)):
        cluster = ligand.graph.node_clusters[i]
        for node in cluster.nodes:
#             node = cluster.nodes[pp_id]
            atom_idx = node.atom_indices
            if level == 'cluster':
                for atom_id in atom_idx:
                    atom = mol.GetAtoms()[atom_id]
                    atom.SetProp("atomNote", str(i))
                atommap.update({atom_id:my_cmap(my_norm(i))[:3] for atom_id in atom_idx})
            elif level == 'pp':
                for atom_id in atom_idx:
                    atom = mol.GetAtoms()[atom_id]
                    atom.SetProp("atomNote", str(node.index))
                atommap.update({atom_id:my_cmap(my_norm(node.index))[:3] for atom_id in atom_idx})
                # pp_id += 1
                
    highlights = {
        "highlightAtoms": list(atommap.keys()),
        "highlightAtomColors": atommap,
        "highlightBonds": list(bondmap.keys()),
        "highlightBondColors": bondmap,
    }
    mol_ = rdMolDraw2D.PrepareMolForDrawing(mol)

    # imgsize = (600, 300)
    # drawer = rdMolDraw2D.MolDraw2DSVG(*imgsize)
    # drawer.DrawMolecule(mol_, **highlights)
    # drawer.FinishDrawing()
    # svg = drawer.GetDrawingText()
    # display(SVG(svg.replace('svg:','')))

    return mol_, highlights


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
        print(pp_node.types, positions)
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


def extract_all_pp(ligand):
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
        # print(pp_node.types, positions)
        if len(pp_node.types) > 1:
            # print('Multiple types:', pp_node.types)
            types = [pp_type_mapping[type] for type in pp_node.types]
        else:
            types = pp_type_mapping[pp_node.types[0]]  # take the first one

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
    # pp_type_array = np.array(pp_type_list, dtype=int)
    pp_index_array = np.array(pp_index_list, dtype=int)

    return atom_indice_list, positions_array, pp_type_list, pp_index_array


def extract_selected_pp(selected_pp, num_class):
    pp_type_mapping = PP_TYPE_MAPPING

    atom_indice_list = []
    position_list = []
    pp_type_list = []
    pp_index_list = []
    
    for pp_node in selected_pp:
        atom_indices = list([pp_node.atom_indices]) if type(pp_node.atom_indices)==int else list(sorted(pp_node.atom_indices))
        positions = pp_node.positions.squeeze()
        index = pp_node.index
        # types = [one_hot_encoding[type] for type in pp_node.types]
        types = pp_type_mapping[pp_node.types[-1]]  # we can't have multiple types for one pharmacophore, so we just take the first one

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


def compute_target(x, pos, pp_atom_indices, pp_positions, pp_types, pp_index, center_tensor, noise_std=0.01):
        '''
            Compute the target of the diffusion bridge, which is each atom's feat/pos destination regarding its pharmacophore membership
            Should we include a bit noise when initializing the target pos?
            TODO: We don't move the CoM to zero during data preparation now, should initiate the target pos as CoM rather than zeros!!! But then how do we init the target pos during sampling stage?
        '''

        target_x = torch.zeros(x.size(0), x.size(1))
        target_pos = torch.zeros(pos.size(0), pos.size(1))
        node_pp_index = torch.zeros(x.size(0), dtype=torch.long)
        atom_in_pp = []

        for atom_indices in pp_atom_indices:
            atom_in_pp += atom_indices
        
        if len(atom_in_pp) != x.size(0):
            # some atoms are not in any pharmacophore, we cluster them and set their target pos to the cluster center
            non_pp_atom_indices, non_pp_group_center_positions = CombinedSparseGraphDataset.cluster_non_pp(pos, atom_in_pp)
        else:
            # all atoms are in pharmacophores
            non_pp_atom_indices = {}
            non_pp_group_center_positions = None
        for i in range(x.size(0)):
            if i not in atom_in_pp:  # if the atom is not in any pharmacophore, we set its target type to Linker:0 and target position to CoM plus a bit noise
                target_x[i] = torch.nn.functional.one_hot(torch.tensor([0]), num_classes=pp_types.size(1)).to(torch.float)
                # target_pos[i] = torch.zeros(pos.size(1))
                # target_pos[i] = center_tensor + torch.randn_like(center_tensor) # * noise_std
                node_pp_index[i] = -1
                for j, atom_indices in non_pp_atom_indices.items():
                    if i in atom_indices:
                        target_pos[i] = non_pp_group_center_positions[j] + torch.randn_like(non_pp_group_center_positions[j]) * noise_std
                        break
            else:  # if the atom is in a pharmacophore, we set its target type to the pharmacophore type and target position to the pharmacophore position
                for j, atom_indices in enumerate(pp_atom_indices):
                    if i in atom_indices:
                        target_x[i] = pp_types[j]
                        target_pos[i] = pp_positions[j] + torch.randn_like(pp_positions[j]) * noise_std
                        node_pp_index[i] = j    # = pp_index[j]
                        break
        
        # if args.augment_noise > 0:
        #     # Add noise eps ~ N(0, augment_noise) around points.
        #     eps = sample_center_gravity_zero_gaussian_with_mask(x.size(), x.device, node_mask)
        #     x = x + eps * args.augment_noise

        return target_x, target_pos, node_pp_index

def process_one(aromatic, rdmol, selected_pp, filename):
    if aromatic:
        num_feat_class = max(len(PP_TYPE_MAPPING.keys()), len(MAP_ATOM_TYPE_AROMATIC_TO_INDEX.keys()))
    else:
        num_feat_class = max(len(PP_TYPE_MAPPING.keys()), len(ATOM_TYPE_MAPPING.keys()))
    try:
        x, x_aromatic, atomic_numbers, pos, num_nodes = CombinedSparseGraphDataset.extract_atom_features(rdmol, num_feat_class, aromatic=aromatic)

    except KeyError as e:  # some elements are not considered, skip such ligands
        print(f'Ligand contains rare elements: {e}')
    try:
        pp_atom_indices, pp_positions, pp_types, pp_index = extract_selected_pp(selected_pp, num_feat_class)
        assert pp_positions.size(1) == 3
    except Exception as e:
        print('extract pp failed')
        print(e)
    
    pp_center_tensor = CombinedSparseGraphDataset.compute_pp_center(pp_positions)
    # print(pp_center_tensor.size())
    assert pp_center_tensor.size(0) == 3
    
    if aromatic:
        feat = x_aromatic
    else:
        feat = x

    target_x, target_pos, node_pp_index = compute_target(feat, pos, pp_atom_indices, pp_positions, pp_types, pp_index, pp_center_tensor)
    x_ctr, pos_ctr, Gt_mask = CombinedSparseGraphDataset.combine_target(feat, pos, target_x, target_pos)
    target_x_ctr, target_pos_ctr, _ = CombinedSparseGraphDataset.combine_target(target_x, target_pos, target_x, target_pos)

    data = Data(x=x_ctr, pos=pos_ctr, target_x=target_x_ctr, target_pos=target_pos_ctr, Gt_mask=Gt_mask, ligand_name=filename)

    return data


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

    # TODO (fixed): in sampling, we compute the center as the mean of the positions of all atoms' pp or non-pp clusters.
    # But here we compute the center as the mean of the positions of all pp clusters (not repeated for atoms).
    # Solution: we now center the pp_positions in the main script
    # ref_pp_positions = center2zero(ref_pp_positions, mean_dim=0)

    match = np.zeros_like(ref_pps)
    for i, ref_pp in enumerate(ref_pps):
        ref_pos = ref_pp_positions[i]
        for j, pp in enumerate(pps):
            if type(pp) == list:
                for pp_ in pp:
                    if pp_ == ref_pp:
                        pos = pp_positions[j]
                        dist = np.sqrt(np.sum((pos - ref_pos) ** 2))
                        if dist < threshold:
                            match[i] = 1
                            continue
                            # match[i] += 1
            else:
                if pp == ref_pp:
                    pos = pp_positions[j]
                    dist = np.sqrt(np.sum((pos - ref_pos) ** 2))
                    if dist < threshold:
                        match[i] = 1
                        continue
                        # match[i] += 1
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


def save_matching_scores(match_dict, score_dict, save_path):
    with open(save_path + '_matches.pkl', 'wb') as f:
        pickle.dump(match_dict, f)
        # pickle.dump(score_dict, f)
    score_df = pd.DataFrame.from_dict(score_dict, orient='index', columns=['score'])
    score_df.to_csv(save_path + '_scores.csv')


def plot_matching_scores(score_dict, save_path):
    scores = np.array([v for v in score_dict.values()])
    scores = np.clip(scores, 0, 1)
    ax = sns.histplot(scores, bins=15)
    # ax.bar_label('{:.2f} %'.format(ax.containers[0]))
    ax.set_title('Pharmacophore Matching Scores')
    ax.set_xlabel('Matching Score')
    ax.set_ylabel('Frequency')
    plt.savefig(save_path + '_score_dist.png', dpi=300)