import os
import argparse
import sys
sys.path.append('data_processing')
sys.path.append('model')
import pickle
from torch_geometric.loader import DataLoader
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule
from tqdm import tqdm
from data_processing.paired_data import PharmacophoreDataset, CombinedGraphDataset
from data_processing.qm9_data import MAP_ATOM_TYPE_AROMATIC_TO_INDEX
from data_processing.utils import MAP_ATOMIC_NUMBER_TO_INDEX
from data_processing.reconstruction import get_atomic_number_from_index, is_aromatic_from_index, reconstruct_from_generated
from model.pp_bridge import PPBridge
from model.pp_bridge_sampler import PPBridgeSampler
from script_utils import load_data, load_qm9_data


@torch.no_grad()
def reconstruct(x, h, Gt_mask, batch_info, ligand_names, mol_save_path, datamodule='QM9Dataset', softmax_h=True, remove_H=False, basic_mode=False, optimization=True):
    if datamodule == 'QM9Dataset':
        index_to_atom_type_aromatic = {v: k for k, v in MAP_ATOM_TYPE_AROMATIC_TO_INDEX.items()}
    num_graphs = max(batch_info).item() + 1
    success = 0
    for i in tqdm(range(num_graphs)):
        index_i = (batch_info == i)
        x_i = x[index_i][Gt_mask[index_i]]
        h_i = h[index_i][Gt_mask[index_i]]
        if softmax_h:
            h_i = F.softmax(h_i, dim=-1)
        if remove_H:
            h_i = h_i[:, :-1]
        h_class = torch.argmax(h_i, dim=-1)
        atom_index = h_class.detach().cpu()
        if datamodule == 'QM9Dataset':
            # TODO: consider when qm9 data has no aromatic information, currently all qm9 data has aromatic information. 
            # but we can't differentiate the data by feat_size since they are all 8
            atom_type = get_atomic_number_from_index(atom_index, index_to_atom_type=index_to_atom_type_aromatic)
            atom_aromatic = is_aromatic_from_index(atom_index, index_to_atom_type=index_to_atom_type_aromatic)
        else:
            if basic_mode: 
                # there are actually two conditions in basic mode: one is the data already have no aromatic information (feat_size=8), the other is we dont consider aromaticity
                if x.size(-1) == 8:
                    index_to_atom_type = MAP_ATOMIC_NUMBER_TO_INDEX
                    atom_type = get_atomic_number_from_index(atom_index, index_to_atom_type=index_to_atom_type)
                    atom_aromatic = None

                # consider when we have aromaticity in the data but we dont want to consider it 
                else:
                    atom_type = get_atomic_number_from_index(atom_index)
                    atom_aromatic = None
            else:
                atom_type = get_atomic_number_from_index(atom_index)
                atom_aromatic = is_aromatic_from_index(atom_index)
        pos = x_i.detach().cpu().tolist()
        try:
            mol = reconstruct_from_generated(pos, atom_type, atom_aromatic, basic_mode=basic_mode)
            mol_name = ligand_names[i]
            if optimization:
                mol = Chem.AddHs(mol, addCoords = True)
                AllChem.EmbedMolecule(mol)
                UFFOptimizeMolecule(mol)
            with Chem.SDWriter(os.path.join(mol_save_path, mol_name + '.sdf')) as w:
                w.write(mol)
            success += 1
        except:
            continue
    # print(f'Successfully reconstructed {success}/{num_graphs} molecules In this batch')
    return success


def sample(config_file, ckpt_path, save_path, steps=40, device='cuda:0', remove_H=False, basic_mode=False, optimization=True):
    config = OmegaConf.load(config_file)
    # save_path = os.path.join(save_path, config.model.denoiser.bridge_type)
    os.makedirs(save_path, exist_ok=True)
    if not basic_mode:
        rec_mol_path = os.path.join(save_path, 'reconstructed_mols_aromatic_mode_optimized' if optimization else 'reconstructed_mols_aromatic_mode')
        gen_res_file = os.path.join(save_path, 'generation_res_aromatic_mode_optimized.pkl' if optimization else 'generation_res_aromatic_mode.pkl')
    else:
        rec_mol_path = os.path.join(save_path, 'reconstructed_mols_optimized' if optimization else 'reconstructed_mols')
        gen_res_file = os.path.join(save_path, 'generation_res_optimized.pkl' if optimization else 'generation_res.pkl')
    os.makedirs(rec_mol_path, exist_ok=True)
    sampler = PPBridgeSampler(config, ckpt_path, device)

    dataset_root_path = config.data.root # '/data/conghao001/pharmacophore2drug/PP2Drug/data/small_dataset' # config.data.root
    # print(f'Loading data from {dataset_root_path}')
    # print((not basic_mode))
    datamodule = config.data.module
    if datamodule == 'QM9Dataset':
        test_dataset, test_loader = load_qm9_data(root=dataset_root_path, split='test', batch_size=config.sampling.batch_size)
    else:
        test_dataset, test_loader = load_data(datamodule, dataset_root_path, split='test', batch_size=config.sampling.batch_size, aromatic=config.data.aromatic)

    print(f'Loading data from {test_dataset.processed_paths[0]}')
    success = 0
    # all_x, all_x_traj, all_h, all_h_traj, all_nfe = [], [], [], [], []
    for batch in tqdm(test_loader):
        batch = batch.to(device)
        if datamodule == 'CombinedSparseGraphDataset' or datamodule == 'QM9Dataset' or datamodule == 'CombinedUnconditionalDataset':
            node_mask = torch.ones([1, batch.x.size(0)], dtype=torch.bool, device=device)
            # ligand_names = batch.smiles
        else:
            node_mask = batch.node_mask
            # ligand_names = batch.ligand_name
        with torch.no_grad():
            _, _, Gt_mask, batch_info = sampler.preprocess(batch.target_pos, batch.target_x, node_mask=node_mask, Gt_mask=batch.Gt_mask, batch_info=batch.batch, device=device)  # Gt_mask and batch_info are for reconstruction
            x, x_traj, h, h_traj, nfe = sampler.sample(batch.target_pos, batch.target_x, steps, node_mask=node_mask, Gt_mask=batch.Gt_mask, batch_info=batch.batch, 
                                                       sigma_min=config.data.feat.sigma_min, sigma_max=config.data.feat.sigma_max, churn_step_ratio=0., device=device)
        success += reconstruct(x, h, Gt_mask, batch.batch, batch.ligand_name, rec_mol_path, datamodule, remove_H=remove_H, basic_mode=basic_mode, optimization=optimization)
    #     all_x.append(x)
    #     all_x_traj.append(x_traj)
    #     all_h.append(h)
    #     all_h_traj.append(h_traj)
    #     all_nfe.append(nfe)
    # with open(gen_res_file, 'wb') as f:
    #     pickle.dump({
    #         'x': all_x,
    #         'x_traj': all_x_traj,
    #         'h': all_h,
    #         'h_traj': all_h_traj,
    #         'nfe': all_nfe
    #     }, f)

    print(f'Successfully reconstructed {success}/{len(test_dataset)} molecules In total')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='config/vp_bridge.yml', help='Path to the configuration file')
    parser.add_argument('--ckpt', '-k', type=str, default='lightning_logs/vp_bridge_2024-05-05_23_23_05.637117/epoch=10-val_loss=1815365.00.ckpt', help='Path to the checkpoint file')
    parser.add_argument('--save', '-s', type=str, default='../generation_results', help='Path to save the reconstructed molecules')
    parser.add_argument('--steps', type=int, default=200, help='Number of steps for sampling')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='Which GPU to use')
    parser.add_argument('--remove_H', '-r', action='store_false', help='Whether to remove Hydrogens in the reconstruction')
    parser.add_argument('--basic_mode', '-b', action='store_true', help='Whether to use the basic mode for reconstruction, dont add this if you want to consider aromaticity')
    parser.add_argument('--no_optimization', '-no_opt', action='store_false', help='Whether to optimize the generated molecules')
    
    args = parser.parse_args()

    device = torch.device(f'cuda:{int(args.gpu)}' if torch.cuda.is_available() else 'cpu')
    sample(args.config, args.ckpt, args.save, int(args.steps), device, remove_H=args.remove_H, basic_mode=args.basic_mode, optimization=args.no_optimization)