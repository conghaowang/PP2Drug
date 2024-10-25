import os
import argparse
import sys
sys.path.append('../data_processing')
sys.path.append('../model')
sys.path.append('../')
import pickle
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
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
from sample import reconstruct

def find_data(ligand, config):
    dataset_root_path = os.path.join('..', config.data.root) # '/data/conghao001/pharmacophore2drug/PP2Drug/data/small_dataset' # config.data.root
    datamodule = config.data.module
    if datamodule == 'QM9Dataset':
        test_dataset, test_loader = load_qm9_data(root=dataset_root_path, split='test', batch_size=config.sampling.batch_size)
    else:
        test_dataset, test_loader = load_data(datamodule, dataset_root_path, split='test', batch_size=config.sampling.batch_size, aromatic=config.data.aromatic)

    for data in tqdm(test_dataset):
        if data.ligand_name == ligand:
            break
    
    assert data.ligand_name == ligand
    return data


def sample_one(ligand, config_file, ckpt_path, root_path, bridge_type, num_samples=100, steps=40, device='cuda:0', remove_H=False, basic_mode=False, optimization=True):
    config = OmegaConf.load(config_file)
    save_path = os.path.join(root_path, bridge_type)
    os.makedirs(save_path, exist_ok=True)

    # folder_name = ligand.split('.')[0]
    pdb_id = ligand[ligand.rfind('rec')+4:ligand.rfind('rec')+8] # ligand.split('.')[0]
    # if not basic_mode:
    #     rec_mol_path = os.path.join(save_path, ligand, 'aromatic')
    # else:
    #     rec_mol_path = os.path.join(save_path, ligand, 'basic')
    rec_mol_path = os.path.join(save_path, ligand, 'basic' if basic_mode else 'aromatic')
    rec_mol_path += '_optimized' if optimization else ''
    os.makedirs(rec_mol_path, exist_ok=True)
    gen_res_file = rec_mol_path + '.pkl'
    sampler = PPBridgeSampler(config, ckpt_path, device)

    # we should keep basic_mode consistent with the data. currently if the data has no aromatic information, we actually use the basic_mode no matter what the cmd argument is
    if config.data.aromatic:
        data_path = os.path.join(root_path, ligand, ligand + '_aromatic.pt')
    else:
        data_path = os.path.join(root_path, ligand, ligand + '.pt')

    data = torch.load(data_path)
    # data = find_data(ligand, config)
    data_list = [data for _ in range(num_samples)]
    batch = Batch.from_data_list(data_list)

    success = 0
    batch = batch.to(device)
    node_mask = torch.ones([1, batch.x.size(0)], dtype=torch.bool, device=device)
    all_x, all_h = [], []
    with torch.no_grad():
        _, _, Gt_mask, batch_info = sampler.preprocess(batch.target_pos, batch.target_x, node_mask=node_mask, Gt_mask=batch.Gt_mask, batch_info=batch.batch, device=device)  # Gt_mask and batch_info are for reconstruction
        x, x_traj, h, h_traj, nfe = sampler.sample(batch.target_pos, batch.target_x, steps, node_mask=node_mask, Gt_mask=batch.Gt_mask, batch_info=batch.batch, 
                                                    sigma_min=config.data.feat.sigma_min, sigma_max=config.data.feat.sigma_max, churn_step_ratio=0., device=device)
        
    all_x += x
    all_h += h
    with open(gen_res_file, 'wb') as f:
        pickle.dump({
            'x': x.cpu(),
            'h': h.cpu()
        }, f)
    save_names = [f'{pdb_id}_{i}' for i in range(num_samples)]
    success += reconstruct(x, h, Gt_mask, batch.batch, save_names, rec_mol_path, 'CombinedSparseGraphDataset', remove_H=remove_H, basic_mode=basic_mode, optimization=optimization, covalent_factor=2.0)
    print(f'Successfully reconstructed {success}/{num_samples} molecules In total')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', '-n', type=int, default=10, help='Number of samples to generate')
    parser.add_argument('--ligand', '-l', type=str, required=True, help='Ligand file to generate from')
    parser.add_argument('--config', '-c', type=str, default='/home2/conghao001/pharmacophore2drug/PP2Drug/src/lightning_logs/vp_bridge_egnn_CombinedSparseGraphDataset_2024-08-19_21_05_04.140916/vp_bridge_egnn.yml', help='Path to the configuration file')
    parser.add_argument('--ckpt', '-k', type=str, default='/home2/conghao001/pharmacophore2drug/PP2Drug/src/lightning_logs/vp_bridge_egnn_CombinedSparseGraphDataset_2024-08-19_21_05_04.140916/epoch=166-val_loss=75.36.ckpt', help='Path to the checkpoint file')
    parser.add_argument('--save', '-s', type=str, default='structure_based', help='Path to save the reconstructed molecules')
    parser.add_argument('--bridge_type', '-bt', type=str, default='vp', help='Type of bridge to use')
    parser.add_argument('--steps', type=int, default=200, help='Number of steps for sampling')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='Which GPU to use')
    parser.add_argument('--remove_H', '-r', action='store_false', help='Whether to remove Hydrogens in the reconstruction')
    parser.add_argument('--basic_mode', '-b', action='store_true', help='Whether to use the basic mode for reconstruction, dont add this if you want to consider aromaticity')
    parser.add_argument('--no_optimization', '-no_opt', action='store_false', help='Whether to optimize the generated molecules')

    args = parser.parse_args()
    device = torch.device(f'cuda:{int(args.gpu)}' if torch.cuda.is_available() else 'cpu')
    print(f'Optimization: {args.no_optimization}')
    sample_one(args.ligand, args.config, args.ckpt, args.save, args.bridge_type, num_samples=args.num_samples, steps=args.steps, device=device, remove_H=args.remove_H, basic_mode=args.basic_mode, optimization=args.no_optimization)