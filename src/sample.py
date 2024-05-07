import os
import argparse
import sys
sys.path.append('data_processing')
sys.path.append('model')
from torch_geometric.loader import DataLoader
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from rdkit import Chem
from tqdm import tqdm
from data_processing.paired_data import PharmacophoreDataset, CombinedGraphDataset
from data_processing.reconstruction import get_atomic_number_from_index, is_aromatic_from_index, reconstruct_from_generated
from model.pp_bridge import PPBridge
from model.pp_bridge_sampler import PPBridgeSampler
from script_utils import load_data


def reconstruct(x, h, Gt_mask, batch_info, ligand_names, save_path):
    num_graphs = max(batch_info).item() + 1
    success = 0
    for i in range(num_graphs):
        index_i = batch_info==i
        x_i = x[Gt_mask][index_i]
        h_i = h[Gt_mask][index_i]
        h_class = torch.argmax(h_i, dim=-1)
        atom_index = h_class.detach().cpu()
        atom_type = get_atomic_number_from_index(atom_index)
        atom_aromatic = is_aromatic_from_index(atom_index)
        pos = x_i.detach().cpu().tolist()
        try:
            mol = reconstruct_from_generated(pos, atom_type, atom_aromatic, basic_mode=False)
            mol_name = ligand_names[i]
            with Chem.SDWriter(os.path.join(save_path, mol_name + '.sdf')) as w:
                w.write(mol)
            success += 1
        except:
            continue
    # print(f'Successfully reconstructed {success}/{num_graphs} molecules In this batch')
    return success


def sample(config_file, ckpt_path, save_path, steps=40, device='cuda:0'):
    config = OmegaConf.load(config_file)
    save_path = os.path.join(save_path, config.model.denoiser.bridge_type)
    os.makedirs(save_path, exist_ok=True)
    sampler = PPBridgeSampler(config, ckpt_path, device)

    dataset_root_path = config.data.root # '/data/conghao001/pharmacophore2drug/PP2Drug/data/small_dataset' # config.data.root
    print(f'Loading data from {dataset_root_path}')
    datamodule = config.data.module
    test_dataset, test_loader = load_data(datamodule, dataset_root_path, split='test', batch_size=config.training.batch_size)

    success = 0
    for batch in tqdm(test_loader):
        batch = batch.to(device)
        with torch.no_grad():
            _, _, Gt_mask, batch_info = sampler.preprocess(batch.target_pos, batch.target_x, node_mask=batch.node_mask, Gt_mask=batch.Gt_mask, batch_info=batch.batch, device=device)  # Gt_mask and batch_info are for reconstruction
            x, x_traj, h, h_traj, nfe = sampler.sample(batch.target_pos, batch.target_x, steps, node_mask=batch.node_mask, Gt_mask=batch.Gt_mask, batch_info=batch.batch, 
                                                       sigma_min=config.model.denoiser.sigma_min, sigma_max=config.model.denoiser.sigma_max, churn_step_ratio=0.33, device=device)
        success += reconstruct(x, h, Gt_mask, batch.batch, batch.ligand_name, save_path)

    print(f'Successfully reconstructed {success}/{len(test_dataset)} molecules In total')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='config/vp_bridge.yml', help='Path to the configuration file')
    parser.add_argument('--ckpt', '-k', type=str, default='lightning_logs/vp_bridge_2024-05-05_23_23_05.637117/epoch=10-val_loss=1815365.00.ckpt', help='Path to the checkpoint file')
    parser.add_argument('--save', '-s', type=str, default='../generation_results', help='Path to save the reconstructed molecules')
    parser.add_argument('--steps', type=int, default=40, help='Number of steps for sampling')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='Which GPU to use')
    args = parser.parse_args()

    device = torch.device(f'cuda:{int(args.gpu)}' if torch.cuda.is_available() else 'cpu')
    sample(args.config, args.ckpt, args.save, int(args.steps), device)