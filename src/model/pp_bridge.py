# import pytorch_lightning as pl
import lightning as pl
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR, ExponentialLR
import wandb
from omegaconf import OmegaConf
import copy
import os
import sys
sys.path.append('..')

from model.EGNN_backbone import EGNN, EGNN_combined_graph
from model.SE3_transformer_backbone import SE3Transformer
from model.utils.time_scheduler import UniformSampler, RealUniformSampler
from model.utils.utils_diffusion import append_dims, vp_logs, vp_logsnr, mean_flat, scatter_mean_flat, scatter_flat, center2zero, center2zero_with_mask, center2zero_combined_graph, center2zero_sparse_graph, sample_zero_center_gaussian, sample_zero_center_gaussian_with_mask
from script_utils import instantiate_from_config

class PPBridge(pl.LightningModule):
    def __init__(
            self, 
            config, 
            sigma_data: float = 0.5,
            sigma_max=80.0,
            sigma_min=0.002,
            beta_d=2,
            beta_min=0.1,
            cov_xy=0., # 0 for uncorrelated, sigma_data**2 / 2 for  C_skip=1/2 at sigma_max
            rho=7.0,
            feature_size=64,
            weight_schedule="karras",
            bridge_type='both',
            resume_step=0,
            device='cuda',
        ):
        super().__init__()
        model_config = config['model']['denoiser']
        backbone_config = config['model']['backbone']
        training_config = config['training']
        data_config = config['data']

        # self.sigma_data = model_config['sigma_data']
        # self.sigma_max = model_config['sigma_max'] 
        # self.sigma_min = model_config['sigma_min']
        self.sigma_data_feat = data_config['feat']['sigma_data']
        self.sigma_data_pos = data_config['pos']['sigma_data']
        self.sigma_data_end_feat = data_config['feat']['sigma_data_end']
        self.sigma_data_end_pos = data_config['pos']['sigma_data_end']

        self.sigma_max_feat = data_config['feat']['sigma_max']
        self.sigma_max_pos = data_config['pos']['sigma_max']
        self.sigma_min_feat = data_config['feat']['sigma_min']
        self.sigma_min_pos = data_config['pos']['sigma_min']

        self.beta_d = model_config['beta_d']
        self.beta_min = model_config['beta_min']
        # self.sigma_data_end_feat = self.sigma_data_feat
        # self.sigma_data_end_pos = self.sigma_data_pos
        # self.cov_xy = cov_xy
        self.cov_xy_feat = self.sigma_data_feat**2 / 2
        self.cov_xy_pos = self.sigma_data_pos**2 / 2
        self.c = 1

        self.weight_schedule = model_config['weight_schedule']
        self.bridge_type = model_config['bridge_type']    # previous pred_mode

        self.datamodule = data_config['module']
        if self.datamodule.startswith('Combined') or self.datamodule == 'QM9Dataset':
            '''
                xT_type | xT_mode       | effect
                none    | none          | single graph of x0: xT is not input into backbone, but it is still used to calculate bridge scalings
                noise   | none          | single graph of x0 (same as above)
                noise   | concat_graph  | x0 + noise
                pp      | concat_graph  | x0 + pp
            '''
            # self.xT_type = 'none'
            self.xT_type = backbone_config['xT_type']    # type of xT, noise or target pp or none
            self.xT_mode = backbone_config['xT_mode']
            if self.xT_type == 'none':
                assert self.xT_mode == 'none'
            elif self.xT_type == 'pp':
                assert self.xT_mode == 'concat_graph'
        else:
            self.xT_type = backbone_config['xT_type']    # type of xT, noise or target pp or none
            self.xT_mode = backbone_config['xT_mode']    # mode of including xT, concat_graph or concat_latent or none. TODO: remove concat_latent, it's not equivariant
        if self.xT_type == 'none':
            assert self.xT_mode == 'none'

        self.rho = rho
        self.num_timesteps = 40     # for uniform time scheduler, not used now
        self.feature_size = backbone_config['feature_size']
        self.lr = training_config['learning_rate']
        self.lr_anneal_steps = training_config['lr_anneal_steps']
        self.use_lr_scheduler = training_config['use_lr_scheduler']
        self.lr_scheduler_config = training_config['lr_scheduler_config']
        if self.use_lr_scheduler:
            print('Using lr scheduler')
            assert self.lr_scheduler_config is not None

        if backbone_config['type'] == 'EGNN':
            if self.datamodule.startswith('Combined') or self.datamodule == 'QM9Dataset':
                # self.backbone = EGNN(
                #     max_feat_num=backbone_config['feature_size'], 
                #     coord_dim=data_config['coord_dim'],
                #     n_layers=backbone_config['num_layers'], 
                #     hidden_size=backbone_config['hidden_size'], 
                #     max_node_num=data_config['max_node_num'] * 2,    # Gt and GT are already integrated
                #     include_charge=False, 
                #     time_cond=backbone_config['time_cond'],
                #     xT_mode=self.xT_mode,
                #     data_mode='combined',
                # )
                self.backbone = EGNN_combined_graph(
                    num_layers=backbone_config['num_layers'],
                    input_dim=backbone_config['feature_size'],
                    hidden_dim=backbone_config['hidden_size'],
                    edge_feat_dim=4,
                    num_r_gaussian=1,
                    time_cond=backbone_config['time_cond'],
                    norm=True,
                )
            else:
                self.backbone = EGNN(
                        max_feat_num=backbone_config['feature_size'], 
                        coord_dim=data_config['coord_dim'],
                        n_layers=backbone_config['num_layers'], 
                        hidden_size=backbone_config['hidden_size'], 
                        max_node_num=data_config['max_node_num'], 
                        include_charge=False, 
                        time_cond=backbone_config['time_cond'],
                        xT_mode=self.xT_mode,
                        data_mode='single',
                    )
        elif backbone_config['type'] == 'SE3Transformer':
            self.backbone = SE3Transformer(
                num_blocks=backbone_config['num_blocks'],
                num_layers=backbone_config['num_layers'],
                input_dim=backbone_config['feature_size'],
                hidden_dim=backbone_config['hidden_size'],
                n_heads=backbone_config['num_heads'],
                k=backbone_config['knn'],
                edge_feat_dim=4,
                num_r_gaussian=backbone_config['num_r_gaussian'],
                num_node_types=backbone_config['num_node_types'],
                act_fn=backbone_config['act_fn'],
                norm=backbone_config['norm'],
                cutoff_mode=backbone_config['cutoff_mode'],
                ew_net_type=backbone_config['ew_net_type'],
                num_x2h=backbone_config['num_x2h'],
                num_h2x=backbone_config['num_h2x'],
                r_max=backbone_config['r_max'],
                x2h_out_fc=backbone_config['x2h_out_fc'],
                sync_twoup=backbone_config['sync_twoup'],
                time_cond=backbone_config['time_cond'],
            )
        else:
            raise NotImplementedError(f"Backbone type {backbone_config['type']} not implemented")

        self.use_ema = model_config['use_ema']
        self.ema_decay = model_config['ema_decay']
        self.weight_decay = model_config['weight_decay']
        self.loss_x_weight = model_config['loss_x_weight']

        self.batch_size = training_config['batch_size']
        self.log_interval = training_config['log_interval']
        self.save_interval = training_config['save_interval']
        self.test_interval = training_config['test_interval']
        self.total_training_steps = training_config['total_training_steps']

        self.save_hyperparameters(ignore=['backbone'])
        # try:
        #     OmegaConf.save(config=config, f=os.path.join(self.trainer.log_dir, self.bridge_type+'.yml'))
        # except Exception as e:
        #     print('config not saved since', e)
        # resuming from checkpoint or any step has not been implemented yet
        self.resume_checkpoint = training_config.get('resume_checkpoint', None)
        self._resume_checkpoint()
        self.resume_step = resume_step

        self.optimizer = torch.optim.RAdam(self.backbone.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.resume_step:
            self._load_optimizer_state()
            self._load_ema()
            raise NotImplementedError("Optimizer state loading not implemented")
        else:
            self.ema_params = copy.deepcopy(list(self.backbone.parameters()))

        # self.generator = None    # generator is not implemented here, this model is only for training
        scheduler_type = model_config['schedule_sampler']
        if scheduler_type == 'uniform':
            print('Using uniform sampler')
            self.time_scheduler = UniformSampler(self.num_timesteps)
        elif scheduler_type == 'real-uniform':
            print('Using real uniform sampler')
            self.time_scheduler = RealUniformSampler(self.sigma_max_feat, self.sigma_min_feat)  # disregard sigma_min and sigma_max now
        else:
            raise NotImplementedError(f"unknown schedule sampler: {scheduler_type}")

    def _resume_checkpoint(self):
        pass

    def _load_optimizer_state(self):
        pass

    def _load_ema(self):
        self.ema_params = copy.deepcopy(list(self.backbone.parameters()))

    def _update_ema(self):
        """
            Update target parameters to be closer to those of source parameters using
            an exponential moving average.

            :param target_params: the target parameter sequence. (ema_params)
            :param source_params: the source parameter sequence. (model parameters)
            :param rate: the EMA rate (closer to 1 means slower).
        """
        for targ, src in zip( self.ema_params, self.backbone.parameters()):
            targ = targ.to(src.device)
            # print(targ.device, src.device)
            targ.detach().mul_(self.ema_decay).add_(src, alpha = 1 - self.ema_decay)

    # def load_from_checkpoint(self, checkpoint_path):
    #     checkpoint = torch.load(checkpoint_path)
    #     self.backbone.load_state_dict(checkpoint['state_dict'])

    def preprocess(self, x0, xT, h0, hT, node_mask, Gt_mask=None, num_node=None, batch_info=None, use_mass=False):
        if self.datamodule.startswith('Combined') or self.datamodule == 'QM9Dataset':

            # xT is already included in x0
            # print(x0.size(), node_mask.size(), Gt_mask.size())
            # xT = xT

            if self.datamodule == 'CombinedGraphDataset':
                x0 = center2zero_combined_graph(x0, node_mask, Gt_mask)
                # convert the dense graphs into sparse ones

                # x0_ = x0.view(x0.size(0) * x0.size(1), -1)
                # xT_ = xT.view(xT.size(0) * xT.size(1), -1)
                # node_mask_ = node_mask.view(node_mask.size(0) * node_mask.size(1), -1)

                if self.xT_mode == 'concat_graph':
                    # xT could be pp graphs or noise
                    # xT and x0 are concatenated to form a combined graph

                    # assert self.xT_type == 'pp' or self.xT_type == 'noise'
                    node_mask = node_mask.squeeze(-1)
                    # print(node_mask.size(), x0.size(), x0[0].size())
                    # x0 = x0_[node_mask_].view(x0.size(0), x0.size(1), -1)
                    # xT = xT_[node_mask_].view(xT.size(0), xT.size(1), -1)
                    bs = x0.size(0)
                    x0_, xT_, h0_, hT_ = [], [], [], []
                    sparse_Gt_mask = []
                    batch_all = []
                    for batch_idx in range(bs):
                        x0_.append(x0[batch_idx][node_mask[batch_idx]])
                        xT_.append(xT[batch_idx][node_mask[batch_idx]])
                        h0_.append(h0[batch_idx][node_mask[batch_idx]])
                        hT_.append(hT[batch_idx][node_mask[batch_idx]])
                        N = node_mask[batch_idx].sum().item()    # 2 x number of nodes
                        # print('number of nodes in the current graph', N)
                        Gt_mask_batch = torch.zeros(N, device=self.device)
                        # print(Gt_mask_batch.size(), Gt_mask_batch)
                        Gt_mask_batch[:(N//2)] = 1
                        Gt_mask_batch = Gt_mask_batch.bool()
                        sparse_Gt_mask.append(Gt_mask_batch)
                        batch_all.append(torch.ones(N, device=self.device, dtype=torch.long) * batch_idx)
                    
                    x0 = torch.cat(x0_, dim=0)
                    xT = torch.cat(xT_, dim=0)
                    h0 = torch.cat(h0_, dim=0)
                    hT = torch.cat(hT_, dim=0)
                    Gt_mask = torch.cat(sparse_Gt_mask, dim=0)
                    batch_info = torch.cat(batch_all, dim=0)

                elif self.xT_mode == 'none':
                    # xT is not concatenated to x0, but it is still used for sampling xt
                    # x0 and xT are single graphs
                    assert self.xT_type != 'pp'
                    # node_mask = node_mask.squeeze(-1)
                    Gt_mask = Gt_mask.squeeze(-1)
                    bs = x0.size(0)
                    x0_, xT_, h0_, hT_ = [], [], [], []
                    sparse_Gt_mask = []
                    batch_single = []
                    for batch_idx in range(bs):
                        x0_.append(x0[batch_idx][Gt_mask[batch_idx]])
                        xT_.append(xT[batch_idx][Gt_mask[batch_idx]])
                        h0_.append(h0[batch_idx][Gt_mask[batch_idx]])
                        hT_.append(hT[batch_idx][Gt_mask[batch_idx]])
                        N = Gt_mask[batch_idx].sum().item()    # number of nodes
                        assert node_mask[batch_idx].sum().item() == 2 * N
                        # print('number of nodes in the current graph', N)
                        Gt_mask_batch = torch.zeros(N, device=self.device)
                        # print(Gt_mask_batch.size(), Gt_mask_batch)
                        Gt_mask_batch[:(N)] = 1
                        Gt_mask_batch = Gt_mask_batch.bool()
                        sparse_Gt_mask.append(Gt_mask_batch)
                        batch_single.append(torch.ones(N, device=self.device, dtype=torch.long) * batch_idx)
                    
                    x0 = torch.cat(x0_, dim=0)
                    xT = torch.cat(xT_, dim=0)
                    h0 = torch.cat(h0_, dim=0)
                    hT = torch.cat(hT_, dim=0)
                    Gt_mask = torch.cat(sparse_Gt_mask, dim=0)
                    batch_info = torch.cat(batch_single, dim=0)
                # x0, xT, h0, hT = x0_, xT_, h0_, hT_
                # print(x0.size(), h0.size(), xT.size(), hT.size(), x0.device, Gt_mask.size())
                else:
                    raise NotImplementedError(f"Unknown xT_mode: {self.xT_mode}")
            elif self.datamodule == 'CombinedSparseGraphDataset' or self.datamodule == 'QM9Dataset':
                # data is already in sparse format
                
                # x0 = x0[0]
                Gt_mask = Gt_mask[0]
                node_mask = node_mask[0]
                x0 = center2zero_sparse_graph(x0, Gt_mask.squeeze(-1), batch_info)
                xT = center2zero_sparse_graph(xT, Gt_mask.squeeze(-1), batch_info)

                if self.xT_type == 'noise':
                    # xT = xT[0]  # (1, N, 3) => (N, 3)
                    # hT = hT[0]  # (1, N, h) => (N, h)
                    bs = torch.max(batch_info).item() + 1

                    x0_, xT_, h0_, hT_ = [], [], [], []
                    for i in range(bs):
                        batch_idx = (batch_info == i)
                        x0_batch = x0[batch_idx]
                        # xT_batch = xT[batch_idx]
                        h0_batch = h0[batch_idx]
                        # hT_batch = hT[batch_idx]
                        Gt_mask_batch = Gt_mask[batch_idx].squeeze(-1)
                        # print(x0_batch.size(), h0_batch.size(), Gt_mask_batch.size())

                        # directly change x0 and h0, and rebuild xT and hT
                        # x0_Gt = x0_batch[Gt_mask_batch]
                        # x0_noise = torch.randn_like(x0_Gt, device=x0.device)
                        x0_noise = sample_zero_center_gaussian(x0_batch[Gt_mask_batch].size(), device=x0.device)
                        # x0_noise = xT_batch[~Gt_mask_batch]
                        # x0_batch_ = torch.cat([x0_Gt, x0_noise], dim=0)
                        xT_batch_ = torch.cat([x0_noise, x0_noise], dim=0)
                        # x0_.append(x0_batch_)
                        xT_.append(xT_batch_)
                        x0[batch_idx][~Gt_mask_batch] = x0_noise

                        # h0_Gt = h0_batch[Gt_mask_batch]
                        h0_noise = torch.randn_like(h0_batch[Gt_mask_batch], device=h0.device)
                        # h0_noise = hT_batch[~Gt_mask_batch]
                        # h0_batch_ = torch.cat([h0_Gt, h0_noise], dim=0)
                        hT_batch_ = torch.cat([h0_noise, h0_noise], dim=0)
                        # h0_.append(h0_batch_)
                        hT_.append(hT_batch_)
                        h0[batch_idx][~Gt_mask_batch] = h0_noise

                    xT = torch.cat(xT_, dim=0)
                    hT = torch.cat(hT_, dim=0)

                # print(x0.size(), xT.size(), h0.size(), hT.size(), Gt_mask.size(), node_mask.size())
                # print(batch_info)
                # pass
            else:
                raise ValueError(f"Unknown datamodule: {self.datamodule}")
        else:
            # move the center to zero
            x0 = center2zero_with_mask(x0, node_mask)
            if self.xT_type == 'noise':
                xT = center2zero_with_mask(xT, node_mask, check_mask=False)
            else:
                xT = center2zero_with_mask(xT, node_mask)
        return x0, xT, h0, hT, Gt_mask, batch_info

    def get_input(self, batch):
        '''
            h: atom features (x in data)
            x: atom positions (pos in data)
            node_mask: mask out sparse nodes
        '''

        if self.datamodule == 'CombinedSparseGraphDataset' or self.datamodule == 'QM9Dataset':
            # information below is not useful for sparse graphs
            num_nodes = batch.x.size(0)
            # print(num_nodes)
            node_mask = torch.ones([1, num_nodes], device=batch.x.device, dtype=torch.bool)
            # edge_mask = torch.ones([1, num_nodes * num_nodes], device=batch.x.device, dtype=torch.bool)
            edge_mask = None
            CoM = None
        else:
            node_mask = batch.node_mask
            edge_mask = batch.edge_mask
            CoM = batch.CoM
            num_nodes = batch.num_nodes
        batch_info = batch.batch

        node_mask = node_mask.unsqueeze(-1)
        h0 = batch.x
        x0 = batch.pos

        if self.xT_type == 'noise':
            # print('Using noise as xT')
            # h0, x0 = batch.x, batch.pos
            # print(node_mask.size())
            hT = torch.randn_like(h0, device=h0.device)
            hT = F.one_hot(hT.argmax(dim=-1), num_classes=h0.size(-1)).float().to(h0.device)
            # print(hT.size())
            hT = hT * node_mask
            # print(hT.size())
            xT = torch.randn_like(x0, device=x0.device)
            # print(xT.size())
            xT = xT * node_mask
            # print(xT.size())
        # elif self.xT_type == 'none':
        #     hT = None
        #     xT = None 
        else:
            # h0, hT = batch.x, batch.target_x
            # x0, xT = batch.pos, batch.target_pos
            hT = batch.target_x
            xT = batch.target_pos
            
       
        # print(node_mask.size())
        if self.datamodule.startswith('Combined') or self.datamodule == 'QM9Dataset':
            # GT is already included in Gt, so xT and hT don't matter
            Gt_mask = batch.Gt_mask.view(node_mask.size(0), node_mask.size(1), -1)
            # print(Gt_mask.size(), node_mask.size())
            x0, xT, h0, hT, Gt_mask, batch_info = self.preprocess(x0, xT, h0, hT, node_mask, Gt_mask=Gt_mask, num_node=num_nodes, batch_info=batch_info)
        else:
            x0, xT, h0, hT, _, _ = self.preprocess(x0, xT, h0, hT, node_mask)
        return h0, hT, x0, xT, node_mask, edge_mask, Gt_mask, CoM, num_nodes, batch_info

    def forward(self, batch):
        t, weights = self.time_scheduler.sample(self.batch_size, self.device)    # t size: (batch_size) also = number of graphs
        # t, weights = self.time_scheduler.sample_zero2one(self.batch_size, self.device)    # disregard sigma_min and sigma_max now
        # print('t size:', t.size())
        # print('sampled t', t)
        losses = self.compute_losses(self.backbone, batch, t)
        return losses
    
    def get_snr(self, sigmas):
        if self.bridge_type.startswith('vp'):
            return vp_logsnr(sigmas, self.beta_d, self.beta_min).exp()
        else:
            return sigmas**-2

    def get_sigmas(self, sigmas):
        return sigmas
    
    def get_weightings(self, sigma, sigma_max, sigma_data, sigma_data_end, cov_xy):
        snrs = self.get_snr(sigma)
        
        if self.weight_schedule == "snr":
            weightings = snrs
        elif self.weight_schedule == "snr+1":
            weightings = snrs + 1
        elif self.weight_schedule == "karras":
            weightings = snrs + 1.0 / sigma_data**2
        elif self.weight_schedule.startswith("bridge_karras"):
            if self.bridge_type == 've':
                A = sigma**4 / sigma_max**4 * sigma_data_end**2 + (1 - sigma**2 / sigma_max**2)**2 * sigma_data**2 + 2*sigma**2 / sigma_max**2 * (1 - sigma**2 / sigma_max**2) * cov_xy + self.c**2 * sigma**2 * (1 - sigma**2 / sigma_max**2)
                weightings = A / ((sigma/sigma_max)**4 * (sigma_data_end**2 * sigma_data**2 - cov_xy**2) + sigma_data**2 * self.c**2 * sigma**2 * (1 - sigma**2/sigma_max**2) )
            
            elif self.bridge_type == 'vp':
                
                logsnr_t = vp_logsnr(sigma, self.beta_d, self.beta_min)
                logsnr_T = vp_logsnr(1, self.beta_d, self.beta_min)
                logs_t = vp_logs(sigma, self.beta_d, self.beta_min)
                logs_T = vp_logs(1, self.beta_d, self.beta_min)

                a_t = (logsnr_T - logsnr_t +logs_t -logs_T).exp()
                b_t = -torch.expm1(logsnr_T - logsnr_t) * logs_t.exp()
                c_t = -torch.expm1(logsnr_T - logsnr_t) * (2*logs_t - logsnr_t).exp()
                # print('a_t, b_t, c_t', a_t, b_t, c_t)

                A = a_t**2 * sigma_data_end**2 + b_t**2 * sigma_data**2 + 2*a_t * b_t * cov_xy + self.c**2 * c_t
                weightings = A / (a_t**2 * (sigma_data_end**2 * sigma_data**2 - cov_xy**2) + sigma_data**2 * self.c**2 * c_t )
                # print('loss weightings', weightings)
                
            elif self.bridge_type == 'vp_simple' or  self.bridge_type == 've_simple':

                weightings = torch.ones_like(snrs)
        elif self.weight_schedule == "truncated-snr":
            weightings = torch.clamp(snrs, min=1.0)
        elif self.weight_schedule == "uniform":
            weightings = torch.ones_like(snrs)
        else:
            raise NotImplementedError()

        return weightings
    
    def get_bridge_scalings(self, sigma, sigma_max, sigma_data, sigma_data_end, cov_xy):
        '''
            Compute the scalings for the bridge at a given timestep.
            The scalings are the same for feature matrix x and position matrix pos.
        '''
        if self.bridge_type == 've':
            A = sigma**4 / sigma_max**4 * sigma_data_end**2 + (1 - sigma**2 / sigma_max**2)**2 * sigma_data**2 + 2*sigma**2 / sigma_max**2 * (1 - sigma**2 / sigma_max**2) * cov_xy + self.c **2 * sigma**2 * (1 - sigma**2 / sigma_max**2)
            c_in = 1 / (A) ** 0.5
            c_skip = ((1 - sigma**2 / sigma_max**2) * sigma_data**2 + sigma**2 / sigma_max**2 * cov_xy)/ A
            c_out =((sigma/sigma_max)**4 * (sigma_data_end**2 * sigma_data**2 - cov_xy**2) + sigma_data**2 *  self.c **2 * sigma**2 * (1 - sigma**2/sigma_max**2) )**0.5 * c_in
            return c_skip, c_out, c_in
    
        elif self.bridge_type == 'vp':

            logsnr_t = vp_logsnr(sigma, self.beta_d, self.beta_min)
            logsnr_T = vp_logsnr(1, self.beta_d, self.beta_min)
            logs_t = vp_logs(sigma, self.beta_d, self.beta_min)
            logs_T = vp_logs(1, self.beta_d, self.beta_min)

            a_t = (logsnr_T - logsnr_t +logs_t -logs_T).exp()
            b_t = -torch.expm1(logsnr_T - logsnr_t) * logs_t.exp()
            c_t = -torch.expm1(logsnr_T - logsnr_t) * (2*logs_t - logsnr_t).exp()

            A = a_t**2 * sigma_data_end**2 + b_t**2 * sigma_data**2 + 2*a_t * b_t * cov_xy + self.c**2 * c_t
            
            
            c_in = 1 / (A) ** 0.5
            c_skip = (b_t * sigma_data**2 + a_t * cov_xy)/ A
            c_out =(a_t**2 * (sigma_data_end**2 * sigma_data**2 - cov_xy**2) + sigma_data**2 *  self.c **2 * c_t )**0.5 * c_in
            return c_skip, c_out, c_in
    
        elif self.bridge_type == 've_simple' or self.bridge_type == 'vp_simple':
            
            c_in = torch.ones_like(sigma)
            c_out = torch.ones_like(sigma) 
            c_skip = torch.zeros_like(sigma)
            return c_skip, c_out, c_in
        
        else:
            raise NotImplementedError(f"unknown bridge type: {self.bridge_type}")
        
    def denoise(self, backbone, sigmas, h_t, x_t, h_T, x_T, node_mask=None, edge_mask=None, Gt_mask=None, batch_info=None):
        '''
            node_mask and edge_mask should appear in the model_kwargs
        '''

        c_skip_h, c_out_h, c_in_h = [
            append_dims(x, h_t.ndim) for x in self.get_bridge_scalings(sigmas, self.sigma_max_feat, self.sigma_data_feat, self.sigma_data_end_feat, self.cov_xy_feat)
        ]
        c_skip_x, c_out_x, c_in_x = [
            append_dims(x, x_t.ndim) for x in self.get_bridge_scalings(sigmas, self.sigma_max_pos, self.sigma_data_pos, self.sigma_data_end_pos, self.cov_xy_pos)
        ]
        # print('c_skip, c_out, c_in', c_skip, c_out, c_in)
               
        rescaled_t = 1000 * 0.25 * torch.log(sigmas + 1e-44)
        if self.datamodule.startswith('Combined') or self.datamodule == 'QM9Dataset':
            # print(h_t.size(), rescaled_t.size(), c_in.size())
            output_h, output_x = backbone(c_in_h * h_t, c_in_x * x_t, rescaled_t, Gt_mask, batch_info)
        else:
            output_h, output_x = backbone(c_in_h * h_t, c_in_x * x_t, h_T, x_T, rescaled_t, batch_info, node_mask, edge_mask, Gt_mask, self.xT_mode)    # model_output is the output of F_theta (something between x0^hat and the noise at t)
        denoised_h = c_out_h * output_h + c_skip_h * h_t
        denoised_x = c_out_x * output_x + c_skip_x * x_t
        return output_h, output_x, denoised_h, denoised_x   # denoised is the output of the D_theta (the predicted x0^hat)

    def sample_noise(self, h_start, x_start, mask=None):
        noise_h = torch.randn_like(h_start, device=self.device)
        if self.datamodule.startswith('Combined') or self.datamodule == 'QM9Dataset':  # features are in sparse representations (dim is 2)
            # noise_h = sample_zero_center_gaussian(h_start.size(), self.device)
            noise_x = sample_zero_center_gaussian(x_start.size(), self.device)
        else:
            # noise_h = sample_zero_center_gaussian_with_mask(h_start.size(), self.device, mask)
            noise_x = sample_zero_center_gaussian_with_mask(x_start.size(), self.device, mask)

        return noise_h, noise_x
    
    def compute_losses(self, backbone, batch, sigmas, noise_h=None, noise_x=None):
        h_start, h_T, x_start, x_T, node_mask, edge_mask, Gt_mask, _, num_nodes, batch_info = self.get_input(batch)
        # print(h_start.size(), h_T.size(), x_start.size(), x_T.size(), Gt_mask.size())

        # sample CoM-free noise 
        if noise_h is None and noise_x is None:
            # noise = torch.randn_like(h_start)
            noise_h, noise_x = self.sample_noise(h_start, x_start, node_mask)

        if self.datamodule.startswith('Combined') or self.datamodule == 'QM9Dataset':
            # print(batch_info)
            sigmas = sigmas[batch_info]
            # print('extending selected t to', sigmas.size())
            # print(sigmas)
        sigmas = torch.minimum(sigmas, torch.ones_like(sigmas)* self.sigma_max_pos)    # use sigma_max_feat, sigma_max for both feat and pos should be the same
        # print('sigmas', sigmas)

        losses = {}
        feat_dims = h_start.dim()
        pos_dims = x_start.dim()

        def bridge_sample(x0, xT, t, dims, noise, sigma_max):
            t = append_dims(t, dims)    # [1, 1, 2, 2, 2, ...] => [[1], [1], [2], [2], [2], ...]
            # std_t = th.sqrt(t)* th.sqrt(1 - t / sigma_max)
            if self.bridge_type.startswith('ve'):
                std_t = t* torch.sqrt(1 - t**2 / sigma_max**2)
                mu_t= t**2 / sigma_max**2 * xT + (1 - t**2 / sigma_max**2) * x0
                samples = (mu_t +  std_t * noise )
            elif self.bridge_type.startswith('vp'):
                logsnr_t = vp_logsnr(t, self.beta_d, self.beta_min)
                logsnr_T = vp_logsnr(sigma_max, self.beta_d, self.beta_min)
                logs_t = vp_logs(t, self.beta_d, self.beta_min)
                logs_T = vp_logs(sigma_max, self.beta_d, self.beta_min)

                a_t = (logsnr_T - logsnr_t +logs_t -logs_T).exp()
                b_t = -torch.expm1(logsnr_T - logsnr_t) * logs_t.exp()
                std_t = (-torch.expm1(logsnr_T - logsnr_t)).sqrt() * (logs_t - logsnr_t/2).exp()
                
                a = a_t * xT
                b = b_t * x0
                c = std_t * noise
                samples = a + b + c
                
            return samples

        # h_t = bridge_sample(h_start, h_T, sigmas, feat_dims, noise_h)
        # x_t = bridge_sample(x_start, x_T, sigmas, pos_dims, noise_x)
        if self.datamodule.startswith('Combined') or self.datamodule == 'QM9Dataset':
            # only sample the Gt part but keep the GT part the same
            # Gt_mask = batch.Gt_mask
            h_t_sampled = bridge_sample(h_start, h_T, sigmas, feat_dims, noise_h, self.sigma_max_feat)
            x_t_sampled = bridge_sample(x_start, x_T, sigmas, pos_dims, noise_x, self.sigma_max_pos)
            # print(h_start.size(), h_t_sampled.size(), Gt_mask_.size())

            if self.datamodule == 'CombinedGraphDataset':
                # for CombinedSparseDataset, do nothing
                Gt_mask = Gt_mask.unsqueeze(-1)
            # h_start_ = h_start.view(h_start.size(0) * h_start.size(1), -1)
            # h_t_sampled_ = h_t_sampled.view(h_t_sampled.size(0) * h_t_sampled.size(1), -1)
            # print(Gt_mask.size())
            h_t = h_start * (~Gt_mask) + h_t_sampled * Gt_mask
            # print(h_t.size())
            # h_t = h_t.view(h_start.size(0), h_start.size(1), -1)

            # x_start_ = x_start.view(x_start.size(0) * x_start.size(1), -1)
            # x_t_sampled_ = x_t_sampled.view(x_t_sampled.size(0) * x_t_sampled.size(1), -1)
            x_t = x_start * (~Gt_mask) + x_t_sampled * Gt_mask
            # x_t = x_t.view(x_start.size(0), x_start.size(1), -1)
        else:
            Gt_mask = None
            h_t = bridge_sample(h_start, h_T, sigmas, feat_dims, noise_h, self.sigma_max_feat)
            x_t = bridge_sample(x_start, x_T, sigmas, pos_dims, noise_x, self.sigma_max_pos) 
        _, _, denoised_x, denoised_pos = self.denoise(backbone, sigmas, h_t, x_t, h_T, x_T, node_mask=node_mask, edge_mask=edge_mask, Gt_mask=Gt_mask.squeeze(-1), batch_info=batch_info)

        x_weights = self.get_weightings(sigmas, self.sigma_max_feat, self.sigma_data_feat, self.sigma_data_end_feat, self.cov_xy_feat)
        pos_weights = self.get_weightings(sigmas, self.sigma_max_pos, self.sigma_data_pos, self.sigma_data_end_pos, self.cov_xy_pos)
        x_weights =  append_dims((x_weights), feat_dims)
        pos_weights = append_dims((pos_weights), pos_dims)  
        # print(x_weights==pos_weights)
        if self.datamodule.startswith('Combined') or self.datamodule == 'QM9Dataset':
            Gt_mask_ = Gt_mask.squeeze(-1)

            if self.datamodule == 'CombinedSparseGraphDataset' or self.datamodule == 'QM9Dataset':
                # center the original x to zero
                # original_x, original_h = batch.pos[Gt_mask_], batch.x[Gt_mask_]
                # original_x = center2zero(original_x, mean_dim=0)

                # center to zero by the pp graph center
                original_x, original_h = x_start[Gt_mask_], h_start[Gt_mask_]
            else:
                # original_x, original_h = batch.original_pos, batch.original_x
                # original_x = center2zero(original_x, mean_dim=1)

                original_x, original_h = x_start[Gt_mask_], h_start[Gt_mask_]
            # Gt_mask = batch.Gt_mask
            # Gt_mask_ = Gt_mask.view(h_start.size(0), h_start.size(1))
            
            # print(Gt_mask_.sum())
            # print(denoised_x[Gt_mask_].size(), original_h.size())

            loss_x_mse = (denoised_x[Gt_mask_] - original_h) ** 2
            # loss_x_ce = F.cross_entropy(denoised_x[Gt_mask_], original_h.argmax(dim=-1), reduction='none').unsqueeze(-1)
            loss_pos_mse = (denoised_pos[Gt_mask_] - original_x) ** 2
            # print(denoised_x[Gt_mask_].size(), original_h.argmax(dim=-1).size(), loss_x_ce.size())

            # losses["x_mse"] = mean_flat(loss_x_mse)   
            losses['x_mse'] = scatter_mean_flat(loss_x_mse, batch_info[Gt_mask_])   
            # losses['x_ce'] = mean_flat(loss_x_ce)
            # losses['x_ce'] = scatter_mean_flat(loss_x_ce, batch_info[Gt_mask_])

            # losses['pos_mse'] = mean_flat((denoised_pos[Gt_mask_] - original_x) ** 2)
            losses["pos_mse"] = scatter_mean_flat(loss_pos_mse, batch_info[Gt_mask_])

            # losses["weighted_x_mse"] = mean_flat(x_weights[Gt_mask_] * loss_x_mse)
            losses["weighted_x_mse"] = scatter_mean_flat(x_weights[Gt_mask_] * loss_x_mse, batch_info[Gt_mask_])
            # losses['weighted_x_ce'] = mean_flat(x_weights[Gt_mask_] * loss_x_ce)
            # losses['weighted_x_ce'] = scatter_mean_flat(x_weights[Gt_mask_] * loss_x_ce, batch_info[Gt_mask_])

            # losses["weighted_pos_mse"] = mean_flat(pos_weights[Gt_mask_] * (denoised_pos[Gt_mask_] - original_x) ** 2)
            losses["weighted_pos_mse"] = scatter_mean_flat(pos_weights[Gt_mask_] * loss_pos_mse, batch_info[Gt_mask_])

            losses['loss'] = scatter_flat(self.loss_x_weight * torch.sum(x_weights[Gt_mask_] * loss_x_mse, dim=-1) + 
                                               torch.sum(pos_weights[Gt_mask_] * loss_pos_mse, dim=-1),
                                               batch_info[Gt_mask_])
            
            # losses['loss'] = scatter_flat(self.loss_x_weight * torch.sum(x_weights[Gt_mask_] * loss_x_ce, dim=-1) + 
            #                                    torch.sum(pos_weights[Gt_mask_] * loss_pos_mse, dim=-1),
            #                                    batch_info[Gt_mask_])

        else:
            losses["x_mse"] = mean_flat((denoised_x - h_start) ** 2)      # x should be the atom type one hot encoding, think twice of mse loss
            losses['pos_mse'] = mean_flat((denoised_pos - x_start) ** 2)
            losses["weighted_x_mse"] = mean_flat(x_weights * (denoised_x - h_start) ** 2)
            losses["weighted_pos_mse"] = mean_flat(pos_weights * (denoised_pos - x_start) ** 2)
            losses['loss_x'] = losses["weighted_x_mse"]
            losses['loss_pos'] = losses["weighted_pos_mse"]

            losses["loss"] = self.loss_x_weight * losses["loss_x"] + losses["loss_pos"]  # should include additoinal weights?
        

        return losses

    def run_step(self, batch, batch_idx):
        # print(batch.x.size(0))
        # print(self.batch_size)
        if self.batch_size != batch.x.size(0):
            # in the last batch
            self.batch_size = batch.x.size(0)
            
        # assert self.batch_size == batch.x.size(0)

        losses = self(batch)
        return losses

    def training_step(self, batch, batch_idx):
        losses = self.run_step(batch, batch_idx)
        final_loss = losses['loss'].mean()

        to_log = {'train_'+k: v.mean() for k, v in losses.items()}
        self.log_dict(to_log, logger=True, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log('global_step', self.global_step, on_step=True, on_epoch=False, prog_bar=True, logger=True, batch_size=self.batch_size)

        if self.use_lr_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False, batch_size=self.batch_size)
        return final_loss
    
    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self._update_ema()

    def validation_step(self, batch, batch_idx):
        losses = self.run_step(batch, batch_idx)
        to_log = {'val_'+k: v.mean() for k, v in losses.items()}
        self.log_dict(to_log, logger=True, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        return losses
    
    def on_validation_epoch_end(self):
        avg_loss = self.trainer.callback_metrics['val_loss']
        # self.logger.experiment.add_scalar('Validation Loss', avg_loss, self.current_epoch)
        self.log('val_loss_epoch', avg_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # TODO: log some reconstructed samples 

    def configure_optimizers(self):
        '''
            Customize the optimization process to update params of both the backbone and the diffusion bridge.
        '''
        # optimizer = torch.optim.RAdam(self.backbone.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        optimizer = torch.optim.AdamW(self.backbone.parameters(), lr=self.lr, amsgrad=True, weight_decay=self.weight_decay)
        if self.use_lr_scheduler:
            lr_scheduler = instantiate_from_config(self.lr_scheduler_config)

            # print("Setting up LambdaLR scheduler...")
            print("Setting up ExponentialLR scheduler...")
            lr_scheduler = [
                {
                    'scheduler': ExponentialLR(optimizer, gamma=0.95), # LambdaLR(optimizer, lr_lambda=lr_scheduler.schedule),
                    'interval': 'epoch',
                    'frequency': 5,
                    'last_epoch': 450,
                    # 'gradient_clip_val': 0.5, 
                    # 'gradient_clip_algorithm': 'norm'  # Choose 'norm' or 'value'
                }]
            print("Done setting up scheduler.")
            return [optimizer], lr_scheduler
        return optimizer