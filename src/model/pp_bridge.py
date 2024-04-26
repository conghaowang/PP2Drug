import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
import copy

from EGNN_backbone import EGNN
from utils.time_scheduler import UniformSampler, RealUniformSampler
from utils.utils_diffusion import append_dims, vp_logs, vp_logsnr, mean_flat

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
            pred_mode='both',
            resume_step=0,
            device='cuda',
        ):
        super().__init__()
        model_config = config['model']['denoiser']
        backbone_config = config['model']['backbone']
        training_config = config['training']

        self.sigma_data = model_config['sigma_data']
        self.sigma_max = model_config['sigma_max'] 
        self.sigma_min = model_config['sigma_min']
        self.beta_d = model_config['beta_d']
        self.beta_min = model_config['beta_min']
        self.sigma_data_end = self.sigma_data
        self.cov_xy = cov_xy
        self.c = 1

        self.weight_schedule = model_config['weight_schedule']
        self.bridge_type = model_config['bridge_type']    # previous pred_mode

        self.rho = rho
        self.num_timesteps = 40
        self.feature_size = model_config['feature_size']
        self.lr = training_config['learning_rate']
        self.lr_anneal_steps = training_config['lr_anneal_steps']

        if backbone_config['type'] == 'EGNN':
            self.backbone = EGNN(backbone_config)
        else:
            raise NotImplementedError(f"Backbone type {backbone_config['type']} not implemented")

        self.use_ema = model_config['use_ema']
        self.ema_decay = model_config['ema_decay']
        self.ema_params = self.backbone.parameters()
        self.weight_decay = model_config['weight_decay']
        self.loss_x_weight = model_config['loss_x_weight']

        self.batch_size = training_config['batch_size']
        self.log_interval = training_config['log_interval']
        self.save_interval = training_config['save_interval']
        self.test_interval = training_config['test_interval']
        self.total_training_steps = training_config['total_training_steps']

        # resuming from checkpoint or any step has not been implemented yet
        self.resume_checkpoint = training_config.get('resume_checkpoint', None)
        self._resume_checkpoint()
        self.resume_step = resume_step

        self.optimizer = torch.optim.RAdam(self.backbone.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.resume_step:
            self._load_optimizer_state()
            raise NotImplementedError("Optimizer state loading not implemented")
        else:
            self.ema_params = copy.deepcopy(self.backbone.parameters())

        # self.generator = None    # generator is not implemented here, this model is only for training
        scheduler_type = model_config['schedule_sampler']
        if scheduler_type == 'uniform':
            self.time_scheduler = UniformSampler(self.num_timesteps)
        elif scheduler_type == 'real-uniform':
            self.time_scheduler = RealUniformSampler(self.sigma_max, self.sigma_min)
        else:
            raise NotImplementedError(f"unknown schedule sampler: {scheduler_type}")

    def _resume_checkpoint(self):
        pass

    def _load_optimizer_state(self):
        pass

    def _update_ema(self):
        """
            Update target parameters to be closer to those of source parameters using
            an exponential moving average.

            :param target_params: the target parameter sequence. (ema_params)
            :param source_params: the source parameter sequence. (model parameters)
            :param rate: the EMA rate (closer to 1 means slower).
        """
        for targ, src in zip( self.ema_params, self.backbone.parameters()):
            targ.detach().mul_(self.ema_decay).add_(src, alpha = 1 - self.ema_decay)


    def preprocess(self, batch):
        # shall we move the CoM to zero here?
        return batch

    def get_input(self, batch):
        x, x_T = batch.x, batch.target_x
        pos, pos_T = batch.pos, batch.target_pos
        return x, x_T, pos, pos_T

    def forward(self, batch):
        t, weights = self.time_scheduler.sample(self.batch_size, self.device)
        losses = self.compute_losses(self.backbone, batch, t)
        return losses
    
    def get_weightings(self, sigma):
        snrs = self.get_snr(sigma)
        
        if self.weight_schedule == "snr":
            weightings = snrs
        elif self.weight_schedule == "snr+1":
            weightings = snrs + 1
        elif self.weight_schedule == "karras":
            weightings = snrs + 1.0 / self.sigma_data**2
        elif self.weight_schedule.startswith("bridge_karras"):
            if self.pred_mode == 've':
                A = sigma**4 / self.sigma_max**4 * self.sigma_data_end**2 + (1 - sigma**2 / self.sigma_max**2)**2 * self.sigma_data**2 + 2*sigma**2 / self.sigma_max**2 * (1 - sigma**2 / self.sigma_max**2) * self.cov_xy + self.c**2 * sigma**2 * (1 - sigma**2 / self.sigma_max**2)
                weightings = A / ((sigma/self.sigma_max)**4 * (self.sigma_data_end**2 * self.sigma_data**2 - self.cov_xy**2) + self.sigma_data**2 * self.c**2 * sigma**2 * (1 - sigma**2/self.sigma_max**2) )
            
            elif self.pred_mode == 'vp':
                
                logsnr_t = vp_logsnr(sigma, self.beta_d, self.beta_min)
                logsnr_T = vp_logsnr(1, self.beta_d, self.beta_min)
                logs_t = vp_logs(sigma, self.beta_d, self.beta_min)
                logs_T = vp_logs(1, self.beta_d, self.beta_min)

                a_t = (logsnr_T - logsnr_t +logs_t -logs_T).exp()
                b_t = -torch.expm1(logsnr_T - logsnr_t) * logs_t.exp()
                c_t = -torch.expm1(logsnr_T - logsnr_t) * (2*logs_t - logsnr_t).exp()

                A = a_t**2 * self.sigma_data_end**2 + b_t**2 * self.sigma_data**2 + 2*a_t * b_t * self.cov_xy + self.c**2 * c_t
                weightings = A / (a_t**2 * (self.sigma_data_end**2 * self.sigma_data**2 - self.cov_xy**2) + self.sigma_data**2 * self.c**2 * c_t )
                
            elif self.pred_mode == 'vp_simple' or  self.pred_mode == 've_simple':

                weightings = torch.ones_like(snrs)
        elif self.weight_schedule == "truncated-snr":
            weightings = torch.clamp(snrs, min=1.0)
        elif self.weight_schedule == "uniform":
            weightings = torch.ones_like(snrs)
        else:
            raise NotImplementedError()

        return weightings
    
    def get_bridge_scalings(self, sigma):
        '''
            Compute the scalings for the bridge at a given timestep.
            The scalings are the same for feature matrix x and position matrix pos.
        '''
        if self.bridge_type == 've':
            A = sigma**4 / self.sigma_max**4 * self.sigma_data_end**2 + (1 - sigma**2 / self.sigma_max**2)**2 * self.sigma_data**2 + 2*sigma**2 / self.sigma_max**2 * (1 - sigma**2 / self.sigma_max**2) * self.cov_xy + self.c **2 * sigma**2 * (1 - sigma**2 / self.sigma_max**2)
            c_in = 1 / (A) ** 0.5
            c_skip = ((1 - sigma**2 / self.sigma_max**2) * self.sigma_data**2 + sigma**2 / self.sigma_max**2 * self.cov_xy)/ A
            c_out =((sigma/self.sigma_max)**4 * (self.sigma_data_end**2 * self.sigma_data**2 - self.cov_xy**2) + self.sigma_data**2 *  self.c **2 * sigma**2 * (1 - sigma**2/self.sigma_max**2) )**0.5 * c_in
            return c_skip, c_out, c_in
    
        elif self.bridge_type == 'vp':

            logsnr_t = vp_logsnr(sigma, self.beta_d, self.beta_min)
            logsnr_T = vp_logsnr(1, self.beta_d, self.beta_min)
            logs_t = vp_logs(sigma, self.beta_d, self.beta_min)
            logs_T = vp_logs(1, self.beta_d, self.beta_min)

            a_t = (logsnr_T - logsnr_t +logs_t -logs_T).exp()
            b_t = -torch.expm1(logsnr_T - logsnr_t) * logs_t.exp()
            c_t = -torch.expm1(logsnr_T - logsnr_t) * (2*logs_t - logsnr_t).exp()

            A = a_t**2 * self.sigma_data_end**2 + b_t**2 * self.sigma_data**2 + 2*a_t * b_t * self.cov_xy + self.c**2 * c_t
            
            
            c_in = 1 / (A) ** 0.5
            c_skip = (b_t * self.sigma_data**2 + a_t * self.cov_xy)/ A
            c_out =(a_t**2 * (self.sigma_data_end**2 * self.sigma_data**2 - self.cov_xy**2) + self.sigma_data**2 *  self.c **2 * c_t )**0.5 * c_in
            return c_skip, c_out, c_in
    
        elif self.bridge_type == 've_simple' or self.bridge_type == 'vp_simple':
            
            c_in = torch.ones_like(sigma)
            c_out = torch.ones_like(sigma) 
            c_skip = torch.zeros_like(sigma)
            return c_skip, c_out, c_in
        
        else:
            raise NotImplementedError(f"unknown bridge type: {self.bridge_type}")
        
    def denoise(self, backbone, sigmas, x_t, pos_t, x_T, pos_T, model_kwargs=None):

        c_skip, c_out, c_in = [
            append_dims(x, x_t.ndim) for x in self.get_bridge_scalings(sigmas)
        ]
               
        rescaled_t = 1000 * 0.25 * torch.log(sigmas + 1e-44)
        output_x, output_pos = backbone(c_in * x_t, c_in * pos_t, rescaled_t, x_T, pos_T, **model_kwargs)    # model_output is the output of F_theta (something between x0^hat and the noise at t)
        denoised_x = c_out * output_x + c_skip * x_t
        denoised_pos = c_out * output_pos + c_skip * pos_t
        return output_x, output_pos, denoised_x, denoised_pos   # denoised is the output of the D_theta (the predicted x0^hat)
    
    def compute_losses(self, backbone, batch, sigmas, noise=None):
        x, x_T, pos, pos_T = self.get_input(batch)
        # sample CoM-free noise 
        if noise is None:
            noise = torch.randn_like(x)
        sigmas = torch.minimum(sigmas, torch.ones_like(sigmas)* self.sigma_max)

        losses = {}
        x_dims = x.dim()
        pos_dims = pos.dim()

        def bridge_sample(x0, xT, t, dims):
            t = append_dims(t, dims)
            # std_t = th.sqrt(t)* th.sqrt(1 - t / self.sigma_max)
            if self.bridge_type.startswith('ve'):
                std_t = t* torch.sqrt(1 - t**2 / self.sigma_max**2)
                mu_t= t**2 / self.sigma_max**2 * xT + (1 - t**2 / self.sigma_max**2) * x0
                samples = (mu_t +  std_t * noise )
            elif self.bridge_type.startswith('vp'):
                logsnr_t = vp_logsnr(t, self.beta_d, self.beta_min)
                logsnr_T = vp_logsnr(self.sigma_max, self.beta_d, self.beta_min)
                logs_t = vp_logs(t, self.beta_d, self.beta_min)
                logs_T = vp_logs(self.sigma_max, self.beta_d, self.beta_min)

                a_t = (logsnr_T - logsnr_t +logs_t -logs_T).exp()
                b_t = -torch.expm1(logsnr_T - logsnr_t) * logs_t.exp()
                std_t = (-torch.expm1(logsnr_T - logsnr_t)).sqrt() * (logs_t - logsnr_t/2).exp()
                
                samples = a_t * xT + b_t * x0 + std_t * noise
                
            return samples

        x_t = bridge_sample(x, x_T, sigmas, x_dims)
        pos_t = bridge_sample(pos, pos_T, sigmas, pos_dims)
        _, _, denoised_x, denoised_pos = self.denoise(backbone, sigmas, x_t, pos_t, x_T, pos_T)

        weights = self.get_weightings(sigmas)
        x_weights =  append_dims((weights), x_dims)
        pos_weights = append_dims((weights), pos_dims)
        losses["x_mse"] = mean_flat((denoised_x - x) ** 2)      # x should be the atom type one hot encoding, so don't use mse loss!!!
        losses['pos_mse'] = mean_flat((denoised_pos - pos) ** 2)
        losses["weighted_x_mse"] = mean_flat(x_weights * (denoised_x - x) ** 2)
        losses["weighted_pos_mse"] = mean_flat(pos_weights * (denoised_pos - pos) ** 2)
        losses['loss_x'] = losses["weighted_x_mse"]
        losses['loss_pos'] = losses["weighted_pos_mse"]

        losses["loss"] = self.loss_x_weight * losses["loss_x"] + losses["loss_pos"]  # should include additoinal weights? targetdiff set the weight of loss_x as 100

        return losses

    def run_step(self, batch, batch_idx):
        # preprocess?
        batch = self.preprocess(batch)
        assert self.batch_size == batch.shape[0]

        losses = self(batch)
        return losses

    def training_step(self, batch, batch_idx):
        losses = self.run_step(batch, batch_idx)
        final_loss = losses['loss']
        return final_loss
    
    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self._update_ema()

    def configure_optimizers(self):
        '''
            Customize the optimization process to update params of both the backbone and the diffusion bridge.
        '''
        return torch.optim.RAdam(self.backbone.parameters(), lr=self.lr, weight_decay=self.weight_decay)