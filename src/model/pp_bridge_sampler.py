import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import sys
sys.path.append('../model')
from pp_bridge import PPBridge
from model.utils.utils_diffusion import append_dims, append_zero, center2zero_combined_graph, center2zero_sparse_graph, center2zero_with_mask, sample_zero_center_gaussian


class PPBridgeSampler(PPBridge):
    def __init__(self, config, ckpt_path, device='cuda'):
        super(PPBridgeSampler, self).__init__(config)
        self.config = config
        # self.bridge_model = PPBridge(config)
        self.bridge_model = PPBridge.load_from_checkpoint(ckpt_path, map_location=device)
        self.bridge_model = self.bridge_model.to(device)
        self.bridge_model.backbone = self.bridge_model.backbone.to(device)
        self.bridge_model.backbone.eval()
        self.bridge_model.eval()
        # self.device = device

    def preprocess(self, xT, hT, node_mask, Gt_mask=None, batch_info=None, device='cuda'):
        node_mask = node_mask.unsqueeze(-1)
        Gt_mask_ = Gt_mask.view(node_mask.size(0), node_mask.size(1), -1)
        if self.bridge_model.datamodule.startswith('Combined') or self.bridge_model.datamodule == 'QM9Dataset':
            if self.bridge_model.datamodule == 'CombinedGraphDataset':
                xT = center2zero_combined_graph(xT, node_mask, Gt_mask_)
                node_mask = node_mask.squeeze(-1)

                bs = xT.size(0)
                xT_, hT_ = [], []
                sparse_Gt_mask = []
                batch_all = []

                for batch_idx in range(bs):
                    xT_.append(xT[batch_idx][node_mask[batch_idx]])
                    hT_.append(hT[batch_idx][node_mask[batch_idx]])
                    N = node_mask[batch_idx].sum().item()    # 2 x number of nodes
                    # print('number of nodes in the current graph', N)
                    Gt_mask_batch = torch.zeros(N, device=device)
                    # print(Gt_mask_batch.size(), Gt_mask_batch)
                    Gt_mask_batch[:(N//2)] = 1
                    Gt_mask_batch = Gt_mask_batch.bool()
                    sparse_Gt_mask.append(Gt_mask_batch)
                    batch_all.append(torch.ones(N, device=device, dtype=torch.long) * batch_idx)
                xT = torch.cat(xT_, dim=0)
                hT = torch.cat(hT_, dim=0)
                Gt_mask = torch.cat(sparse_Gt_mask, dim=0)
                batch_info = torch.cat(batch_all, dim=0)
            elif self.bridge_model.datamodule == 'CombinedSparseGraphDataset' or self.bridge_model.datamodule == 'QM9Dataset':
                Gt_mask = Gt_mask_[0].squeeze(-1)
                node_mask = node_mask[0]
                xT = center2zero_sparse_graph(xT, Gt_mask, batch_info)

                if self.xT_type == 'noise':
                    # xT = xT[0]  # (1, N, 3) => (N, 3)
                    # hT = hT[0]  # (1, N, h) => (N, h)
                    bs = torch.max(batch_info).item() + 1

                    x0_, xT_, h0_, hT_ = [], [], [], []
                    for i in range(bs):
                        batch_idx = (batch_info == i)
                        # x0_batch = x0[batch_idx]
                        xT_batch = xT[batch_idx]
                        # h0_batch = h0[batch_idx]
                        hT_batch = hT[batch_idx]
                        Gt_mask_batch = Gt_mask[batch_idx].squeeze(-1)
                        # print(x0_batch.size(), h0_batch.size(), Gt_mask_batch.size())

                        # directly change x0 and h0, and rebuild xT and hT
                        # x0_Gt = x0_batch[Gt_mask_batch]
                        # x0_noise = torch.randn_like(x0_Gt, device=x0.device)
                        xT_noise = sample_zero_center_gaussian(xT_batch[Gt_mask_batch].size(), device=xT.device)
                        # x0_noise = xT_batch[~Gt_mask_batch]
                        # x0_batch_ = torch.cat([x0_Gt, x0_noise], dim=0)
                        xT_batch_ = torch.cat([xT_noise, xT_noise], dim=0)
                        # x0_.append(x0_batch_)
                        xT_.append(xT_batch_)
                        xT[batch_idx][~Gt_mask_batch] = xT_noise

                        # h0_Gt = h0_batch[Gt_mask_batch]
                        hT_noise = torch.randn_like(hT_batch[Gt_mask_batch], device=hT.device)
                        # h0_noise = hT_batch[~Gt_mask_batch]
                        # h0_batch_ = torch.cat([h0_Gt, h0_noise], dim=0)
                        hT_batch_ = torch.cat([hT_noise, hT_noise], dim=0)
                        # h0_.append(h0_batch_)
                        hT_.append(hT_batch_)
                        hT[batch_idx][~Gt_mask_batch] = hT_noise

                    xT = torch.cat(xT_, dim=0)
                    hT = torch.cat(hT_, dim=0)

        else:
            if self.xT_type == 'noise':
                xT = center2zero_with_mask(xT, node_mask, check_mask=False)
            else:
                xT = center2zero_with_mask(xT, node_mask, check_mask=True)
        
        return xT, hT, Gt_mask, batch_info

    def denoiser(self, backbone, sigma, h_t, x_t, h_T, x_T, node_mask=None, edge_mask=None, Gt_mask=None, batch_info=None, clip_denoised=False):
        if self.bridge_model.datamodule.startswith('Combined') or self.bridge_model.datamodule == 'QM9Dataset':
            _, _, denoised_h, denoised_x = self.bridge_model.denoise(backbone, sigma, h_t, x_t, h_T, x_T, Gt_mask=Gt_mask, batch_info=batch_info)
        else:
            _, _, denoised_h, denoised_x = self.bridge_model.denoise(backbone, sigma, h_t, x_t, h_T, x_T, node_mask=node_mask, edge_mask=edge_mask, Gt_mask=Gt_mask, batch_info=batch_info)
        
        # don't clip: h is aleady activated by softmax and x should not have bounded values
        if clip_denoised:
            denoised_h = denoised_h.clamp(-1, 1)
            denoised_x = denoised_x.clamp(-1, 1)
                
        return denoised_h, denoised_x

    def sample(
        self,
        x_T,
        h_T,
        steps,
        node_mask=None,
        Gt_mask=None,
        batch_info=None,
        clip_denoised=True,
        progress=False,
        callback=None,
        sigma_min=0.002,    # 1e-3
        sigma_max=80,       # 1
        rho=7.0,
        sampler="heun",
        churn_step_ratio=0.,
        guidance=1,
        device="cuda",
    ):
        x_T, h_T, Gt_mask, batch_info = self.preprocess(x_T, h_T, node_mask, Gt_mask, batch_info, device=device)
        backbone = self.bridge_model.backbone
        assert sampler in ["heun", ], 'only heun sampler is supported currently'
        
        sigmas = self.get_sigmas_karras(steps, sigma_min, sigma_max-1e-4, rho, device=device)


        sample_fn = {
            "heun": partial(self.sample_heun, beta_d=self.bridge_model.beta_d, beta_min=self.bridge_model.beta_min),
        }[sampler]

        sampler_args = dict(
                bridge_type=self.bridge_model.bridge_type, churn_step_ratio=churn_step_ratio, sigma_max=sigma_max
            )
        
        
        x_0, x_0_traj, h_0, h_0_traj, nfe = sample_fn(
            self.denoiser,
            x_T,
            h_T,
            sigmas,
            node_mask,
            Gt_mask,
            batch_info,
            progress=progress,
            callback=callback,
            guidance=guidance,
            **sampler_args,
        )
        print('nfe:', nfe)

        # return x_0.clamp(-1, 1), [x.clamp(-1, 1) for x in x_0_traj], h_0.clamp(-1, 1), [h.clamp(-1, 1) for h in h_0_traj], nfe
        return x_0, x_0_traj, h_0, h_0_traj, nfe
      
    def get_sigmas_karras(self, n, sigma_min, sigma_max, rho=7.0, device="cpu"):
        """Constructs the noise schedule of Karras et al. (2022)."""
        ramp = th.linspace(0, 1, n)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return append_zero(sigmas).to(device)


    def get_bridge_sigmas_karras(self, n, sigma_min, sigma_max, rho=7.0, eps=1e-4, device="cpu"):
        
        sigma_t_crit = sigma_max / np.sqrt(2)
        min_start_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_t_crit ** (1 / rho)
        sigmas_second_half = (max_inv_rho + th.linspace(0, 1, n//2 ) * (min_start_inv_rho - max_inv_rho)) ** rho
        sigmas_first_half = sigma_max - ((sigma_max - sigma_t_crit)  ** (1 / rho) + th.linspace(0, 1, n - n//2 +1 ) * (eps  ** (1 / rho)  - (sigma_max - sigma_t_crit)  ** (1 / rho))) ** rho
        sigmas = th.cat([sigmas_first_half.flip(0)[:-1], sigmas_second_half])
        sigmas_bridge = sigmas**2 *(1-sigmas**2/sigma_max**2)
        return append_zero(sigmas).to(device)#, append_zero(sigmas_bridge).to(device)


    def to_d(self, x, sigma, denoised, x_T, sigma_max,   w=1, stochastic=False):
        """Converts a denoiser output to a Karras ODE derivative."""
        grad_pxtlx0 = (denoised - x) / append_dims(sigma**2, x.ndim)
        grad_pxTlxt = (x_T - x) / (append_dims(th.ones_like(sigma)*sigma_max**2, x.ndim) - append_dims(sigma**2, x.ndim))
        gt2 = 2*sigma
        d = - (0.5 if not stochastic else 1) * gt2 * (grad_pxtlx0 - w * grad_pxTlxt * (0 if stochastic else 1))
        if stochastic:
            return d, gt2
        else:
            return d


    def get_d_vp(self, x, denoised, x_T, std_t,logsnr_t, logsnr_T, logs_t, logs_T, s_t_deriv, sigma_t, sigma_t_deriv, w, stochastic=False):
        
        a_t = (logsnr_T - logsnr_t + logs_t - logs_T).exp()
        b_t = -th.expm1(logsnr_T - logsnr_t) * logs_t.exp()
        
        mu_t = a_t * x_T + b_t * denoised 
        
        grad_logq = - (x - mu_t)/std_t**2 / (-th.expm1(logsnr_T - logsnr_t))
        # grad_logpxtlx0 = - (x - logs_t.exp()*denoised)/std_t**2 
        grad_logpxTlxt = -(x - th.exp(logs_t-logs_T)*x_T) /std_t**2  / th.expm1(logsnr_t - logsnr_T)

        f = s_t_deriv * (-logs_t).exp() * x
        gt2 = 2 * (logs_t).exp()**2 * sigma_t * sigma_t_deriv 
        # breakpoint()

        d = f -  gt2 * ((0.5 if not stochastic else 1)* grad_logq - w * grad_logpxTlxt)
        # d = f - (0.5 if not stochastic else 1) * gt2 * (grad_logpxtlx0 - w * grad_logpxTlxt* (0 if stochastic else 1))
        if stochastic:
            return d, gt2
        else:
            return d
        
    def compute_bridge_scaling(
        self, 
        bridge_type='both',
        progress=False,
        callback=None,
        sigma_max=80.0,
        beta_d=2,
        beta_min=0.1,
        guidance=1,
    ):

        

        if bridge_type.startswith('vp'):
            vp_snr_sqrt_reciprocal = lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
            vp_snr_sqrt_reciprocal_deriv = lambda t: 0.5 * (beta_min + beta_d * t) * (vp_snr_sqrt_reciprocal(t) + 1 / vp_snr_sqrt_reciprocal(t))
            s = lambda t: (1 + vp_snr_sqrt_reciprocal(t) ** 2).rsqrt()
            s_deriv = lambda t: -vp_snr_sqrt_reciprocal(t) * vp_snr_sqrt_reciprocal_deriv(t) * (s(t) ** 3)

            logs = lambda t: -0.25 * t ** 2 * (beta_d) - 0.5 * t * beta_min
            
            std =  lambda t: vp_snr_sqrt_reciprocal(t) * s(t)
            
            logsnr = lambda t :  - 2 * th.log(vp_snr_sqrt_reciprocal(t))

            logsnr_T = logsnr(th.as_tensor(sigma_max))
            logs_T = logs(th.as_tensor(sigma_max))

            params = {
                'std': std,
                'logsnr': logsnr,
                'logsnr_T': logsnr_T,
                'logs': logs,
                'logs_T': logs_T,
                's_deriv': s_deriv,
                'vp_snr_sqrt_reciprocal': vp_snr_sqrt_reciprocal,
                'vp_snr_sqrt_reciprocal_deriv': vp_snr_sqrt_reciprocal_deriv,
            }

        else:
            params = {}

        # x = self.sample_heun(s_in, indices, denoiser, x, sigmas, params,
        #                             bridge_type, progress, callback, sigma_max, beta_d, beta_min, churn_step_ratio, guidance)
        # h = self.sample_heun(s_in, indices, denoiser, h, sigmas, params, 
        #                             bridge_type, progress, callback, sigma_max, beta_d, beta_min, churn_step_ratio, guidance)

        return params

    @th.no_grad()
    def sample_heun(
        self, 
        denoiser,
        x,
        h,
        sigmas,
        node_mask,
        Gt_mask,
        batch_info,
        bridge_type='both',
        progress=False,
        callback=None,
        sigma_max=80.0,
        beta_d=2,
        beta_min=0.1,
        churn_step_ratio=0.,
        guidance=1,
    ):
        """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
        x_T = x
        x_traj = [x]
        h_T = h
        h_traj = [h]

        s_in = x.new_ones([x.shape[0]])

        indices = range(len(sigmas) - 1)
        if progress:
            from tqdm.auto import tqdm

            indices = tqdm(indices)
        
        assert churn_step_ratio < 1

        params = self.compute_bridge_scaling(bridge_type, progress, callback, sigma_max, beta_d, beta_min, guidance)

        if bridge_type.startswith('vp'):
            std = params['std']
            logsnr = params['logsnr']
            logsnr_T = params['logsnr_T']
            logs = params['logs']
            logs_T = params['logs_T']
            s_deriv = params['s_deriv']
            vp_snr_sqrt_reciprocal = params['vp_snr_sqrt_reciprocal']
            vp_snr_sqrt_reciprocal_deriv = params['vp_snr_sqrt_reciprocal_deriv']
        
        nfe = 0

        for j, i in enumerate(indices):
            # print('current sigma:', sigmas[i])
            
            if churn_step_ratio > 0:
                # 1 step euler
                sigma_hat = (sigmas[i+1] - sigmas[i]) * churn_step_ratio + sigmas[i]
                
                # print(x.size(), x_T.size())
                denoised_h, denoised_x = denoiser(self.bridge_model.backbone, sigmas[i] * s_in, h, x, h_T, x_T, node_mask=node_mask, Gt_mask=Gt_mask, batch_info=batch_info)
                # print('denoised_h:', denoised_h)
                # print('denoised_x:', denoised_x)
                if bridge_type == 've':
                    d_1_x, gt2_x = self.to_d(x, sigmas[i] , denoised_x, x_T, sigma_max,  w=guidance, stochastic=True)
                    d_1_h, gt2_h = self.to_d(h, sigmas[i] , denoised_h, h_T, sigma_max,  w=guidance, stochastic=True)
                elif bridge_type.startswith('vp'):
                    d_1_x, gt2_x = self.get_d_vp(x, denoised_x, x_T, std(sigmas[i]),logsnr(sigmas[i]), logsnr_T, logs(sigmas[i] ), logs_T, s_deriv(sigmas[i] ), vp_snr_sqrt_reciprocal(sigmas[i] ), vp_snr_sqrt_reciprocal_deriv(sigmas[i] ), guidance, stochastic=True)
                    d_1_h, gt2_h = self.get_d_vp(h, denoised_h, h_T, std(sigmas[i]),logsnr(sigmas[i]), logsnr_T, logs(sigmas[i] ), logs_T, s_deriv(sigmas[i] ), vp_snr_sqrt_reciprocal(sigmas[i] ), vp_snr_sqrt_reciprocal_deriv(sigmas[i] ), guidance, stochastic=True)
                
                dt = (sigma_hat - sigmas[i]) 
                x = x + d_1_x * dt + th.randn_like(x) *((dt).abs() ** 0.5)*gt2_x.sqrt()
                h = h + d_1_h * dt + th.randn_like(h) *((dt).abs() ** 0.5)*gt2_h.sqrt()
                
                nfe += 1
                
                x_traj.append(x.detach().cpu())
                h_traj.append(h.detach().cpu())
            else:
                sigma_hat =  sigmas[i]
            
            # heun step
            # print('current h:', h)
            # print('current x:', x.size())
            # print('xT:', x_T.size())
            # print(h.device, sigma_hat.device, s_in.device, x.device, h.device, h_T.device, x_T.device, node_mask.device, Gt_mask.device, batch_info.device)
            denoised_h, denoised_x = denoiser(self.bridge_model.backbone, sigma_hat * s_in, h, x, h_T, x_T, node_mask=node_mask, Gt_mask=Gt_mask, batch_info=batch_info)
            if bridge_type == 've':
                # d =  (x - denoised ) / append_dims(sigma_hat, x.ndim)
                d_x = self.to_d(x, sigma_hat, denoised_x, x_T, sigma_max, w=guidance)
                d_h = self.to_d(h, sigma_hat, denoised_h, h_T, sigma_max, w=guidance)

            elif bridge_type.startswith('vp'):
                d_x = self.get_d_vp(x, denoised_x, x_T, std(sigma_hat),logsnr(sigma_hat), logsnr_T, logs(sigma_hat), logs_T, s_deriv(sigma_hat), vp_snr_sqrt_reciprocal(sigma_hat), vp_snr_sqrt_reciprocal_deriv(sigma_hat), guidance)
                d_h = self.get_d_vp(h, denoised_h, h_T, std(sigma_hat),logsnr(sigma_hat), logsnr_T, logs(sigma_hat), logs_T, s_deriv(sigma_hat), vp_snr_sqrt_reciprocal(sigma_hat), vp_snr_sqrt_reciprocal_deriv(sigma_hat), guidance)
                
            nfe += 1
            if callback is not None:
                callback(
                    {
                        "x": x,
                        "h": h,
                        "i": i,
                        "sigma": sigmas[i],
                        "sigma_hat": sigma_hat,
                        "denoised_x": denoised_x,
                        "denoised_h": denoised_h,
                    }
                )
            dt = sigmas[i + 1] - sigma_hat
            if sigmas[i + 1] == 0:
                
                x = x + d_x * dt 
                h = h + d_h * dt 
                # print('final h:', h)
                # print('final x:', x)

                
            else:
                # Heun's method
                x_2 = x + d_x * dt
                h_2 = h + d_h * dt
                denoised_2_h, denoised_2_x = denoiser(self.bridge_model.backbone, sigmas[i + 1] * s_in, h_2, x_2, h_T, x_T, node_mask=node_mask, Gt_mask=Gt_mask, batch_info=batch_info)
                if bridge_type == 've':
                    # d_2 =  (x_2 - denoised_2) / append_dims(sigmas[i + 1], x.ndim)
                    d_2_x = self.to_d(x_2,  sigmas[i + 1], denoised_2_x, x_T, sigma_max, w=guidance)
                    d_2_h = self.to_d(h_2,  sigmas[i + 1], denoised_2_h, h_T, sigma_max, w=guidance)
                elif bridge_type.startswith('vp'):
                    d_2_x = self.get_d_vp(x_2, denoised_2_x, x_T, std(sigmas[i + 1]),logsnr(sigmas[i + 1]), logsnr_T, logs(sigmas[i + 1]), logs_T, s_deriv(sigmas[i + 1]),
                                    vp_snr_sqrt_reciprocal(sigmas[i + 1]), vp_snr_sqrt_reciprocal_deriv(sigmas[i + 1]), guidance)
                    d_2_h = self.get_d_vp(h_2, denoised_2_h, h_T, std(sigmas[i + 1]),logsnr(sigmas[i + 1]), logsnr_T, logs(sigmas[i + 1]), logs_T, s_deriv(sigmas[i + 1]),
                                    vp_snr_sqrt_reciprocal(sigmas[i + 1]), vp_snr_sqrt_reciprocal_deriv(sigmas[i + 1]), guidance)
                
                d_prime_x = (d_x + d_2_x) / 2
                d_prime_h = (d_h + d_2_h) / 2

                # noise = th.zeros_like(x) if 'flow' in bridge_type or bridge_type == 'uncond' else generator.randn_like(x)
                x = x + d_prime_x * dt #+ noise * (sigmas[i + 1]**2 - sigma_hat**2).abs() ** 0.5
                h = h + d_prime_h * dt #+ noise * (sigmas[i + 1]**2 - sigma_hat**2).abs() ** 0.5
                nfe += 1
                # print('current h:', h)
                # print('current x:', x)


            # loss = (denoised.detach().cpu() - x0).pow(2).mean().item()
            # losses.append(loss)

            x_traj.append(x.detach().cpu())
            h_traj.append(h.detach().cpu())
            
        return x, x_traj, h, h_traj, nfe