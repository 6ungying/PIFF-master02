# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import argparse
import numpy as np
from tqdm import tqdm
from functools import partial
import torch
import omegaconf
import yaml

from .util import unsqueeze_xdim
from i2sb.VQGAN.vqgan import VQModel

from ipdb import set_trace as debug

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def namespace2dict(config):
    conf_dict = {}
    for key, value in vars(config).items():
        if isinstance(value, argparse.Namespace):
            conf_dict[key] = namespace2dict(value)
        else:
            conf_dict[key] = value
    return conf_dict

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict) or isinstance(value, omegaconf.dictconfig.DictConfig):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def create_model_config():
    # --------------- basic ---------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",         type=str,   default='C:\\Users\\THINKLAB\\Desktop\\PIFF-master02\\configs\\flooding.yaml',        help="config file path")

    opt, _ = parser.parse_known_args()
    
    with open(opt.config, 'r') as f:
        dict_config = yaml.load(f, Loader=yaml.FullLoader)

    namespace_config = dict2namespace(dict_config)
    dict_config = namespace2dict(namespace_config)

    return namespace_config.model

def compute_gaussian_product_coef(sigma1, sigma2):
    """ Given p1 = N(x_t|x_0, sigma_1**2) and p2 = N(x_t|x_1, sigma_2**2)
        return p1 * p2 = N(x_t| coef1 * x0 + coef2 * x1, var) """

    denom = sigma1**2 + sigma2**2
    coef1 = sigma2**2 / denom
    coef2 = sigma1**2 / denom
    var = (sigma1**2 * sigma2**2) / denom
    return coef1, coef2, var

class Diffusion():
    def __init__(self, betas, device):

        self.device = device

        # compute analytic std: eq 11
        std_fwd = np.sqrt(np.cumsum(betas))
        std_bwd = np.sqrt(np.flip(np.cumsum(np.flip(betas))))
        mu_x0, mu_x1, var = compute_gaussian_product_coef(std_fwd, std_bwd)
        std_sb = np.sqrt(var)

        # tensorize everything
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.betas = to_torch(betas).to(device)
        self.std_fwd = to_torch(std_fwd).to(device)
        self.std_bwd = to_torch(std_bwd).to(device)
        self.std_sb  = to_torch(std_sb).to(device)
        self.mu_x0 = to_torch(mu_x0).to(device)
        self.mu_x1 = to_torch(mu_x1).to(device)

    def get_std_fwd(self, step, xdim=None):
        std_fwd = self.std_fwd[step]
        return std_fwd if xdim is None else unsqueeze_xdim(std_fwd, xdim)

    def q_sample(self, step, x0, x1, ot_ode=False):
        """ Sample q(x_t | x_0, x_1), i.e. eq 11 """

        assert x0.shape == x1.shape
        batch, *xdim = x0.shape

        mu_x0  = unsqueeze_xdim(self.mu_x0[step],  xdim)
        mu_x1  = unsqueeze_xdim(self.mu_x1[step],  xdim)
        std_sb = unsqueeze_xdim(self.std_sb[step], xdim)

        xt = mu_x0 * x0 + mu_x1 * x1
        if not ot_ode:
            xt = xt + std_sb * torch.randn_like(xt)
        return xt.detach()

    def p_posterior(self, nprev, n, x_n, x0, ot_ode=False):
        """ Sample p(x_{nprev} | x_n, x_0), i.e. eq 4"""

        assert nprev < n
        std_n     = self.std_fwd[n]
        std_nprev = self.std_fwd[nprev]
        std_delta = (std_n**2 - std_nprev**2).sqrt()

        mu_x0, mu_xn, var = compute_gaussian_product_coef(std_nprev, std_delta)

        xt_prev = mu_x0 * x0 + mu_xn * x_n
        if not ot_ode and nprev > 0:
            xt_prev = xt_prev + var.sqrt() * torch.randn_like(xt_prev)

        return xt_prev

    def ddpm_sampling(self, steps, pred_x0_fn, x1, rainfall_emb, mask=None, ot_ode=False, log_steps=None, verbose=True, ode_method=None):
        xt = x1.detach().to(self.device)

        xs = []
        pred_x0s = []

        log_steps = log_steps or steps
        assert steps[0] == log_steps[0] == 0

        if not ode_method:
            steps = steps[::-1]

            pair_steps = zip(steps[1:], steps[:-1])
            pair_steps = tqdm(pair_steps, desc='DDPM sampling', total=len(steps)-1) if verbose else pair_steps
            for prev_step, step in pair_steps:
                assert prev_step < step, f"{prev_step=}, {step=}"

                pred_x0 = pred_x0_fn(x1, xt, rainfall_emb, step)
                xt = self.p_posterior(prev_step, step, xt, pred_x0, ot_ode=ot_ode)

                if prev_step in log_steps:
                    pred_x0s.append(pred_x0.detach().cpu())
                    xs.append(xt.detach().cpu())

            stack_bwd_traj = lambda z: torch.flip(torch.stack(z, dim=1), dims=(1,))
            return stack_bwd_traj(xs), stack_bwd_traj(pred_x0s)
        elif ode_method == 'ddim':
            step_size = 1 / len(steps)
            t = 1
            for i in range(len(steps)):
                drift = pred_x0_fn(x1, xt, rainfall_emb, t, ode_method)
                x0 = x1 - drift
                t = t - step_size
                xt = (1-t) * x0 + t * x1

                xs.append(x0.detach().cpu())
            stack_bwd_traj = lambda z: torch.flip(torch.stack(z, dim=1), dims=(1,))
            return stack_bwd_traj(xs), stack_bwd_traj(xs)
        elif ode_method == 'ddpm':
            step_size = 1 / len(steps)
            t = 1
            for i in range(len(steps)):
                drift = pred_x0_fn(x1, xt, rainfall_emb, t, ode_method)
                x0 = x1 - drift
                t = t - step_size
                diffusion_coefficient = (t**2 * (1-t)**2) / (t**2 + (1-t)**2)
                xt = (1-t) * x0 + t * x1 + torch.randn_like(xt) * np.sqrt(diffusion_coefficient) 

                xs.append(x0.detach().cpu())
            stack_bwd_traj = lambda z: torch.flip(torch.stack(z, dim=1), dims=(1,))
            return stack_bwd_traj(xs), stack_bwd_traj(xs)
        
        elif ode_method == 'euler':
            step_size = 1 / len(steps)
            t = 1
            # t = t.view(-1, 1, 1, 1)
            for i in range(len(steps)):
                drift = pred_x0_fn(x1, xt, rainfall_emb, t, ode_method)
                xt = xt - step_size * drift
                t = t - step_size

                xs.append(xt.detach().cpu())
            stack_bwd_traj = lambda z: torch.flip(torch.stack(z, dim=1), dims=(1,))
            return stack_bwd_traj(xs), stack_bwd_traj(xs)
        
        elif ode_method == 'heun':
            step_size = 1 / len(steps)
            t = 1
            # t = t.view(-1, 1, 1, 1)
            for i in range(len(steps)):
                drift = pred_x0_fn(x1, xt, rainfall_emb, t, ode_method)
                drift_p = pred_x0_fn(x1, xt - step_size * drift, rainfall_emb, t - step_size, ode_method)

                xt = xt - step_size * (drift + drift_p) / 2
                t = t - step_size

                xs.append(xt.detach().cpu())
            stack_bwd_traj = lambda z: torch.flip(torch.stack(z, dim=1), dims=(1,))
            return stack_bwd_traj(xs), stack_bwd_traj(xs)
        elif ode_method == 'rk4':
            step_size = 1 / len(steps)
            t = 1
            # t = t.view(-1, 1, 1, 1)
            for i in range(len(steps)):
                drift = pred_x0_fn(x1, xt, rainfall_emb, t, ode_method)
                k1 = step_size * drift

                drift_p = pred_x0_fn(x1, xt - k1 / 2, rainfall_emb, t - step_size / 2, ode_method)
                k2 = step_size * drift_p

                drift_p = pred_x0_fn(x1, xt - k2 / 2, rainfall_emb, t - step_size / 2, ode_method)
                k3 = step_size * drift_p

                drift_p = pred_x0_fn(x1, xt - k3, rainfall_emb, t - step_size, ode_method)
                k4 = step_size * drift_p

                xt = xt - (k1 + 2 * k2 + 2 * k3 + k4) / 6
                t = t - step_size

                xs.append(xt.detach().cpu())
            stack_bwd_traj = lambda z: torch.flip(torch.stack(z, dim=1), dims=(1,))
            return stack_bwd_traj(xs), stack_bwd_traj(xs)
        
        elif ode_method == 'euler-maruyama':
            step_size = 1 / len(steps)
            t = 1
            # t = t.view(-1, 1, 1, 1)
            for i in range(len(steps)):
                drift = pred_x0_fn(x1, xt, rainfall_emb, t, ode_method)
                # diffusion_coefficient = (t**2 * (1-t)**2) / (t**2 + (1-t)**2) 
                # diffusion_coefficient = (t * (1-t)) ** (1/2)
                diffusion_coefficient = 0.1
                xt = xt - step_size * drift + torch.randn_like(xt) * np.sqrt(step_size) * diffusion_coefficient

                t = t - step_size

                xs.append(xt.detach().cpu())
            stack_bwd_traj = lambda z: torch.flip(torch.stack(z, dim=1), dims=(1,))
            return stack_bwd_traj(xs), stack_bwd_traj(xs)