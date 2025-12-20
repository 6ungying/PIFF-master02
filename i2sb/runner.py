# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import numpy as np
import pickle

import torch
import torch.nn.functional as F
from torch.optim import AdamW, lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP

from torch_ema import ExponentialMovingAverage
import torchvision.utils as tu
import torchmetrics
from tqdm import tqdm

import distributed_util as dist_util
from evaluation import build_resnet50
from matplotlib import pyplot as plt

from . import util
from .network import Image256Net
from .diffusion import Diffusion, disabled_train, create_model_config
from i2sb.VQGAN.vqgan import VQModel
from i2sb.base.modules.encoders.modules import SpatialRescaler
from torch.utils.data import DataLoader
from corruption.mixture import floodDataset
from .embedding import RainfallEmbedder

from ipdb import set_trace as debug
from i2sb.physics_loss import PhysicsInformedLoss

def build_optimizer_sched(opt, rainfall_embber, net, log):

    optim_dict = {"lr": opt.lr, 'weight_decay': opt.l2_norm}
    params = list(net.parameters()) + list(rainfall_embber.parameters())
    optimizer = AdamW(params, **optim_dict)
    log.info(f"[Opt] Built AdamW optimizer {optim_dict=}!")

    if opt.lr_gamma < 1.0:
        sched_dict = {"step_size": opt.lr_step, 'gamma': opt.lr_gamma}
        sched = lr_scheduler.StepLR(optimizer, **sched_dict)
        log.info(f"[Opt] Built lr step scheduler {sched_dict=}!")
    else:
        sched = None

    if opt.load:
        checkpoint = torch.load(opt.load, map_location="cpu")
        if "optimizer" in checkpoint.keys():
            optimizer.load_state_dict(checkpoint["optimizer"])
            log.info(f"[Opt] Loaded optimizer ckpt {opt.load}!")
        else:
            log.warning(f"[Opt] Ckpt {opt.load} has no optimizer!")
        if sched is not None and "sched" in checkpoint.keys() and checkpoint["sched"] is not None:
            sched.load_state_dict(checkpoint["sched"])
            log.info(f"[Opt] Loaded lr sched ckpt {opt.load}!")
        else:
            log.warning(f"[Opt] Ckpt {opt.load} has no lr sched!")

    return optimizer, sched

def make_beta_schedule(n_timestep=1000, linear_start=1e-4, linear_end=2e-2):
    # return np.linspace(linear_start, linear_end, n_timestep)
    betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return betas.numpy()

def all_cat_cpu(opt, log, t):
    if not opt.distributed: return t.detach().cpu()
    gathered_t = dist_util.all_gather(t.to(opt.device), log=log)
    return torch.cat(gathered_t).detach().cpu()

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def plot_grad_flow(model):
    ave_grads = []
    layers = []
    for n, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().item())
    fig = plt.figure(figsize=(10,5))
    plt.plot(ave_grads, alpha=0.5, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k")
    # plt.xticks(range(0,len(ave_grads),1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("Average Gradient Magnitude")
    plt.title("Gradient Flow")
    plt.grid(True)
    plt.show()

class Runner(object):
    def __init__(self, opt, log, save_opt=True):
        super(Runner,self).__init__()

        self.opt = opt
        self.model_config = create_model_config()
        if opt.latent_space:
            self.vqgan = VQModel(**vars(self.model_config.VQGAN.params)).eval()
            self.vqgan.train = disabled_train
            for param in self.vqgan.parameters():
                param.requires_grad = False
            print(f"load vqgan from {self.model_config.VQGAN.params.ckpt_path}")
            
            self.cond_stage_model = SpatialRescaler(**vars(self.model_config.CondStageParams))
            self.vqgan.to(opt.device)
            self.cond_stage_model.to(opt.device)

        # Save opt.
        if save_opt:
            opt_pkl_path = opt.ckpt_path / "options.pkl"
            with open(opt_pkl_path, "wb") as f:
                pickle.dump(opt, f)
            log.info("Saved options pickle to {}!".format(opt_pkl_path))

        betas = make_beta_schedule(n_timestep=opt.interval, linear_end=opt.beta_max / opt.interval)
        betas = np.concatenate([betas[:opt.interval//2], np.flip(betas[:opt.interval//2])])
        self.diffusion = Diffusion(betas, opt.device)
        log.info(f"[Diffusion] Built I2SB diffusion: steps={len(betas)}!")

        noise_levels = torch.linspace(opt.t0, opt.T, opt.interval, device=opt.device) * opt.interval
        self.net = Image256Net(log, noise_levels=noise_levels, use_fp16=opt.use_fp16, cond=opt.cond_x1, spm=opt.spm)
        self.rainfall_emb = RainfallEmbedder(256, 1)
        params = list(self.net.parameters()) + list(self.rainfall_emb.parameters())
        self.ema = ExponentialMovingAverage(params, decay=opt.ema)

        if opt.load:
            checkpoint = torch.load(opt.load, map_location="cpu")
            self.net.load_state_dict(checkpoint['net'])
            log.info(f"[Net] Loaded network ckpt: {opt.load}!")
            self.ema.load_state_dict(checkpoint["ema"])
            log.info(f"[Ema] Loaded ema ckpt: {opt.load}!")
            self.rainfall_emb.load_state_dict(checkpoint['embedding'])
            log.info(f"[Embedding] Loaded embedding ckpt: {opt.load}!")
            if opt.normalize_latent:
                self.net.ori_latent_mean = checkpoint["ori_latent_mean"]
                self.net.ori_latent_std = checkpoint["ori_latent_std"]
                self.net.cond_latent_mean = checkpoint["cond_latent_mean"]
                self.net.cond_latent_std = checkpoint["cond_latent_std"]
                log.info(f"[Latent] Loaded latent mean/std ckpt: {opt.load}!")

        self.net.to(opt.device)
        self.ema.to(opt.device)
        self.rainfall_emb.to(opt.device)

        # Initialize physics loss if enabled (質量守恆方程)
        if getattr(opt, 'use_physics', False):
            try:
                # 初始化質量守恆物理損失
                self.physics_loss = PhysicsInformedLoss(
                    dx=getattr(opt, 'dx', 20.0),
                    dy=getattr(opt, 'dy', 20.0),
                    dt=getattr(opt, 'dt', 3600.0)  # 1小時 = 3600秒
                )
                self.physics_loss.to(opt.device)
                self.physics_weight = getattr(opt, 'physics_weight', 1.0)
                log.info(f"[Physics] Enabled mass conservation loss with weight={self.physics_weight}")
            except Exception as e:
                log.error(f"[Physics] Failed to initialize PhysicsInformedLoss: {e}")
                self.physics_loss = None
        else:
            self.physics_loss = None
            log.info("[Physics] Physics-informed loss disabled")

        self.log = log

        if opt.eval:
            self.net.ori_latent_mean = self.net.ori_latent_mean.to(opt.device)
            self.net.ori_latent_std = self.net.ori_latent_std.to(opt.device)
            self.net.cond_latent_mean = self.net.cond_latent_mean.to(opt.device)
            self.net.cond_latent_std = self.net.cond_latent_std.to(opt.device)

    def logger(self, msg, **kwargs):
        print(msg, **kwargs)

    def get_latent_mean_std(self):
        train_dataset = floodDataset(True)
        train_loader = DataLoader(train_dataset,
                                  batch_size=128,
                                  shuffle=True,
                                  num_workers=8,
                                  drop_last=True)

        total_ori_mean = None
        total_ori_var = None
        total_cond_mean = None
        total_cond_var = None
        # max_batch_num = 30000 // self.config.data.train.batch_size

        def calc_mean(batch, total_ori_mean=None, total_cond_mean=None):
            (x, x_cond, _) = batch
            x = x.to(self.opt.device)
            x_cond = x_cond.to(self.opt.device)

            x_latent = self.vqgan.encoder(x)
            x_cond_latent = self.vqgan.encoder(x_cond)
            x_mean = x_latent.mean(axis=[0, 2, 3], keepdim=True)
            total_ori_mean = x_mean if total_ori_mean is None else x_mean + total_ori_mean

            x_cond_mean = x_cond_latent.mean(axis=[0, 2, 3], keepdim=True)
            total_cond_mean = x_cond_mean if total_cond_mean is None else x_cond_mean + total_cond_mean
            return total_ori_mean, total_cond_mean

        def calc_var(batch, ori_latent_mean=None, cond_latent_mean=None, total_ori_var=None, total_cond_var=None):
            (x, x_cond, _) = batch
            x = x.to(self.opt.device)
            x_cond = x_cond.to(self.opt.device)

            x_latent = self.vqgan.encoder(x)
            x_cond_latent = self.vqgan.encoder(x_cond)
            x_var = ((x_latent - ori_latent_mean) ** 2).mean(axis=[0, 2, 3], keepdim=True)
            total_ori_var = x_var if total_ori_var is None else x_var + total_ori_var

            x_cond_var = ((x_cond_latent - cond_latent_mean) ** 2).mean(axis=[0, 2, 3], keepdim=True)
            total_cond_var = x_cond_var if total_cond_var is None else x_cond_var + total_cond_var
            return total_ori_var, total_cond_var

        self.logger(f"start calculating latent mean")
        batch_count = 0
        for train_batch in tqdm(train_loader, total=len(train_loader), smoothing=0.01):
            # if batch_count >= max_batch_num:
            #     break
            batch_count += 1
            total_ori_mean, total_cond_mean = calc_mean(train_batch, total_ori_mean, total_cond_mean)

        ori_latent_mean = total_ori_mean / batch_count
        self.net.ori_latent_mean = ori_latent_mean

        cond_latent_mean = total_cond_mean / batch_count
        self.net.cond_latent_mean = cond_latent_mean

        self.logger(f"start calculating latent std")
        batch_count = 0
        for train_batch in tqdm(train_loader, total=len(train_loader), smoothing=0.01):
            # if batch_count >= max_batch_num:
            #     break
            batch_count += 1
            total_ori_var, total_cond_var = calc_var(train_batch,
                                                     ori_latent_mean=ori_latent_mean,
                                                     cond_latent_mean=cond_latent_mean,
                                                     total_ori_var=total_ori_var,
                                                     total_cond_var=total_cond_var)
            # break

        ori_latent_var = total_ori_var / batch_count
        cond_latent_var = total_cond_var / batch_count

        self.net.ori_latent_std = torch.sqrt(ori_latent_var)
        self.net.cond_latent_std = torch.sqrt(cond_latent_var)
        # self.logger(self.net.ori_latent_mean)
        # self.logger(self.net.ori_latent_std)
        # self.logger(self.net.cond_latent_mean)
        # self.logger(self.net.cond_latent_std)
    
    @torch.no_grad()
    def encode(self, x, cond=True):
        normalize = self.opt.normalize_latent 
        model = self.vqgan
        x_latent = model.encoder(x)
        if not self.model_config.latent_before_quant_conv:
            x_latent = model.quant_conv(x_latent)
        if normalize:
            if cond:
                x_latent = (x_latent - self.net.cond_latent_mean) / self.net.cond_latent_std
            else:
                x_latent = (x_latent - self.net.ori_latent_mean) / self.net.ori_latent_std
        return x_latent

    @torch.no_grad()
    def decode(self, x_latent, cond=True):
        normalize = self.opt.normalize_latent
        if normalize:
            if cond:
                x_latent = x_latent * self.net.cond_latent_std + self.net.cond_latent_mean
            else:
                x_latent = x_latent * self.net.ori_latent_std + self.net.ori_latent_mean
        model = self.vqgan
        if self.model_config.latent_before_quant_conv:
            x_latent = model.quant_conv(x_latent)
        x_latent_quant, loss, _ = model.quantize(x_latent)
        out = model.decode(x_latent_quant)
        return out
    
    def compute_label(self, step, x0, xt, x1):
        """ Eq 12 """
        std_fwd = self.diffusion.get_std_fwd(step, xdim=x0.shape[1:])
        label = (xt - x0) / std_fwd
        # label = x1 - x0
        return label.detach()

    def compute_pred_x0(self, step, xt, x1, net_out, clip_denoise=False):
        """ Given network output, recover x0. This should be the inverse of Eq 12 """
        std_fwd = self.diffusion.get_std_fwd(step, xdim=xt.shape[1:])
        # pred_x0 = x1 - net_out
        pred_x0 = xt - std_fwd * net_out
        if clip_denoise: pred_x0.clamp_(-1., 1.)
        return pred_x0

    def sample_batch(self, opt, loader, corrupt_method):
        # loader 是由 setup_loader 包出來的 generator
        # 這裡直接取一個 batch
        (
            flood_img,   # [N, 1, H, W]
            vx_img,      # [N, 1, H, W]
            vy_img,      # [N, 1, H, W]
            dem_img,     # [N, 1, H, W]
            mask_flood,  # [N, 1, H, W]
            mask_vx,     # [N, 1, H, W]
            mask_vy,     # [N, 1, H, W]
            y,           # 其它標籤 / 參數
            img_name,    # 檔名
            vx_img_name,
            vy_img_name,
            spm,         # 你的 SPM 條件
            next_timestep_data  # 下一時間步資料 (用於物理損失)
        ) = next(loader)

        # 如果 next_timestep_data 為 None，表示沒有下一時間步（例如最後一筆資料）
        # 物理損失會自動跳過

        # --------- 確保都是單通道，然後 concat 成 3 通道 ----------
        def ensure_single_channel(t):
            if t.dim() == 4 and t.shape[1] > 1:
                return t[:, 0:1, :, :]
            return t

        flood_img = ensure_single_channel(flood_img)
        vx_img    = ensure_single_channel(vx_img)
        vy_img    = ensure_single_channel(vy_img)
        dem_img   = ensure_single_channel(dem_img)

        # [N, 3, H, W]
        clean_img   = torch.cat([flood_img, vx_img, vy_img], dim=1)   # h, u, v
        mask        = torch.cat([mask_flood, mask_vx, mask_vy], dim=1)
        corrupt_img = torch.cat([dem_img, dem_img, dem_img], dim=1)   # dem 複製成 3 通道

        # --------- NaN 處理：淹水 / 流速 / DEM ----------
        # 淹水深度 NaN -> 小正值 0.01
        if torch.isnan(clean_img).any():
            flood_nan = torch.isnan(clean_img[:, 0:1])
            clean_img[:, 0:1] = torch.where(flood_nan,
                                            torch.tensor(0.01, device=clean_img.device),
                                            clean_img[:, 0:1])

            # 流速 NaN -> 0
            vel_nan = torch.isnan(clean_img[:, 1:])
            clean_img[:, 1:] = torch.where(vel_nan,
                                           torch.tensor(0.0, device=clean_img.device),
                                           clean_img[:, 1:])

        # DEM NaN -> 該 channel 有效值平均；全部都是 NaN 就設為 0
        if torch.isnan(corrupt_img).any():
            for i in range(corrupt_img.shape[0]):
                for c in range(corrupt_img.shape[1]):
                    cd = corrupt_img[i, c]
                    if torch.isnan(cd).any():
                        valid = ~torch.isnan(cd)
                        if valid.any():
                            mean_val = cd[valid].mean()
                            corrupt_img[i, c] = torch.where(torch.isnan(cd), mean_val, cd)
                        else:
                            corrupt_img[i, c] = torch.zeros_like(cd)

        if torch.isnan(mask).any():
            mask = torch.nan_to_num(mask, nan=0.0)

        # --------- 丟到 device 上 ----------
        device = opt.device
        y      = y.to(device)
        x0     = clean_img.to(device)
        x1     = corrupt_img.to(device)
        mask   = mask.to(device)
        spm    = spm.to(device)

        # 條件：用 x1 當條件 or 不用
        cond = x1 if opt.cond_x1 else None

        # 如果要對 x1 加噪音
        if getattr(opt, "add_x1_noise", False):
            x1 = x1 + torch.randn_like(x1)

        assert x0.shape == x1.shape, f"x0/x1 shape mismatch: {x0.shape} vs {x1.shape}"

        # --------- latent space 模式 ----------
        if opt.latent_space:
            x0 = self.encode(x0, cond=False)
            x1 = self.encode(x1, cond=False)
            if cond is not None:
                cond = self.cond_stage_model(cond)
            if mask is not None:
                mask = self.cond_stage_model(mask)

        # 目前先讓 next_timestep_data = None，物理 loss 之後再補
        return x0, x1, mask, y, cond, spm, next_timestep_data


    def train(self, opt, train_dataset, val_dataset, corrupt_method):
        gradient_list = []
        embedder_gradient_list = []
        losses = []

        def plot_losses():
            plt.figure(figsize=(12, 8))
            plt.plot(np.log(losses))
            plt.xlabel("Iterations")
            plt.ylabel("Log Loss")
            plt.title("Log Loss per Iteration")
            plt.savefig("C:\\Users\\THINKLAB\\Desktop\\PIFF-master02\\results\\losses_log_scale_otode_continuous_128_norm_SDE.png")

        def plot_model_gradients(model, embedder):
            def format_scientific(val):
                # Format the number using scientific notation with no decimal places.
                s = f"{val:.0e}"  # e.g., 0.000001 --> "1e-06"
                # Remove any extra zero in the exponent, e.g., change "e-06" to "e-6"
                s = s.replace("e-0", "e-")
                s = s.replace("e+0", "e+")
                return s
            gradients = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    # Compute the mean absolute value of the gradient
                    grad_norm = param.grad.abs().mean().item()
                    gradients[name] = grad_norm
            gradient_list.append(gradients)
            # print(model)
            print("Top 25 parameters with the largest gradient magnitudes:")
            sorted_gradients = sorted(gradients.items(), key=lambda x: x[1], reverse=True)[:25]
            for param_name, grad_val in sorted_gradients:
                print(f"{param_name}: {format_scientific(grad_val)}")

            print("\nBottom 25 parameters with the smallest gradient magnitudes:")
            sorted_gradients_low = sorted(gradients.items(), key=lambda x: x[1])[:25]
            for param_name, grad_val in sorted_gradients_low:
                print(f"{param_name}: {format_scientific(grad_val)}")

            plt.figure(figsize=(12, 8))
            for epoch, gradients in enumerate(gradient_list):
                keys = list(gradients.keys())
                values = [gradients[key] for key in keys]

                plt.plot(range(len(keys)), np.log(values), label=f"Iter {(epoch+1)*200}")
            
            # plt.xticks(range(len(keys)), keys, rotation=45, ha='right')
            plt.xlabel("Parameters")
            plt.ylabel("Log Mean Gradient Magnitude")
            plt.title("Log Gradient Magnitude per Parameter Across 200 iterations")
            plt.legend()
            plt.tight_layout()
            plt.savefig("C:\\Users\\THINKLAB\\Desktop\\PIFF-master02\\results\\gradient_plot.png")

            embedder_gradients = {}
            for name, param in embedder.named_parameters():
                if param.grad is not None:
                    # Compute the mean absolute value of the gradient
                    grad_norm = param.grad.abs().mean().item()
                    embedder_gradients[name] = grad_norm
            
            embedder_gradient_list.append(embedder_gradients)
            print("Top 25 parameters with the largest gradient magnitudes:")
            sorted_gradients = sorted(embedder_gradients.items(), key=lambda x: x[1], reverse=True)[:25]
            for param_name, grad_val in sorted_gradients:
                print(f"{param_name}: {format_scientific(grad_val)}")
            print("\nBottom 25 parameters with the smallest gradient magnitudes:")
            sorted_gradients_low = sorted(embedder_gradients.items(), key=lambda x: x[1])[:25]
            for param_name, grad_val in sorted_gradients_low:
                print(f"{param_name}: {format_scientific(grad_val)}")

            fig = plt.figure(figsize=(12, 8))
            for epoch, gradients in enumerate(embedder_gradient_list):
                keys = list(gradients.keys())
                values = [gradients[key] for key in keys]

                plt.plot(range(len(keys)), np.log(values), label=f"Iter {(epoch+1)*200}")
            
            # plt.xticks(range(len(keys)), keys, rotation=45, ha='right')
            plt.xlabel("Parameters")
            plt.ylabel("Log Mean Gradient Magnitude")
            plt.title("Log Gradient Magnitude per Parameter Across 200 iterations")
            plt.legend()
            plt.tight_layout()
            plt.savefig("C:\\Users\\THINKLAB\\Desktop\\PIFF-master02\\results\\embedder_gradient_plot.png")

        self.writer = util.build_log_writer(opt)
        log = self.log

        net = self.net
        ema = self.ema
        # print(net)
        rainfall_embber = self.rainfall_emb
        print(rainfall_embber)
        optimizer, sched = build_optimizer_sched(opt, rainfall_embber, net, log)

        train_loader = util.setup_loader(train_dataset, opt.microbatch)
        val_loader   = util.setup_loader(val_dataset,   opt.microbatch)

        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=2).to(opt.device)
        self.resnet = build_resnet50().to(opt.device)

        net.train()
        n_inner_loop = opt.batch_size // (opt.global_size * opt.microbatch)
        for it in range(opt.num_itr):
            optimizer.zero_grad()

            for _ in range(n_inner_loop):
                # ===== sample boundary pair =====
                x0, x1, mask, y, cond, spm, next_timestep_data = self.sample_batch(opt, train_loader, corrupt_method)

                # ===== compute loss =====
                if opt.timestep_importance == 'continuous':
                    t1, t0 = 1, 0
                    step = torch.rand((x0.shape[0],)) * (t1 - t0)
                    # make step shape = (x0.shape[0], 1, 1, 1)
                    step = step.view(-1, 1, 1, 1).to(x0.device)
                    if opt.ot_ode:
                        xt = (1-step) * x0 + step * x1
                        label = x1 - x0
                    if not opt.ot_ode:
                        # var = (step**2 * (1-step)**2) / (step**2 + (1-step)**2)
                        # randn + loss x1-x0 perform good
                        var = step * (1-step)
                        rand = torch.randn_like(x0) * 0.1
                        xt = (1-step) * x0 + step * x1 + rand 
                        dvar = (1-2*step) / (2*var.sqrt() + 1e-2) 
                        label = x1 - x0 
                else:
                    step = torch.randint(0, opt.interval, (x0.shape[0],))
                    step = step.to(x0.device)
                    xt = self.diffusion.q_sample(step, x0, x1, ot_ode=opt.ot_ode)
                    label = self.compute_label(step, x0, xt, x1)


                # xt = self.diffusion.q_sample(step, x0, x1, ot_ode=opt.ot_ode)
                # label = self.compute_label(step, x0, xt, x1)

                rainfall_emb = rainfall_embber(y)
                pred = net(xt, step, rainfall_emb, cond=cond, spm=spm)
                assert xt.shape == label.shape == pred.shape

                # Check for NaN in inputs before loss calculation (silent handling)
                if torch.isnan(xt).any() or torch.isnan(label).any() or torch.isnan(pred).any():
                    continue  # Skip this batch silently

                # MSE loss - 註解 mask 部分，因為物理損失需要計算完整的值
                # if mask is not None:
                #     pred_mask = mask * pred
                #     label_mask = mask * label
                #     mse_loss = F.mse_loss(pred, label) + F.mse_loss(pred_mask, label_mask) * 0
                # else:
                #     mse_loss = F.mse_loss(pred, label)
                
                # 直接計算完整 MSE loss（不使用 mask）
                mse_loss = F.mse_loss(pred, label)
                
                # Add numerical stability check for MSE loss (silent handling)
                if torch.isnan(mse_loss) or torch.isinf(mse_loss):
                    continue  # Skip this batch silently
                
                # Physics loss (if enabled) - 質量守恆方程
                total_loss = mse_loss
                physics_loss_value = torch.tensor(0.0, device=opt.device)
                phys_summary = {}

                if self.physics_loss is not None and pred is not None and not getattr(opt, 'latent_space', False) and next_timestep_data is not None:
                    # 檢查 batch 中是否有有效的下一時間步資料
                    # next_timestep_data 是一個 list，每個元素是 (next_flood, next_mask) 或 None
                    valid_indices = [i for i, data in enumerate(next_timestep_data) if data is not None]
                    
                    if len(valid_indices) > 0:
                        # 用網路輸出重建 x0_hat (當前時間步預測)
                        if opt.timestep_importance == 'continuous':
                            x0_hat = x1 - pred
                        else:
                            x0_hat = self.compute_pred_x0(step, xt, x1, pred, clip_denoise=False)

                        if x0_hat.shape[1] >= 3:
                            # 只處理有下一時間步資料的樣本
                            for idx in valid_indices:
                                try:
                                    # 當前時間步 t 的預測值 (單個樣本)
                                    pred_h_t = x0_hat[idx:idx+1, 0:1, :, :]  # 水深
                                    pred_u_t = x0_hat[idx:idx+1, 1:2, :, :]  # x方向流速
                                    pred_v_t = x0_hat[idx:idx+1, 2:3, :, :]  # y方向流速

                                    # 下一時間步 t+1 的資料
                                    next_data = next_timestep_data[idx]
                                    # next_data 包含: (next_flood, next_vx, next_vy, next_mask_flood, next_mask_vx, next_mask_vy)
                                    next_flood, next_vx, next_vy, next_mask_flood, next_mask_vx, next_mask_vy = next_data
                                    
                                    # next_flood 已經是 tensor 且已標準化 (在 get_next_timestep_data 中處理)
                                    # 確保是 4D tensor [1, C, H, W]
                                    if next_flood.dim() == 3:
                                        next_flood = next_flood.unsqueeze(0)
                                    
                                    pred_h_t1 = next_flood.to(pred_h_t.device)

                                    # 使用水深 mask（只需要一個通道）
                                    phys_mask = None
                                    if mask is not None:
                                        phys_mask = mask[idx:idx+1, 0:1, :, :].to(dtype=pred_h_t.dtype, device=pred_h_t.device)

                                    # 取得降雨資料 (從 y 中提取當前時間步的降雨強度)
                                    if torch.is_tensor(y) and y.numel() > 0:
                                        if y.dim() > 1:
                                            rainfall_t = y[idx].float().mean()
                                        else:
                                            rainfall_t = y[idx].float()
                                    else:
                                        rainfall_t = torch.tensor(0.0, device=pred_h_t.device)

                                    # 計算質量守恆物理損失（只傳入 h_t+1，不需要 u_t+1, v_t+1）
                                    physics_loss_single, phys_summary_single = self.physics_loss(
                                        pred_h_t, pred_u_t, pred_v_t,  # 時間 t 的狀態
                                        pred_h_t1,                      # 時間 t+1 的水深
                                        rainfall_t,                     # 降雨強度
                                        mask=phys_mask
                                    )

                                    # 累加物理損失 (平均多個樣本)
                                    physics_loss_value = physics_loss_value + physics_loss_single / len(valid_indices)
                                    
                                    # 只保留最後一個樣本的 summary
                                    if idx == valid_indices[-1]:
                                        phys_summary = phys_summary_single
                                        
                                except Exception as e:
                                    # 靜默處理錯誤，繼續下一個樣本
                                    continue

                            total_loss = mse_loss + self.physics_weight * physics_loss_value
                        else:
                            # 只在第一次出現時輸出警告
                            if not hasattr(self, '_physics_warning_shown'):
                                print(f"[Physics] Skip: x0_hat channels={x0_hat.shape[1]} < 3 (need h,u,v)")
                                self._physics_warning_shown = True

                total_loss.backward()
                losses.append(total_loss.item())
            
            # Strong gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.5)
            # plot_grad_flow(net)
            optimizer.step()
            ema.update()
            if sched is not None: sched.step()

            # -------- logging --------
            if self.physics_loss is not None:
                # 簡化日誌：只顯示主要損失項
                log.info("train_it {}/{} | lr:{} | mse:{:.4f} | phys:{:.4f} | total:{:.4f}".format(
                    1+it,
                    opt.num_itr,
                    "{:.2e}".format(optimizer.param_groups[0]['lr']),
                    mse_loss.item(),
                    physics_loss_value.item() if isinstance(physics_loss_value, torch.Tensor) else physics_loss_value,
                    total_loss.item(),
                ))
            else:
                log.info("train_it {}/{} | lr:{} | loss:{}".format(
                    1+it,
                    opt.num_itr,
                    "{:.2e}".format(optimizer.param_groups[0]['lr']),
                    "{:+.4f}".format(total_loss.item()),
                ))
            if it % 10 == 0:
                self.writer.add_scalar(it, 'total_loss', total_loss.detach())
                self.writer.add_scalar(it, 'mse_loss', mse_loss.detach())
                if self.physics_loss is not None:
                    self.writer.add_scalar(it, 'mass_conservation_loss', physics_loss_value.detach())
                    if phys_summary:
                        self.writer.add_scalar(it, 'physics/mean_dhdt', phys_summary['mean_dhdt'])
                        self.writer.add_scalar(it, 'physics/mean_dhu_dx', phys_summary['mean_dhu_dx'])
                        self.writer.add_scalar(it, 'physics/mean_dhv_dy', phys_summary['mean_dhv_dy'])
                        self.writer.add_scalar(it, 'physics/mean_R', phys_summary['mean_R'])

            if it % 200 == 0 and it != 0:
                plot_losses()
            #     plot_model_gradients(net, rainfall_embber)

            if it % 100 == 0:  # 改為每 100 步保存一次
                if opt.global_rank == 0:
                    torch.save({
                        "net": self.net.state_dict(),
                        'embedding': self.rainfall_emb.state_dict(),
                        "ema": ema.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "sched": sched.state_dict() if sched is not None else sched,
                    }, opt.ckpt_path / "latest.pt")
                    log.info(f"Saved latest({it=}) checkpoint to {opt.ckpt_path=}!")
                if opt.distributed:
                    torch.distributed.barrier()

            # if it == 500 or it % 3000 == 0: # 0, 0.5k, 3k, 6k 9k
            #     net.eval()
            #     self.evaluation(opt, it, val_loader, corrupt_method)
            #     net.train()
        self.writer.close()

    @torch.no_grad()
    def ddpm_sampling(self, opt, x1, y, mask=None, cond=None, clip_denoise=False, nfe=None, log_count=10, verbose=True, eval=False, ode_method=None):

        # create discrete time steps that split [0, INTERVAL] into NFE sub-intervals.
        # e.g., if NFE=2 & INTERVAL=1000, then STEPS=[0, 500, 999] and 2 network
        # evaluations will be invoked, first from 999 to 500, then from 500 to 0.
        nfe = nfe or opt.interval-1
        assert 0 < nfe < opt.interval == len(self.diffusion.betas)
        steps = util.space_indices(opt.interval, nfe+1)

        # create log steps
        log_count = min(len(steps)-1, log_count)
        log_steps = [steps[i] for i in util.space_indices(len(steps)-1, log_count)]
        assert log_steps[0] == 0
        self.log.info(f"[DDPM Sampling] steps={opt.interval}, {nfe=}, {log_steps=}!")

        if opt.latent_space and eval:
            x1 = self.encode(x1, cond=False)
            cond = self.cond_stage_model(cond)
            
        x1 = x1.to(opt.device)
        if cond is not None: cond = cond.to(opt.device)
        mask = None
        if mask is not None:
            mask = mask.to(opt.device)
            x1 = (1. - mask) * x1 + mask * torch.randn_like(x1)

        with self.ema.average_parameters():
            self.net.eval()

            def pred_x0_fn(x1, xt, rainfall_emb, step, ode=None):
                if not ode:
                    step = torch.full((xt.shape[0],), step, device=opt.device, dtype=torch.long)
                    out = self.net(xt, step, rainfall_emb, cond=cond)
                    return self.compute_pred_x0(step, x1, xt, out, clip_denoise=clip_denoise)
                else:
                    step = torch.full((xt.shape[0],), step, device=opt.device, dtype=torch.float32)
                    out = self.net(xt, step, rainfall_emb, cond=cond)
                    return out

            rainfall_emb = self.rainfall_emb(y)
            xs, pred_x0 = self.diffusion.ddpm_sampling(
                steps, pred_x0_fn, x1, rainfall_emb, mask=mask, ot_ode=opt.ot_ode, log_steps=log_steps, verbose=verbose, ode_method=ode_method
            )

        b, *xdim = x1.shape
        # assert xs.shape == pred_x0.shape == (b, log_count, *xdim)

        if opt.latent_space and eval:
            xs = xs[:, 0, ...].to(opt.device)
            # xs = xs.squeeze(1)
            xs = self.decode(xs, cond=False)

        return xs, pred_x0

    @torch.no_grad()
    def evaluation(self, opt, it, val_loader, corrupt_method):

        log = self.log
        log.info(f"========== Evaluation started: iter={it} ==========")

        img_clean, img_corrupt, mask, y, cond, spm, _ = self.sample_batch(opt, val_loader, corrupt_method)

        x1 = img_corrupt.to(opt.device)
        xs, pred_x0s = self.ddpm_sampling(
            opt, x1, y, mask=mask, cond=cond, clip_denoise=opt.clip_denoise, verbose=opt.global_rank==0
        )

        log.info("Collecting tensors ...")
        img_clean   = all_cat_cpu(opt, log, img_clean)
        img_corrupt = all_cat_cpu(opt, log, img_corrupt)
        y           = all_cat_cpu(opt, log, y)
        xs          = all_cat_cpu(opt, log, xs)
        pred_x0s    = all_cat_cpu(opt, log, pred_x0s)

        batch, len_t, *xdim = xs.shape
        assert img_clean.shape == img_corrupt.shape == (batch, *xdim)
        assert xs.shape == pred_x0s.shape
        # assert y.shape == (batch,)
        log.info(f"Generated recon trajectories: size={xs.shape}")

        def log_image(tag, img, nrow=10):
            self.writer.add_image(it, tag, tu.make_grid((img+1)/2, nrow=nrow)) # [1,1] -> [0,1]

        # def log_accuracy(tag, img):
        #     pred = self.resnet(img.to(opt.device)) # input range [-1,1]
        #     accu = self.accuracy(pred, img_clean.to(opt.device))
        #     self.writer.add_scalar(it, tag, accu)

        log.info("Logging images ...")
        img_recon = xs[:, 0, ...]
        log_image("image/clean",   img_clean)
        log_image("image/corrupt", img_corrupt)
        log_image("image/recon",   img_recon)
        log_image("debug/pred_clean_traj", pred_x0s.reshape(-1, *xdim), nrow=len_t)
        log_image("debug/recon_traj",      xs.reshape(-1, *xdim),      nrow=len_t)

        # log.info("Logging accuracies ...")
        # log_accuracy("accuracy/clean",   img_clean)
        # log_accuracy("accuracy/corrupt", img_corrupt)
        # log_accuracy("accuracy/recon",   img_recon)

        log.info(f"========== Evaluation finished: iter={it} ==========")
        torch.cuda.empty_cache()
