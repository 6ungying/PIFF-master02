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
        if sched is not None and "sched" in checkpoint.keys() and checkpoint["sched"] is not None:
            sched.load_state_dict(checkpoint["sched"])

    return optimizer, sched

def make_beta_schedule(n_timestep=1000, linear_start=1e-4, linear_end=2e-2):
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
            self.cond_stage_model = SpatialRescaler(**vars(self.model_config.CondStageParams))
            self.vqgan.to(opt.device)
            self.cond_stage_model.to(opt.device)

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
        if (hasattr(opt, 'spm') and opt.spm):
            ca4d = False
            spm = True
        elif (hasattr(opt, 'ca4d') and opt.ca4d):
            ca4d = True
            spm = False
        else:
            spm = False
            ca4d = False

        self.net = Image256Net(log, noise_levels=noise_levels, use_fp16=opt.use_fp16, cond=opt.cond_x1, ca4d=ca4d, spm=spm)
        self.rainfall_emb = RainfallEmbedder(256, 1)
        params = list(self.net.parameters()) + list(self.rainfall_emb.parameters())
        self.ema = ExponentialMovingAverage(params, decay=opt.ema)

        if opt.load:
            checkpoint = torch.load(opt.load, map_location="cpu")
            net = checkpoint['net']
            for key in net.keys():
                if 'spm' in key:
                    log.info(f"[Net] Detected SPM checkpoint key: {key}")
                elif 'ca4d' in key:
                    log.info(f"[Net] Detected CA4D checkpoint key: {key}")

            self.net.load_state_dict(checkpoint['net'])
            log.info(f"[Net] Loaded network ckpt: {opt.load}!")
            self.ema.load_state_dict(checkpoint["ema"])
            self.rainfall_emb.load_state_dict(checkpoint['embedding'])
            if opt.normalize_latent:
                self.net.ori_latent_mean = checkpoint["ori_latent_mean"]
                self.net.ori_latent_std = checkpoint["ori_latent_std"]
                self.net.cond_latent_mean = checkpoint["cond_latent_mean"]
                self.net.cond_latent_std = checkpoint["cond_latent_std"]

        self.net.to(opt.device)
        self.ema.to(opt.device)
        self.rainfall_emb.to(opt.device)

        if getattr(opt, 'use_physics', False):
            try:
                self.physics_loss = PhysicsInformedLoss(
                    dx=getattr(opt, 'dx', 20.0),
                    dy=getattr(opt, 'dy', 20.0),
                    dt=getattr(opt, 'dt', 3600.0),
                    h_mean=0.986,
                    h_std=0.0405,
                    u_mean=0.561,
                    u_std=0.078,
                    v_mean=0.495,
                    v_std=0.0789,
                    pixel_to_mps=0.066369,
                )
                self.physics_loss.to(opt.device)
                self.physics_weight = getattr(opt, 'physics_weight', 1.0)
                log.info(f"[Physics] [OK] 質量守恆損失已啟用 (weight={self.physics_weight})")
            except Exception as e:
                log.error(f"[Physics] 初始化失敗: {e}")
                self.physics_loss = None
        else:
            self.physics_loss = None
            log.info("[Physics] 物理損失未啟用")

        self.log = log

        if opt.eval:
            self.net.ori_latent_mean = self.net.ori_latent_mean.to(opt.device)
            self.net.ori_latent_std = self.net.ori_latent_std.to(opt.device)
            self.net.cond_latent_mean = self.net.cond_latent_mean.to(opt.device)
            self.net.cond_latent_std = self.net.cond_latent_std.to(opt.device)

    def logger(self, msg, **kwargs):
        print(msg, **kwargs)

    def get_latent_mean_std(self): pass
    def encode(self, x, cond=True): pass
    def decode(self, x_latent, cond=True): pass

    def compute_label(self, step, x0, xt, x1):
        std_fwd = self.diffusion.get_std_fwd(step, xdim=x0.shape[1:])
        label = (xt - x0) / std_fwd
        return label.detach()

    def compute_pred_x0(self, step, xt, x1, net_out, clip_denoise=False):
        std_fwd = self.diffusion.get_std_fwd(step, xdim=xt.shape[1:])
        pred_x0 = xt - std_fwd * net_out
        if clip_denoise: pred_x0.clamp_(-1., 1.)
        return pred_x0

    def sample_batch(self, opt, loader, corrupt_method):
        (
            flood_img, vx_img, vy_img, dem_img,
            mask_flood, mask_vx, mask_vy,
            y, img_name, vx_img_name, vy_img_name,
            spm, ca4d, next_timestep_data, max_depth, dem_id
        ) = next(loader)

        def ensure_single_channel(t):
            if t.dim() == 4 and t.shape[1] > 1:
                return t[:, 0:1, :, :]
            return t

        flood_img = ensure_single_channel(flood_img)
        vx_img    = ensure_single_channel(vx_img)
        vy_img    = ensure_single_channel(vy_img)
        dem_img   = ensure_single_channel(dem_img)

        # [N, 3, H, W]
        clean_img   = torch.cat([flood_img, vx_img, vy_img], dim=1)
        corrupt_img = torch.cat([dem_img, dem_img, dem_img], dim=1)

        # [MODIFIED] 強制設定 Mask 為 None (完全不使用 Mask)
        # 即使 dataset 回傳了 None，我們也明確指定 mask 變數為 None，
        # 避免後續程式碼誤用 mask 進行運算
        mask = None

        # NaN 處理
        if torch.isnan(clean_img).any():
            flood_nan = torch.isnan(clean_img[:, 0:1])
            clean_img[:, 0:1] = torch.where(flood_nan, torch.tensor(0.01, device=clean_img.device), clean_img[:, 0:1])
            vel_nan = torch.isnan(clean_img[:, 1:])
            clean_img[:, 1:] = torch.where(vel_nan, torch.tensor(0.0, device=clean_img.device), clean_img[:, 1:])

        if torch.isnan(corrupt_img).any():
             corrupt_img = torch.nan_to_num(corrupt_img, nan=0.0)

        device = opt.device
        y      = y.to(device)
        x0     = clean_img.to(device)
        x1     = corrupt_img.to(device)
        ca4d   = ca4d.to(device) if ca4d is not None else None
        
        # mask 已經是 None，不需要 .to(device)

        cond = x1 if opt.cond_x1 else None

        if getattr(opt, "add_x1_noise", False):
            x1 = x1 + torch.randn_like(x1)

        return x0, x1, mask, y, cond, spm, ca4d, next_timestep_data, max_depth, dem_id

    def train(self, opt, train_dataset, val_dataset, corrupt_method):
        gradient_list = []
        embedder_gradient_list = []
        losses = []
        
        self.writer = util.build_log_writer(opt)
        log = self.log
        net = self.net
        ema = self.ema
        rainfall_embber = self.rainfall_emb
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
                x0, x1, mask, y, cond, spm, ca4d, next_timestep_data, max_depth, dem_id = self.sample_batch(opt, train_loader, corrupt_method)

                # ... (timestep sampling) ...
                if opt.timestep_importance == 'continuous':
                    t1, t0 = 1, 0
                    step = torch.rand((x0.shape[0],)) * (t1 - t0)
                    step = step.view(-1, 1, 1, 1).to(x0.device)
                    if opt.ot_ode:
                        xt = (1-step) * x0 + step * x1
                        label = x1 - x0
                    if not opt.ot_ode:
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

                rainfall_emb = rainfall_embber(y)
                pred = net(xt, step, rainfall_emb, cond=cond, spm=spm, ca4d=ca4d)
                
                if torch.isnan(xt).any() or torch.isnan(label).any() or torch.isnan(pred).any():
                    continue

                # [MODIFIED] MSE Loss: 直接計算 (等同於 mask=None)，不套用任何遮罩
                mse_loss = F.mse_loss(pred, label)
                
                if torch.isnan(mse_loss) or torch.isinf(mse_loss):
                    continue

                total_loss = mse_loss
                physics_loss_value = torch.tensor(0.0, device=opt.device)
                phys_summary = {}

                if self.physics_loss is not None and pred is not None and not getattr(opt, 'latent_space', False) and next_timestep_data is not None:
                    valid_indices = [i for i, data in enumerate(next_timestep_data) if data is not None]
                    
                    if len(valid_indices) > 0:
                        if opt.timestep_importance == 'continuous':
                            x0_hat = x1 - pred
                        else:
                            x0_hat = self.compute_pred_x0(step, xt, x1, pred, clip_denoise=False)

                        if x0_hat.shape[1] >= 3:
                            for idx in valid_indices:
                                try:
                                    pred_h_t = x0_hat[idx:idx+1, 0:1, :, :]
                                    pred_u_t = x0_hat[idx:idx+1, 1:2, :, :]
                                    pred_v_t = x0_hat[idx:idx+1, 2:3, :, :]

                                    next_data = next_timestep_data[idx]
                                    next_flood, next_vx, next_vy, _, _, _ = next_data 
                                    
                                    if next_flood.dim() == 3:
                                        next_flood = next_flood.unsqueeze(0)
                                    pred_h_t1 = next_flood.to(pred_h_t.device)

                                    if torch.is_tensor(y) and y.numel() > 0:
                                        if y.dim() > 1:
                                            rainfall_t = y[idx].float().mean()
                                        else:
                                            rainfall_t = y[idx].float()
                                    else:
                                        rainfall_t = torch.tensor(0.0, device=pred_h_t.device)

                                    sample_max_depth = None
                                    if max_depth is not None:
                                        if torch.is_tensor(max_depth):
                                            sample_max_depth = max_depth[idx:idx+1]
                                        elif isinstance(max_depth, (list, tuple)):
                                            sample_max_depth = torch.tensor([max_depth[idx]], device=pred_h_t.device, dtype=pred_h_t.dtype)
                                        else:
                                            sample_max_depth = torch.tensor([max_depth], device=pred_h_t.device, dtype=pred_h_t.dtype)

                                    # [MODIFIED] Physics Loss: 明確傳入 mask=None
                                    physics_loss_single, phys_summary_single = self.physics_loss(
                                        pred_h_t, pred_u_t, pred_v_t,
                                        pred_h_t1,
                                        rainfall_t,
                                        mask=None,  # <-- 強制 mask=None (全圖計算)
                                        max_depth=sample_max_depth
                                    )

                                    physics_loss_value = physics_loss_value + physics_loss_single / len(valid_indices)
                                    if idx == valid_indices[-1]:
                                        phys_summary = phys_summary_single
                                        
                                except Exception as e:
                                    continue

                            total_loss = mse_loss + self.physics_weight * physics_loss_value

                total_loss.backward()
                losses.append(total_loss.item())
            
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.5)
            optimizer.step()
            ema.update()
            if sched is not None: sched.step()

            if self.physics_loss is not None:
                log.info("train_it {}/{} | lr:{} | mse:{:.4f} | phys:{:.4f} | total:{:.4f}".format(
                    1+it, opt.num_itr, "{:.2e}".format(optimizer.param_groups[0]['lr']),
                    mse_loss.item(), physics_loss_value.item() if isinstance(physics_loss_value, torch.Tensor) else physics_loss_value, total_loss.item(),
                ))
            else:
                log.info("train_it {}/{} | lr:{} | loss:{}".format(
                    1+it, opt.num_itr, "{:.2e}".format(optimizer.param_groups[0]['lr']), "{:+.4f}".format(total_loss.item()),
                ))
            
            if it % 10 == 0:
                self.writer.add_scalar(it, 'total_loss', total_loss.detach())
                self.writer.add_scalar(it, 'mse_loss', mse_loss.detach())
                if self.physics_loss is not None:
                    self.writer.add_scalar(it, 'mass_conservation_loss', physics_loss_value.detach())

            if it % 100 == 0:
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

        self.writer.close()

    @torch.no_grad()
    def ddpm_sampling(self, opt, x1, y, mask=None, cond=None, clip_denoise=False, nfe=None, log_count=10, verbose=True, eval=False, ode_method=None, ca4d=None, spm=None):
        nfe = nfe or opt.interval-1
        steps = util.space_indices(opt.interval, nfe+1)
        log_count = min(len(steps)-1, log_count)
        log_steps = [steps[i] for i in util.space_indices(len(steps)-1, log_count)]
        self.log.info(f"[DDPM Sampling] steps={opt.interval}, {nfe=}, {log_steps=}!")

        x1 = x1.to(opt.device)
        if cond is not None: cond = cond.to(opt.device)
        if ca4d is not None: ca4d = ca4d.to(opt.device)
        if spm is not None: spm = spm.to(opt.device)
        
        # mask = None # (傳入的 mask 已經是 None，所以不需要特別設定)

        with self.ema.average_parameters():
            self.net.eval()

            def pred_x0_fn(x1, xt, rainfall_emb, step, ode=None):
                if not ode:
                    step = torch.full((xt.shape[0],), step, device=opt.device, dtype=torch.long)
                    out = self.net(xt, step, rainfall_emb, cond=cond, spm=spm, ca4d=ca4d)
                    return self.compute_pred_x0(step, x1, xt, out, clip_denoise=clip_denoise)
                else:
                    step = torch.full((xt.shape[0],), step, device=opt.device, dtype=torch.float32)
                    out = self.net(xt, step, rainfall_emb, cond=cond, spm=spm, ca4d=ca4d)
                    return out

            rainfall_emb = self.rainfall_emb(y)
            xs, pred_x0 = self.diffusion.ddpm_sampling(
                steps, pred_x0_fn, x1, rainfall_emb, mask=mask, ot_ode=opt.ot_ode, log_steps=log_steps, verbose=verbose, ode_method=ode_method
            )

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
