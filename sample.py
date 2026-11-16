# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# -------------------------------------------------------

import os
import copy
import argparse
import random
from pathlib import Path
from easydict import EasyDict as edict

import numpy as np

import torch
import torch.distributed as dist
from torch.multiprocessing import Process
from torch.utils.data import DataLoader, Subset
from torch_ema import ExponentialMovingAverage
import torchvision.utils as tu

from logger import Logger
import distributed_util as dist_util
from i2sb import Runner, download_ckpt
from corruption import build_corruption
from dataset import imagenet
from i2sb import ckpt_util

import colored_traceback.always
from ipdb import set_trace as debug
from corruption.mixture import floodDataset, singleDEMFloodDataset, yilanDataset, hsinchuDataset

RESULT_DIR = Path("results")

def set_seed(seed):
    # https://github.com/pytorch/pytorch/issues/7068
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.

def build_subset_per_gpu(opt, dataset, log):
    n_data = len(dataset)
    n_gpu  = opt.global_size
    n_dump = (n_data % n_gpu > 0) * (n_gpu - n_data % n_gpu)

    # create index for each gpu
    total_idx = np.concatenate([np.arange(n_data), np.zeros(n_dump)]).astype(int)
    idx_per_gpu = total_idx.reshape(-1, n_gpu)[:, opt.global_rank]
    log.info(f"[Dataset] Add {n_dump} data to the end to be devided by {n_gpu=}. Total length={len(total_idx)}!")

    # build subset
    indices = idx_per_gpu.tolist()
    subset = Subset(dataset, indices)
    log.info(f"[Dataset] Built subset for gpu={opt.global_rank}! Now size={len(subset)}!")
    return subset

def collect_all_subset(sample, log):
    batch, *xdim = sample.shape
    gathered_samples = dist_util.all_gather(sample, log)
    gathered_samples = [sample.cpu() for sample in gathered_samples]
    # [batch, n_gpu, *xdim] --> [batch*n_gpu, *xdim]
    return torch.stack(gathered_samples, dim=1).reshape(-1, *xdim)

def build_partition(opt, full_dataset, log):
    n_samples = len(full_dataset)

    part_idx, n_part = [int(s) for s in opt.partition.split("_")]
    assert part_idx < n_part and part_idx >= 0
    assert n_samples % n_part == 0

    n_samples_per_part = n_samples // n_part
    start_idx = part_idx * n_samples_per_part
    end_idx = (part_idx+1) * n_samples_per_part

    indices = [i for i in range(start_idx, end_idx)]
    subset = Subset(full_dataset, indices)
    log.info(f"[Dataset] Built partition={opt.partition}, {start_idx=}, {end_idx=}! Now size={len(subset)}!")
    return subset

def build_val_dataset(opt, log, corrupt_type):
    if "sr4x" in corrupt_type:
        val_dataset = imagenet.build_lmdb_dataset(opt, log, train=False) # full 50k val
    elif "inpaint" in corrupt_type:
        mask = corrupt_type.split("-")[1]
        val_dataset = imagenet.InpaintingVal10kSubset(opt, log, mask) # subset 10k val + mask
    elif corrupt_type == "mixture":
        from corruption.mixture import MixtureCorruptDatasetVal
        val_dataset = imagenet.build_lmdb_dataset_val10k(opt, log)
        val_dataset = MixtureCorruptDatasetVal(opt, val_dataset) # subset 10k val + mixture
    else:
        val_dataset = imagenet.build_lmdb_dataset_val10k(opt, log) # subset 10k val

    # build partition
    if opt.partition is not None:
        val_dataset = build_partition(opt, val_dataset, log)
    return val_dataset

def get_recon_imgs_fn(opt, nfe):
    test_name = opt.sampling_method + "-dcvar"
    sample_dir = RESULT_DIR / opt.ckpt / "test3_nfe{}{}_{}".format(
        nfe, "_clip" if opt.clip_denoise else "", test_name
    )
    os.makedirs(sample_dir, exist_ok=True)

    recon_imgs_fn = sample_dir / "recon{}.pt".format(
        "" if opt.partition is None else f"_{opt.partition}"
    )
    return recon_imgs_fn

def compute_batch(ckpt_opt, corrupt_type, corrupt_method, out):
    """Prepare depth/DEM batches; velocity tuple slots are intentionally unused."""
    if len(out) == 16:
        depth, _, _, dem, _, _, _, rainfall, paths, _, _, spm, ca4d, next_data, max_depth, dem_id = out
    elif len(out) == 15:
        depth, _, _, dem, _, _, _, rainfall, paths, _, _, ca4d, next_data, max_depth, dem_id = out
        spm = None
    else:
        raise ValueError(f"Unsupported flood dataset output length: {len(out)}")
    device = ckpt_opt.device
    clean = torch.nan_to_num(depth[:, :1].to(device), nan=0.01)
    x1 = torch.nan_to_num(dem[:, :1].to(device), nan=0.0)
    y = torch.as_tensor(rainfall, dtype=torch.long, device=device)
    if y.dim() == 1:
        y = y.unsqueeze(0)
    y = y.clamp(0, 99)
    spm = spm.to(device) if getattr(ckpt_opt, 'spm', False) and spm is not None else None
    ca4d = ca4d.to(device) if getattr(ckpt_opt, 'ca4d', False) and ca4d is not None else None
    cond = x1.detach() if ckpt_opt.cond_x1 else None
    if ckpt_opt.add_x1_noise:
        x1 = x1 + torch.randn_like(x1)
    return clean, x1, None, cond, y, paths, spm, ca4d, (next_data, max_depth, dem_id)


def depth_to_image(depth, image_path, single_dem=False):
    """Undo the depth normalization used by each dataset."""
    name = str(image_path).lower()
    if 'hsinchu' in name:
        mean, std = 0.9751028, 0.0256081
    elif 'yilan' in name:
        mean, std = 0.9880, 0.0279
    else:
        mean, std = (0.987, 0.0343) if single_dem else (0.986, 0.0405)
    return (depth * std + mean).clamp(0, 1)


def save_depth_image(depth, image_path, output_dir, single_dem=False):
    from PIL import Image
    depth = depth_to_image(depth, image_path, single_dem)
    name = Path(str(image_path).replace('\\', '/')).stem
    pixels = (depth[0].detach().cpu().numpy() * 255).round().astype(np.uint8)
    Image.fromarray(pixels).save(Path(output_dir) / f'recon_{name}.png')


@torch.no_grad()
def main(opt):
    log = Logger(opt.global_rank, ".log")

    ckpt_arg = Path(opt.ckpt)
    if ckpt_arg.is_absolute() and ckpt_arg.exists():
        ckpt_dir = ckpt_arg
    elif (RESULT_DIR / opt.ckpt).exists():
        ckpt_dir = RESULT_DIR / opt.ckpt
    elif ckpt_arg.exists():
        ckpt_dir = ckpt_arg
    else:
        log.info(f"Checkpoint folder not found: tried '{ckpt_arg}' and 'results/{opt.ckpt}'")
        raise FileNotFoundError(f"Checkpoint folder not found: '{opt.ckpt}'.")

    ckpt_opt = ckpt_util.build_ckpt_option(opt, log, ckpt_dir)
    corrupt_type = ckpt_opt.corrupt
    nfe = opt.nfe or ckpt_opt.interval-1

    # ===== CRITICAL: Sync physical-model flags from checkpoint to sampling opt =====
    # Dataset must know whether to load SPM / CA4D
    opt.spm = getattr(ckpt_opt, "spm", False)
    opt.ca4d = getattr(ckpt_opt, "ca4d", False)
    log.info(f"[Config Sync] Physical guidance from checkpoint: spm={opt.spm}, ca4d={opt.ca4d}")

    # 如果命令列有指定 test_dem_list，使用它；否則從 ckpt_opt 繼承
    if opt.test_dem_list:
        opt.test_dem_list = [int(x.strip()) for x in opt.test_dem_list.split(',')]
        log.info(f"Using test DEMs from command line: {opt.test_dem_list}")
    elif hasattr(ckpt_opt, 'test_dem_list') and ckpt_opt.test_dem_list:
        opt.test_dem_list = ckpt_opt.test_dem_list
        log.info(f"Using test DEMs from training config: {opt.test_dem_list}")
    else:
        opt.test_dem_list = None
        log.info("No test_dem_list specified, using single-DEM mode")

    # 如果命令列有指定 test_rain_list，使用它；否則從 ckpt_opt 繼承
    if opt.test_rain_list:
        opt.test_rain_list = [int(x.strip()) for x in opt.test_rain_list.split(',')]
        log.info(f"Using test rainfall scenarios from command line: {opt.test_rain_list}")
    elif hasattr(ckpt_opt, 'test_rain_list') and ckpt_opt.test_rain_list:
        opt.test_rain_list = ckpt_opt.test_rain_list
        log.info(f"Using test rainfall scenarios from training config: {opt.test_rain_list}")
    else:
        opt.test_rain_list = None
        log.info("No test_rain_list specified")

    corrupt_method = build_corruption(opt, log, corrupt_type=corrupt_type)

    # [MODIFIED] 根據參數選擇數據集
    if hasattr(opt, 'use_hsinchu') and opt.use_hsinchu:
        log.info("Using hsinchuDataset for Hsinchu testing (--use-hsinchu specified)")
        val_dataset = hsinchuDataset(opt)
    elif hasattr(opt, 'use_yilan') and opt.use_yilan:
        log.info("Using yilanDataset for Yilan multi-terrain testing (--use-yilan specified)")
        val_dataset = yilanDataset(opt)
    elif opt.use_single_dem:
        log.info("Using singleDEMFloodDataset for single-DEM testing (--use-single-dem specified)")
        val_dataset = singleDEMFloodDataset(opt, test=True)
    elif opt.test_dem_list:
        # opt.test_dem_list = [int(x.strip()) for x in opt.test_dem_list.split(',')]
        log.info(f"Using floodDataset for multi-DEM testing (test_dem_list: {opt.test_dem_list})")
        val_dataset = floodDataset(opt, test=True)
    else:
        log.info("Using singleDEMFloodDataset for single-DEM testing (default)")
        val_dataset = singleDEMFloodDataset(opt, test=True)
    
    from i2sb.util import custom_collate_fn
    subset_dataset = build_subset_per_gpu(opt, val_dataset, log)
    val_loader = DataLoader(subset_dataset,
        batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=0, drop_last=False,
        collate_fn=custom_collate_fn, 
    )

    runner = Runner(ckpt_opt, log, save_opt=False)

    if opt.use_fp16:
        runner.ema.copy_to() 
        runner.net.diffusion_model.convert_to_fp16()
        runner.ema = ExponentialMovingAverage(runner.net.parameters(), decay=0.99) 

    recon_imgs_fn = get_recon_imgs_fn(opt, nfe)
    log.info(f"Recon images will be saved to {recon_imgs_fn}!")

    for loader_itr, out in enumerate(val_loader):
        print(f"[DEBUG] Processing batch {loader_itr + 1}/{len(val_loader)}")

        # [MODIFIED] 解包包含 SPM 和 CA4D 的結果
        corrupt_img, x1, mask, cond, y, image_name, spm, ca4d, _ = compute_batch(ckpt_opt, corrupt_type, corrupt_method, out)
        
        # ===== CRITICAL SAFEGUARD: Check that physical guidance is actually loaded =====
        if getattr(ckpt_opt, "spm", False) and spm is None:
            raise RuntimeError(
                "[SPM ERROR] Checkpoint was trained with SPM (--spm), but spm_image is None! "
                "Check: (1) opt.spm={}, (2) SPM data path exists, (3) SPM filenames match pattern.".format(opt.spm)
            )
        
        if getattr(ckpt_opt, "ca4d", False) and ca4d is None:
            raise RuntimeError(
                "[CA4D ERROR] Checkpoint was trained with CA4D (--ca4d), but ca4d_image is None! "
                "Check: (1) opt.ca4d={}, (2) CA4D data path exists, (3) CA4D filenames match pattern.".format(opt.ca4d)
            )
        
        # ===== Debug print on first batch =====
        if loader_itr == 0:
            print(f"[DEBUG] x1 shape={x1.shape}, range=[{x1.min().item():.3f}, {x1.max().item():.3f}]")
            print(f"[DEBUG] y shape={y.shape}, range=[{y.min().item():.3f}, {y.max().item():.3f}]")
            if spm is not None:
                print(f"[DEBUG] spm shape={spm.shape}, range=[{spm.min().item():.3f}, {spm.max().item():.3f}]")
            else:
                print(f"[DEBUG] spm: None")
            if ca4d is not None:
                print(f"[DEBUG] ca4d shape={ca4d.shape}, range=[{ca4d.min().item():.3f}, {ca4d.max().item():.3f}]")
            else:
                print(f"[DEBUG] ca4d: None")
        
        # [MODIFIED] 只傳入 CA4D 參數 (不傳 SPM,因為 ddpm_sampling 不支援)
        xs, _ = runner.ddpm_sampling(
            ckpt_opt, x1, y, mask=mask, cond=cond, clip_denoise=opt.clip_denoise, nfe=nfe, 
            verbose=opt.n_gpu_per_node==1, eval=True, ode_method=opt.sampling_method,
            ca4d=ca4d, spm=spm
        )
        recon_img = xs[:, 0, ...].to(opt.device) 

        assert recon_img.shape == corrupt_img.shape

        for i in range(len(recon_img)):
            rec = recon_img[i]

            save_depth_image(rec, image_name[i], recon_imgs_fn.parent,
                             single_dem=isinstance(val_dataset, singleDEMFloodDataset))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",           type=int,  default=0)
    parser.add_argument("--n-gpu-per-node", type=int,  default=1,           help="number of gpu on each node")
    parser.add_argument("--master-address", type=str,  default='localhost', help="address for master")
    parser.add_argument("--node-rank",      type=int,  default=0,           help="the index of node")
    parser.add_argument("--num-proc-node",  type=int,  default=1,           help="The number of nodes in multi node env")
    parser.add_argument("--latent-space", action="store_true", default=False, help="use latent space model")
    parser.add_argument("--eval",        action="store_true", default=True, help="")
    # data
    parser.add_argument("--image-size",     type=int,  default=256)
    parser.add_argument("--dataset-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data", help="data root containing tainan_dem.png, train_rf.csv, train/, dems/, etc.")
    parser.add_argument("--partition",      type=str,  default=None,        help="e.g., '0_4' means the first 25%% of the dataset")
    parser.add_argument("--use-single-dem", action="store_true", default=False, help="use single DEM flood dataset for testing (instead of multi-DEM)")
    parser.add_argument("--use-yilan",      action="store_true", default=False, help="use Yilan multi-terrain flood dataset for testing")
    parser.add_argument("--use-hsinchu",    action="store_true", default=False, help="use Hsinchu flood dataset for testing")
    parser.add_argument("--test-dem-list",  type=str,  default=None,        help="Comma-separated list of test DEM numbers, e.g., '61,62,65'")
    parser.add_argument("--test-rain-list", type=str,  default=None,        help="Comma-separated list of test rainfall scenarios to use for testing, e.g., '9,20,33'")
    
    # [ADDED] Physical guidance flags
    parser.add_argument("--spm",            action="store_true", default=False, help="use SPM guidance during sampling (will be synced from checkpoint)")
    parser.add_argument("--ca4d",           action="store_true", default=False, help="use CA4D guidance during sampling (will be synced from checkpoint)")

    # sample
    parser.add_argument("--batch-size",     type=int,  default=30)
    parser.add_argument("--sampling-method", type=str, default='euler-maruyama', help="sampling method")
    parser.add_argument("--ckpt",           type=str,  default='flood-depth-only',        help="depth-only checkpoint folder or name under results/")
    parser.add_argument("--nfe",            type=int,  default=10,        help="sampling steps")
    parser.add_argument("--clip-denoise",   action="store_true",            help="clamp predicted image to [-1,1] at each")
    parser.add_argument("--use-fp16",       action="store_true",            help="use fp16 network weight for faster sampling")

    arg = parser.parse_args()

    opt = edict(
        distributed=(arg.n_gpu_per_node > 1),
        device="cuda",
    )
    opt.update(vars(arg))

    set_seed(opt.seed)

    if opt.distributed:
        size = opt.n_gpu_per_node

        processes = []
        for rank in range(size):
            opt = copy.deepcopy(opt)
            opt.local_rank = rank
            global_rank = rank + opt.node_rank * opt.n_gpu_per_node
            global_size = opt.num_proc_node * opt.n_gpu_per_node
            opt.global_rank = global_rank
            opt.global_size = global_size
            print('Node rank %d, local proc %d, global proc %d, global_size %d' % (opt.node_rank, rank, global_rank, global_size))
            p = Process(target=dist_util.init_processes, args=(global_rank, global_size, main, opt))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        torch.cuda.set_device(0)
        opt.global_rank = 0
        opt.local_rank = 0
        opt.global_size = 1
        dist_util.init_processes(0, opt.n_gpu_per_node, main, opt)
