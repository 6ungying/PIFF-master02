# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys
import random
import argparse

import copy
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.multiprocessing import Process

from logger import Logger
from distributed_util import init_processes
from corruption import build_corruption
from corruption.mixture import floodDataset, singleDEMFloodDataset
from dataset import imagenet
from i2sb import Runner, download_ckpt

import colored_traceback.always
from ipdb import set_trace as debug

RESULT_DIR = Path("results")

def set_seed(seed):
    # https://github.com/pytorch/pytorch/issues/7068
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.

def create_training_options():
    # --------------- basic ---------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",           type=int,   default=0)
    parser.add_argument("--name",           type=str,   default='flood-single-b128-sde-norm-novar-rand04',        help="experiment ID")
    parser.add_argument("--ckpt",           type=str,   default=None,        help="resumed checkpoint name")
    parser.add_argument("--gpu",            type=int,   default=None,        help="set only if you wish to run on a particular device; use -1 for CPU")
    parser.add_argument("--n-gpu-per-node", type=int,   default=1,           help="number of gpu on each node")
    parser.add_argument("--master-address", type=str,   default='localhost', help="address for master")
    parser.add_argument("--node-rank",      type=int,   default=0,           help="the index of node")
    parser.add_argument("--num-proc-node",  type=int,   default=1,           help="The number of nodes in multi node env")
    parser.add_argument("--eval",           action="store_true", default=False,         help="evaluation mode")
    # parser.add_argument("--amp",            action="store_true")

    # --------------- SB model ---------------
    parser.add_argument("--image-size",     type=int,   default=256)
    parser.add_argument("--dataset-dir", type=Path, default="C:\\Users\\THINKLAB\\Desktop\\PIFF-master02\\data\\dems\\", help="path to dataset")
    parser.add_argument("--latent-space", action="store_true", default=False, help="use latent space model")
    parser.add_argument("--normalize-latent", action="store_true", default=False, help="normalize latent space")
    parser.add_argument("--timestep-importance", type=str, default='continuous', help="use timestep importance")
    parser.add_argument("--corrupt",        type=str,   default='mixture',        help="restoration task")
    parser.add_argument("--t0",             type=float, default=1e-4,        help="sigma start time in network parametrization")
    parser.add_argument("--T",              type=float, default=1.,          help="sigma end time in network parametrization")
    parser.add_argument("--interval",       type=int,   default=1000,        help="number of interval")
    parser.add_argument("--beta-max",       type=float, default=0.1,         help="max diffusion for the diffusion model")
    # parser.add_argument("--beta-min",       type=float, default=0.1)
    parser.add_argument("--ot-ode",         action="store_true",  default=False,           help="use OT-ODE model")
    parser.add_argument("--clip-denoise",   action="store_true",             help="clamp predicted image to [-1,1] at each")

    # optional configs for conditional network
    parser.add_argument("--cond-x1",        action="store_true",  default=True,           help="conditional the network on degraded images")
    parser.add_argument("--spm",          action="store_true",  default=True,           help="use SPM for conditional network")
    parser.add_argument("--add-x1-noise",   action="store_true",             help="add noise to conditional network")

    # --------------- optimizer and loss ---------------
    parser.add_argument("--batch-size",     type=int,   default=128)
    parser.add_argument("--microbatch",     type=int,   default=2,           help="accumulate gradient over microbatch until full batch-size")
    parser.add_argument("--num-itr",        type=int,   default=1000000,     help="training iteration")
    parser.add_argument("--lr",             type=float, default=5e-5,        help="learning rate")
    parser.add_argument("--lr-gamma",       type=float, default=0.99,        help="learning rate decay ratio")
    parser.add_argument("--lr-step",        type=int,   default=1000,        help="learning rate decay step size")
    parser.add_argument("--l2-norm",        type=float, default=0.0)
    parser.add_argument("--ema",            type=float, default=0.999)

    # --------------- physics loss ---------------
    parser.add_argument("--use-physics",    action="store_true", default=False,  help="enable physics-informed loss")
    parser.add_argument("--physics-weight", type=float, default=1.0,         help="weight for physics loss")
    parser.add_argument("--dx",             type=float, default=20.0,        help="spatial resolution in x direction (meters)")
    parser.add_argument("--dy",             type=float, default=20.0,        help="spatial resolution in y direction (meters)")
    parser.add_argument("--dt",             type=float, default=3600.0,      help="temporal resolution (seconds, default=1 hour)")

    # --------------- path and logging ---------------
    # parser.add_argument("--dataset-dir",    type=Path,  default="/dataset",  help="path to LMDB dataset")
    parser.add_argument("--log-dir",        type=Path,  default=".log",      help="path to log std outputs and writer data")
    parser.add_argument("--log-writer",     type=str,   default='tensorbard',        help="log writer: can be tensorbard, wandb, or None")
    parser.add_argument("--wandb-api-key",  type=str,   default=None,        help="unique API key of your W&B account; see https://wandb.ai/authorize")
    parser.add_argument("--wandb-user",     type=str,   default=None,        help="user name of your W&B account")
    
    # --------------- test DEM configuration ---------------
    parser.add_argument("--test-dem-list",  type=str,   default=None,        help="Comma-separated list of test DEM numbers to exclude from training, e.g., '8,29,62'")

    opt = parser.parse_args()
    
    # ========= process test DEM list =========
    if opt.test_dem_list:
        opt.test_dem_list = [int(x.strip()) for x in opt.test_dem_list.split(',')]
        print(f"Test DEMs (excluded from training): {opt.test_dem_list}")
    else:
        opt.test_dem_list = None

    # ========= auto setup =========
    # 嘗試使用 GPU（即使有 sm_120 警告也試試看）
    if opt.gpu is None:
        opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        opt.device = f'cuda:{opt.gpu}' if torch.cuda.is_available() else 'cpu'
    
    print("=" * 60)
    print(f"Using device: {opt.device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print("Note: sm_120 warning can be ignored - will try to run anyway")
    print("=" * 60)
    
    if opt.name is None:
        opt.name = opt.corrupt
    opt.distributed = opt.n_gpu_per_node > 1
    opt.use_fp16 = False # disable fp16 for training

    # log ngc meta data
    if "NGC_JOB_ID" in os.environ.keys():
        opt.ngc_job_id = os.environ["NGC_JOB_ID"]

    # ========= path handle =========
    os.makedirs(opt.log_dir, exist_ok=True)
    opt.ckpt_path = RESULT_DIR / opt.name
    os.makedirs(opt.ckpt_path, exist_ok=True)

    if opt.ckpt is not None:
        ckpt_file = RESULT_DIR / opt.ckpt / "latest.pt"
        assert ckpt_file.exists()
        opt.load = ckpt_file
    else:
        opt.load = None

    # ========= auto assert =========
    assert opt.batch_size % opt.microbatch == 0, f"{opt.batch_size=} is not dividable by {opt.microbatch}!"
    return opt

def main(opt):
    log = Logger(opt.global_rank, opt.log_dir)
    log.info("=======================================================")
    log.info("         Image-to-Image Schrodinger Bridge")
    log.info("=======================================================")

    set_seed(opt.seed + opt.global_rank)

    train_dataset = floodDataset(opt, val=False)
    val_dataset   = floodDataset(opt, val=True)

    corrupt_method = build_corruption(opt, log)

    run = Runner(opt, log)

    # 只需要這次呼叫，不需要 DataLoader！
    run.train(opt, train_dataset, val_dataset, corrupt_method)

    log.info("Finish!")

if __name__ == '__main__':
    opt = create_training_options()

    assert opt.corrupt is not None

    # one-time download: ADM checkpoint
    # download_ckpt("data/")

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
            p = Process(target=init_processes, args=(global_rank, global_size, main, opt))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        # 嘗試使用 GPU（即使有 sm_120 警告）
        if torch.cuda.is_available():
            try:
                torch.cuda.set_device(opt.gpu if opt.gpu is not None else 0)
            except Exception as e:
                print(f"Warning: Could not set device - {e}")
        
        opt.global_rank = 0
        opt.local_rank = 0
        opt.global_size = 1
        init_processes(0, 1, main, opt)
