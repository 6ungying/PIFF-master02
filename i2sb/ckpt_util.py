# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import requests
from tqdm import tqdm

import pickle

import torch

from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model,
    args_to_dict,
)

from argparse import Namespace

from pathlib import Path
from easydict import EasyDict as edict

from ipdb import set_trace as debug

ADM_IMG256_UNCOND_CKPT = "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt"
I2SB_IMG256_UNCOND_PKL = "256x256_diffusion_uncond_fixedsigma.pkl"
I2SB_IMG256_UNCOND_CKPT = "256x256_diffusion_uncond_fixedsigma.pt"
I2SB_IMG256_COND_PKL = "256x256_diffusion_cond_fixedsigma.pkl"
I2SB_IMG256_COND_CKPT = "256x256_diffusion_cond_fixedsigma.pt"

def download(url, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)

def create_argparser():
    return Namespace(
        attention_resolutions='32,16,8',
        batch_size=4,
        channel_mult='',
        class_cond=False,
        clip_denoised=True,
        diffusion_steps=1000,
        dropout=0.0,
        image_size=256,
        learn_sigma=True,
        adm_ckpt='256x256_diffusion_uncond.pt',
        noise_schedule='linear',
        num_channels=256,
        num_head_channels=64,
        num_heads=4,
        num_heads_upsample=-1,
        num_res_blocks=2,
        num_samples=4,
        predict_xstart=False,
        resblock_updown=True,
        rescale_learned_sigmas=False,
        rescale_timesteps=False,
        timestep_respacing='250',
        use_checkpoint=False,
        use_ddim=False,
        use_fp16=True,
        use_kl=False,
        use_new_attention_order=False,
        use_scale_shift_norm=True
    )

def extract_model_kwargs(kwargs):
    return {
        "image_size": kwargs["image_size"],
        "num_channels": kwargs["num_channels"],
        "num_res_blocks": kwargs["num_res_blocks"],
        "channel_mult": kwargs["channel_mult"],
        "learn_sigma": kwargs["learn_sigma"],
        "class_cond": kwargs["class_cond"],
        "use_checkpoint": kwargs["use_checkpoint"],
        "attention_resolutions": kwargs["attention_resolutions"],
        "num_heads": kwargs["num_heads"],
        "num_head_channels": kwargs["num_head_channels"],
        "num_heads_upsample": kwargs["num_heads_upsample"],
        "use_scale_shift_norm": kwargs["use_scale_shift_norm"],
        "dropout": kwargs["dropout"],
        "resblock_updown": kwargs["resblock_updown"],
        "use_fp16": kwargs["use_fp16"],
        "use_new_attention_order": kwargs["use_new_attention_order"],
    }

def extract_diffusion_kwargs(kwargs):
    return {
        "diffusion_steps": kwargs["diffusion_steps"],
        "learn_sigma": False,
        "noise_schedule": kwargs["noise_schedule"],
        "use_kl": kwargs["use_kl"],
        "predict_xstart": kwargs["predict_xstart"],
        "rescale_timesteps": kwargs["rescale_timesteps"],
        "rescale_learned_sigmas": kwargs["rescale_learned_sigmas"],
        "timestep_respacing": kwargs["timestep_respacing"],
    }

def download_adm_image256_uncond_ckpt(ckpt_dir="data/"):
    ckpt_pkl = os.path.join(ckpt_dir, I2SB_IMG256_UNCOND_PKL)
    ckpt_pt  = os.path.join(ckpt_dir, I2SB_IMG256_UNCOND_CKPT)
    if os.path.exists(ckpt_pkl) and os.path.exists(ckpt_pt):
        return

    opt = create_argparser()

    adm_ckpt = os.path.join(ckpt_dir, opt.adm_ckpt)
    if not os.path.exists(adm_ckpt):
        print("Downloading ADM checkpoint to {} ...".format(adm_ckpt))
        download(ADM_IMG256_UNCOND_CKPT, adm_ckpt)
    ckpt_state_dict = torch.load(adm_ckpt, map_location="cpu")

    # pt: remove the sigma prediction
    ckpt_state_dict["out.2.weight"] = ckpt_state_dict["out.2.weight"][:3]
    ckpt_state_dict["out.2.bias"] = ckpt_state_dict["out.2.bias"][:3]
    torch.save(ckpt_state_dict, ckpt_pt)

    # pkl
    kwargs = args_to_dict(opt, model_and_diffusion_defaults().keys())
    kwargs['learn_sigma'] = False
    model_kwargs = extract_model_kwargs(kwargs)
    with open(ckpt_pkl, "wb") as f:
        pickle.dump(model_kwargs, f)

    print(f"Saved adm uncond pretrain models at {ckpt_pkl=} and {ckpt_pt}!")

def download_adm_image256_cond_ckpt(ckpt_dir="data/"):
    ckpt_pkl = os.path.join(ckpt_dir, I2SB_IMG256_COND_PKL)
    ckpt_pt  = os.path.join(ckpt_dir, I2SB_IMG256_COND_CKPT)
    if os.path.exists(ckpt_pkl) and os.path.exists(ckpt_pt):
        return

    opt = create_argparser()

    adm_ckpt = os.path.join(ckpt_dir, opt.adm_ckpt)
    if not os.path.exists(adm_ckpt):
        print("Downloading ADM checkpoint to {} ...".format(adm_ckpt))
        download(ADM_IMG256_UNCOND_CKPT, adm_ckpt)
    ckpt_state_dict = torch.load(adm_ckpt, map_location="cpu")

    # pkl
    kwargs = args_to_dict(opt, model_and_diffusion_defaults().keys())
    kwargs['learn_sigma'] = False
    model_kwargs = extract_model_kwargs(kwargs)
    model_kwargs.update(extract_diffusion_kwargs(kwargs))
    model_kwargs["use_fp16"] = False
    model_kwargs["in_channels"] = 6
    with open(ckpt_pkl, "wb") as f:
        pickle.dump(model_kwargs, f)

    # pt: remove the sigma prediction and add concat module
    ckpt_state_dict["out.2.weight"] = ckpt_state_dict["out.2.weight"][:3]
    ckpt_state_dict["out.2.bias"] = ckpt_state_dict["out.2.bias"][:3]
    model = create_model(**model_kwargs)
    ckpt_state_dict['input_blocks.0.0.weight'] = torch.cat([
        ckpt_state_dict['input_blocks.0.0.weight'],
        model.input_blocks[0][0].weight.data[:, 3:]
    ], dim=1)
    model.load_state_dict(ckpt_state_dict)
    torch.save(ckpt_state_dict, ckpt_pt)

    print(f"Saved adm cond pretrain models at {ckpt_pkl=} and {ckpt_pt}!")

def download_ckpt(ckpt_dir="data/"):
    os.makedirs(ckpt_dir, exist_ok=True)
    download_adm_image256_uncond_ckpt(ckpt_dir=ckpt_dir)
    # download_adm_image256_cond_ckpt(ckpt_dir=ckpt_dir)

def build_ckpt_option(opt, log, ckpt_path):
    ckpt_path = Path(ckpt_path)
    opt_pkl_path = ckpt_path / "options.pkl"
    assert opt_pkl_path.exists()
    
    # 嘗試載入 pickle,如果失敗則使用當前 opt
    try:
        with open(opt_pkl_path, "rb") as f:
            ckpt_opt = pickle.load(f)
        log.info(f"Loaded options from {opt_pkl_path=}!")
    except (ModuleNotFoundError, AttributeError, ImportError) as e:
        log.warning(f"Failed to load checkpoint options: {e}")
        log.warning("Using current opt as fallback - this may work if config is compatible")
        
        # 使用當前 opt 作為 fallback
        import copy
        ckpt_opt = copy.deepcopy(opt)
        
        # 從 checkpoint 名稱推斷一些關鍵設定
        ckpt_name = ckpt_path.name
        if 'sde' in ckpt_name.lower():
            ckpt_opt.ot_ode = False
        if 'cond' in ckpt_name.lower() or 'x1' in ckpt_name.lower():
            ckpt_opt.cond_x1 = True
        else:
            ckpt_opt.cond_x1 = False
        
        # 設定預設值 (參考 train.py 的預設值)
        if not hasattr(ckpt_opt, 'corrupt'):
            ckpt_opt.corrupt = 'mixture'
        if not hasattr(ckpt_opt, 'interval'):
            ckpt_opt.interval = 1000
        if not hasattr(ckpt_opt, 'add_x1_noise'):
            ckpt_opt.add_x1_noise = False
        if not hasattr(ckpt_opt, 'spm'):
            ckpt_opt.spm = False
        if not hasattr(ckpt_opt, 'beta_max'):
            ckpt_opt.beta_max = 0.1
        if not hasattr(ckpt_opt, 't0'):
            ckpt_opt.t0 = 1e-4
        if not hasattr(ckpt_opt, 'T'):
            ckpt_opt.T = 1.0
        if not hasattr(ckpt_opt, 'clip_denoise'):
            ckpt_opt.clip_denoise = False
        if not hasattr(ckpt_opt, 'use_physics'):
            ckpt_opt.use_physics = False
        if not hasattr(ckpt_opt, 'image_size'):
            ckpt_opt.image_size = 256
        if not hasattr(ckpt_opt, 'latent_space'):
            ckpt_opt.latent_space = False
        if not hasattr(ckpt_opt, 'ema'):
            ckpt_opt.ema = 0.999
            
        log.info(f"Inferred settings: cond_x1={ckpt_opt.cond_x1}, ot_ode={ckpt_opt.ot_ode}, interval={ckpt_opt.interval}")

    overwrite_keys = ["use_fp16", "device"]
    for k in overwrite_keys:
        assert hasattr(opt, k)
        setattr(ckpt_opt, k, getattr(opt, k))

    ckpt_opt.load = ckpt_path / "latest.pt"
    return ckpt_opt
