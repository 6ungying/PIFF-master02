# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import pickle
import torch

from guided_diffusion.script_util import create_model

from . import util
from .ckpt_util import (
    I2SB_IMG256_UNCOND_PKL,
    I2SB_IMG256_UNCOND_CKPT,
    I2SB_IMG256_COND_PKL,
    I2SB_IMG256_COND_CKPT,
)

from ipdb import set_trace as debug
from torch import nn

class Image256Net(torch.nn.Module):
    # [MODIFIED] 同時支援 SPM 和 CA4D，透過參數控制
    def __init__(self, log, noise_levels, use_fp16=False, cond=False, pretrained_adm=True, ckpt_dir="data/", spm=False, ca4d=False):
        super(Image256Net, self).__init__()

        # initialize model
        # 使用固定的 pickle 路徑
        ckpt_pkl = "C:\\Users\\THINKLAB\\Desktop\\PIFF-master02\\data\\256x256_diffusion_uncond_fixedsigma.pkl"
        
        with open(ckpt_pkl, "rb") as f:
            kwargs = pickle.load(f)
        
        kwargs["use_fp16"] = use_fp16
        kwargs["num_channels"] = 128
        kwargs['image_size'] = 256
        kwargs['num_heads'] = 8
        kwargs["out_channels"] = 3
        # 網路本體接受 3 通道 (h, u, v)
        kwargs["in_channels"] = 3
        
        self.cond = False  # Disable conditional input (original i2sb cond)
        self.diffusion_model = create_model(**kwargs)
        
        # --- Fix first layer to accept 3 channels instead of 2 ---
        first_conv = self.diffusion_model.input_blocks[0][0]
        if first_conv.in_channels == 2:
            new_conv = torch.nn.Conv2d(
                in_channels=3,
                out_channels=first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias is not None
            )
            
            # Initialize new weights by expanding the original 2-channel weights
            with torch.no_grad():
                old_weight = first_conv.weight
                # For the third channel, use average of first two channels
                third_channel = (old_weight[:, 0:1, :, :] + old_weight[:, 1:2, :, :]) / 2
                new_weight = torch.cat([old_weight, third_channel], dim=1)
                new_conv.weight.copy_(new_weight)
                
                if first_conv.bias is not None:
                    new_conv.bias.copy_(first_conv.bias)
            
            self.diffusion_model.input_blocks[0][0] = new_conv
            log.info("[Net] Modified first layer to accept 3 input channels (h, u, v)")
        
        log.info(f"[Net] Initialized 3-channel network from {ckpt_pkl=}! Size={util.count_parameters(self.diffusion_model)}!")

        self.diffusion_model.apply(self.init_weights)
        self.diffusion_model.eval()
        
        # [MODIFIED] 同時支援 SPM 和 CA4D
        self.spm = spm
        self.ca4d = ca4d
        
        # 互斥檢查
        if self.spm and self.ca4d:
            log.warning("[Net] ⚠️  Both SPM and CA4D are enabled! Only one will be used during forward pass.")
            log.warning("[Net] Priority: CA4D > SPM (CA4D will be used if both are provided)")
        
        # Add SPM adapter if enabled (保留原始功能)
        if self.spm:
            # 1x1 adapter: [B,4,H,W] -> [B,3,H,W]
            # Input: x (3 channels) + spm (1 channel) = 4 channels
            self.spm_adapter = nn.Conv2d(in_channels=4, out_channels=3, kernel_size=1, stride=1, padding=0, bias=True)
            nn.init.kaiming_normal_(self.spm_adapter.weight, mode='fan_out', nonlinearity='relu')
            if self.spm_adapter.bias is not None:
                nn.init.zeros_(self.spm_adapter.bias)
            log.info("[Net] ✅ Added SPM adapter (4->3 channels) to integrate spatial prior maps")
        
        # Add CA4D adapter if enabled
        if self.ca4d:
            # 1x1 adapter: [B,6,H,W] -> [B,3,H,W]
            # Input: x (3 channels) + ca4d (3 channels) = 6 channels
            self.ca4d_adapter = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=1, stride=1, padding=0, bias=True)
            
            nn.init.kaiming_normal_(self.ca4d_adapter.weight, mode='fan_out', nonlinearity='relu')
            if self.ca4d_adapter.bias is not None:
                nn.init.zeros_(self.ca4d_adapter.bias)
            log.info("[Net] ✅ Added CA4D adapter (6->3 channels) to integrate h/vx/vy guidance")
        
        self.noise_levels = noise_levels

    # [MODIFIED] 同時支援 SPM 和 CA4D
    def forward(self, x, steps, rainfall, cond=None, spm=None, ca4d=None):

        t = steps.detach()
        if t.dim() > 1:
            t = t.view(t.shape[0], -1)[:, 0]
        t = t.to(dtype=torch.long, device=x.device)
        assert t.dim() == 1 and t.shape[0] == x.shape[0]

        # [PRIORITY] CA4D > SPM (如果兩者都提供，優先使用 CA4D)
        if self.ca4d and ca4d is not None:
            # CA4D integration
            if not isinstance(ca4d, torch.Tensor):
                ca4d = torch.as_tensor(ca4d, device=x.device)
            else:
                ca4d = ca4d.to(device=x.device)

            # 維度處理
            if ca4d.dim() == 3:
                ca4d = ca4d.unsqueeze(0)  # [3,H,W] -> [1,3,H,W]
            elif ca4d.dim() == 3 and ca4d.shape[1] != 3:
                ca4d = ca4d.unsqueeze(1).repeat(1, 3, 1, 1)

            # Broadcast batch dim
            if ca4d.shape[0] == 1 and x.shape[0] > 1:
                ca4d = ca4d.expand(x.shape[0], -1, -1, -1)

            # Ensure dtype matches
            if ca4d.dtype != x.dtype:
                ca4d = ca4d.to(dtype=x.dtype)

            # Concat: [B,3,H,W] + [B,3,H,W] -> [B,6,H,W]
            x = torch.cat([x, ca4d], dim=1)
            # Adapter: [B,6,H,W] -> [B,3,H,W]
            x = self.ca4d_adapter(x)
            
        elif self.spm and spm is not None:
            # SPM integration (保留原始邏輯)
            if not isinstance(spm, torch.Tensor):
                spm = torch.as_tensor(spm, device=x.device)
            else:
                spm = spm.to(device=x.device)

            # 維度處理 (SPM 是單通道)
            if spm.dim() == 2:
                spm = spm.unsqueeze(0).unsqueeze(0)  # [H,W] -> [1,1,H,W]
            elif spm.dim() == 3:
                if spm.shape[0] == x.shape[0]:
                    spm = spm.unsqueeze(1)  # [B,H,W] -> [B,1,H,W]
                else:
                    spm = spm.unsqueeze(0).unsqueeze(0)  # [H,W] -> [1,1,H,W]
            elif spm.dim() == 4:
                if spm.shape[1] != 1:
                    spm = spm[:, 0:1, ...]  # Take first channel

            # Broadcast batch dim
            if spm.shape[0] == 1 and x.shape[0] > 1:
                spm = spm.expand(x.shape[0], -1, -1, -1)

            # Ensure dtype matches
            if spm.dtype != x.dtype:
                spm = spm.to(dtype=x.dtype)

            # Concat: [B,3,H,W] + [B,1,H,W] -> [B,4,H,W]
            x = torch.cat([x, spm], dim=1)
            # Adapter: [B,4,H,W] -> [B,3,H,W]
            x = self.spm_adapter(x)

        output = self.diffusion_model(x, t, rainfall)
        return output  # [batch, 3, H, W]

    
    def init_weights(self, module):
        if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.Conv1d)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.GroupNorm, nn.LayerNorm)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)