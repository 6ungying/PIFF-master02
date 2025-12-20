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
    def __init__(self, log, noise_levels, use_fp16=False, cond=False, pretrained_adm=True, ckpt_dir="data/", spm=False):
        super(Image256Net, self).__init__()

        # initialize model
        # ckpt_pkl = os.path.join(ckpt_dir, I2SB_IMG256_COND_PKL if cond else I2SB_IMG256_UNCOND_PKL)
        ckpt_pkl = "C:\\Users\\THINKLAB\\Desktop\\PIFF-master02\\data\\256x256_diffusion_uncond_fixedsigma.pkl"
        with open(ckpt_pkl, "rb") as f:
            kwargs = pickle.load(f)
        kwargs["use_fp16"] = use_fp16
        kwargs["num_channels"] = 128
        kwargs['image_size'] = 256
        kwargs['num_heads'] = 8
        kwargs["out_channels"] = 3
        # Always use 3 channels for h, u, v (no conditional input)
        kwargs["in_channels"] = 3
        self.cond = False  # Disable conditional input
        self.diffusion_model = create_model(**kwargs)
        
        # Fix first layer to accept 3 channels instead of 2
        first_conv = self.diffusion_model.input_blocks[0][0]
        if first_conv.in_channels == 2:
            # Create new conv layer with 3 input channels
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
                old_weight = first_conv.weight  # [out_channels, 2, kernel_h, kernel_w]
                # For the third channel, use average of first two channels
                third_channel = (old_weight[:, 0:1, :, :] + old_weight[:, 1:2, :, :]) / 2
                new_weight = torch.cat([old_weight, third_channel], dim=1)
                new_conv.weight.copy_(new_weight)
                
                if first_conv.bias is not None:
                    new_conv.bias.copy_(first_conv.bias)
            
            # Replace the first layer
            self.diffusion_model.input_blocks[0][0] = new_conv
            log.info("[Net] Modified first layer to accept 3 input channels (h, u, v)")
        
        log.info(f"[Net] Initialized 3-channel network from {ckpt_pkl=}! Size={util.count_parameters(self.diffusion_model)}!")
        
        # load (modified) adm ckpt
        # if pretrained_adm:
            # ckpt_pt = os.path.join(ckpt_dir, I2SB_IMG256_COND_CKPT if cond else I2SB_IMG256_UNCOND_CKPT)
            # out = torch.load(ckpt_pt, map_location="cpu")
            # self.diffusion_model.load_state_dict(out)
            # log.info(f"[Net] Loaded pretrained adm {ckpt_pt=}!")

        self.diffusion_model.apply(self.init_weights)
        
        self.diffusion_model.eval()
        # self.cond already set above during initialization
        self.spm = spm
        
        # Add SPM adapter if enabled
        if self.spm:
            # 1x1 adapter to map 4-channel (h,u,v,spm) -> 3-channel (h,u,v)
            # This preserves pretrained weights while allowing SPM integration
            self.spm_adapter = nn.Conv2d(in_channels=4, out_channels=3, kernel_size=1, stride=1, padding=0, bias=True)
            nn.init.kaiming_normal_(self.spm_adapter.weight, mode='fan_out', nonlinearity='relu')
            if self.spm_adapter.bias is not None:
                nn.init.zeros_(self.spm_adapter.bias)
            log.info("[Net] Added SPM adapter (4->3 channels) to integrate spatial prior maps")
        
        self.noise_levels = noise_levels

    def forward(self, x, steps, rainfall, cond=None, spm=None):

        # Accept steps in several shapes: either [B] or [B,1,1,1], etc.
        # t must be a 1-D LongTensor with length batch-size.
        t = steps.detach()
        if t.dim() > 1:
            # collapse extra dims, keep first element per batch
            t = t.view(t.shape[0], -1)[:, 0]
        # ensure correct dtype/device
        t = t.to(dtype=torch.long, device=x.device)
        assert t.dim() == 1 and t.shape[0] == x.shape[0]

        # Conditional input disabled for 3-channel mode
        # x = torch.cat([x, cond], dim=1) if self.cond else x
        
        # SPM integration with robust dimension handling
        if self.spm and spm is not None:
            # Sanitize spm shape/dtype to avoid dimension mismatch errors.
            # Expected spm shape: [B,1,H,W]. Accept common alternatives and convert.
            if not isinstance(spm, torch.Tensor):
                spm = torch.as_tensor(spm, device=x.device)
            else:
                spm = spm.to(device=x.device)

            # Handle possible shapes:
            #  - [H, W] -> [1,1,H,W]
            #  - [B, H, W] -> [B,1,H,W]
            #  - [1, H, W] -> [1,1,H,W]
            #  - [B, C, H, W] (C!=1) -> take first channel
            if spm.dim() == 2:
                spm = spm.unsqueeze(0).unsqueeze(0)
            elif spm.dim() == 3:
                # ambiguous: assume [B, H, W]
                if spm.shape[0] == x.shape[0]:
                    spm = spm.unsqueeze(1)
                else:
                    # treat as single sample [H,W] with missing batch dim
                    spm = spm.unsqueeze(0).unsqueeze(0)
            elif spm.dim() == 4:
                if spm.shape[1] != 1:
                    spm = spm[:, 0:1, ...]

            # Broadcast batch dim if needed (e.g., spm has batch 1 but x has batch >1)
            if spm.shape[0] == 1 and x.shape[0] > 1:
                spm = spm.expand(x.shape[0], -1, -1, -1)

            # Ensure dtype matches
            if spm.dtype != x.dtype:
                spm = spm.to(dtype=x.dtype)

            # Concat SPM as 4th channel: [B,3,H,W] + [B,1,H,W] -> [B,4,H,W]
            x = torch.cat([x, spm], dim=1)
            # Use adapter to map back to 3 channels: [B,4,H,W] -> [B,3,H,W]
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
