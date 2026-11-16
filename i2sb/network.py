# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import torch

from guided_diffusion.script_util import create_model

from . import util
from torch import nn

class Image256Net(torch.nn.Module):
    # [MODIFIED] 同時支援 SPM 和 CA4D，透過參數控制
    def __init__(self, log, noise_levels, use_fp16=False, cond=False, pretrained_adm=False, ckpt_dir="data/", spm=False, ca4d=False):
        super(Image256Net, self).__init__()

        # Build a depth-only UNet; no external architecture pickle is needed.
        self.cond = False
        self.diffusion_model = create_model(
            image_size=256, num_channels=128, num_res_blocks=2,
            attention_resolutions="32,16,8", num_heads=8,
            use_scale_shift_norm=True, resblock_updown=True,
            in_channels=1, out_channels=1, use_fp16=use_fp16,
        )
        log.info(f"[Net] Initialized depth-only network! Size={util.count_parameters(self.diffusion_model)}")

        self.diffusion_model.apply(self.init_weights)
        self.diffusion_model.eval()
        
        # [MODIFIED] 同時支援 SPM 和 CA4D
        self.spm = spm
        self.ca4d = ca4d
        
        # 互斥檢查
        if self.spm and self.ca4d:
            log.warning("[Net] Both SPM and CA4D are enabled! Only one will be used during forward pass.")
            log.warning("[Net] Priority: CA4D > SPM (CA4D will be used if both are provided)")
        
        # Add SPM adapter if enabled (保留原始功能)
        if self.spm:
            self.spm_adapter = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
            nn.init.kaiming_normal_(self.spm_adapter.weight, mode='fan_out', nonlinearity='relu')
            if self.spm_adapter.bias is not None:
                nn.init.zeros_(self.spm_adapter.bias)
            log.info("[Net] Added SPM adapter (2->1 channels) to integrate spatial prior maps")
        
        # Add CA4D adapter if enabled
        if self.ca4d:
            self.ca4d_adapter = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
            
            nn.init.kaiming_normal_(self.ca4d_adapter.weight, mode='fan_out', nonlinearity='relu')
            if self.ca4d_adapter.bias is not None:
                nn.init.zeros_(self.ca4d_adapter.bias)
            log.info("[Net] Added CA4D adapter (2->1 channels) for depth-only (d) guidance")
        
        self.noise_levels = noise_levels

    # [MODIFIED] 同時支援 SPM 和 CA4D
    def forward(self, x, steps, rainfall, cond=None, spm=None, ca4d=None):
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError('Depth-only network expects [B, 1, H, W].')

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
            if ca4d.dim() == 2:
                ca4d = ca4d.unsqueeze(0).unsqueeze(0)
            elif ca4d.dim() == 3:
                ca4d = ca4d.unsqueeze(1)
            if ca4d.dim() != 4 or ca4d.shape[1] != 1:
                raise ValueError('CA4D guidance must contain depth only: [B, 1, H, W].')

            # Broadcast batch dim
            if ca4d.shape[0] == 1 and x.shape[0] > 1:
                ca4d = ca4d.expand(x.shape[0], -1, -1, -1)

            # Ensure dtype matches
            if ca4d.dtype != x.dtype:
                ca4d = ca4d.to(dtype=x.dtype)

            x = torch.cat([x, ca4d], dim=1)
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
                    spm = spm.unsqueeze(0)  # [H,W] -> [1,1,H,W]
            elif spm.dim() == 4:
                if spm.shape[1] != 1:
                    spm = spm[:, 0:1, ...]  # Take first channel

            # Broadcast batch dim
            if spm.shape[0] == 1 and x.shape[0] > 1:
                spm = spm.expand(x.shape[0], -1, -1, -1)

            # Ensure dtype matches
            if spm.dtype != x.dtype:
                spm = spm.to(dtype=x.dtype)

            x = torch.cat([x, spm], dim=1)
            x = self.spm_adapter(x)

        output = self.diffusion_model(x, t, rainfall)
        return output  # [batch, 1, H, W] (depth)

    
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
