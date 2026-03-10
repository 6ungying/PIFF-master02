# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
from torch.utils.tensorboard import SummaryWriter
import wandb

import torch
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def custom_collate_fn(batch):
    """
    自訂 collate 函數來處理包含 None 的 mask 與 next_timestep_data
    batch: list of tuples from dataset.__getitem__
    """
    # 分離出各個欄位
    flood_imgs = []
    vx_imgs = []
    vy_imgs = []
    dem_imgs = []
    
    mask_floods = []
    mask_vxs = []
    mask_vys = []
    
    rainfalls = []
    img_paths = []
    vx_paths = []
    vy_paths = []
    
    ca4ds = [] # 改名 spm -> ca4d
    
    next_timestep_datas = []
    max_depths = []
    dem_ids = []
    
    for item in batch:
        # 根據回傳長度解包 (支援 16, 15, 13, 12 等舊格式相容)
        if len(item) == 16:
            # 最新格式 (同時包含 spm_image 和 ca4d_image)
            (flood, vx, vy, dem, mask_flood, mask_vx, mask_vy, 
             rainfall, img_path, vx_path, vy_path, spm, ca4d, next_data, max_depth, dem_id) = item
        elif len(item) == 15:
            # 舊格式 (含 ca4d, max_depth, dem_id)
            (flood, vx, vy, dem, mask_flood, mask_vx, mask_vy, 
             rainfall, img_path, vx_path, vy_path, ca4d, next_data, max_depth, dem_id) = item
        elif len(item) == 13:
            # 更舊格式
            (flood, vx, vy, dem, mask_flood, mask_vx, mask_vy, 
             rainfall, img_path, vx_path, vy_path, ca4d, next_data) = item
            max_depth = 4.0
            dem_id = -1
        elif len(item) == 12:
            (flood, vx, vy, dem, mask_flood, mask_vx, mask_vy, 
             rainfall, img_path, vx_path, vy_path, ca4d) = item
            next_data = None
            max_depth = 4.0
            dem_id = -1
        else:
            raise ValueError(f"Unexpected item length: {len(item)}")
        
        flood_imgs.append(flood)
        vx_imgs.append(vx)
        vy_imgs.append(vy)
        dem_imgs.append(dem)
        
        # [MODIFIED] Mask 處理：支援 None
        mask_floods.append(torch.from_numpy(mask_flood) if (mask_flood is not None and not torch.is_tensor(mask_flood)) else mask_flood)
        mask_vxs.append(torch.from_numpy(mask_vx) if (mask_vx is not None and not torch.is_tensor(mask_vx)) else mask_vx)
        mask_vys.append(torch.from_numpy(mask_vy) if (mask_vy is not None and not torch.is_tensor(mask_vy)) else mask_vy)
        
        rainfalls.append(rainfall)
        img_paths.append(img_path)
        vx_paths.append(vx_path)
        vy_paths.append(vy_path)
        
        ca4ds.append(ca4d)
        
        next_timestep_datas.append(next_data)
        max_depths.append(max_depth)
        dem_ids.append(dem_id)
    
    # Stack tensors
    flood_batch = torch.stack(flood_imgs)
    vx_batch = torch.stack(vx_imgs)
    vy_batch = torch.stack(vy_imgs)
    dem_batch = torch.stack(dem_imgs)
    
    # [MODIFIED] 若列表中包含 None，則 Batch 結果也為 None (避免 stack 報錯)
    if any(m is None for m in mask_floods):
        mask_flood_batch = None
    else:
        mask_flood_batch = torch.stack(mask_floods)

    if any(m is None for m in mask_vxs):
        mask_vx_batch = None
    else:
        mask_vx_batch = torch.stack(mask_vxs)

    if any(m is None for m in mask_vys):
        mask_vy_batch = None
    else:
        mask_vy_batch = torch.stack(mask_vys)
    
    # Rainfall
    if torch.is_tensor(rainfalls[0]):
        rainfall_batch = torch.stack(rainfalls)
    else:
        import numpy as np
        rainfall_batch = torch.from_numpy(np.array(rainfalls))
    
    # CA4D Stack (處理 None 的情況)
    if all(ca4d is None for ca4d in ca4ds):
        # 如果所有 CA4D 都是 None (測試模式),返回 None
        ca4d_batch = None
    elif any(ca4d is None for ca4d in ca4ds):
        # 如果部分是 None,用零填充
        reference_shape = next((c.shape for c in ca4ds if c is not None), None)
        ca4ds_filled = [ca4d if ca4d is not None else torch.zeros(reference_shape) for ca4d in ca4ds]
        ca4d_batch = torch.stack(ca4ds_filled)
    else:
        # 所有都是有效的 tensor
        ca4d_batch = torch.stack(ca4ds)
    
    # Meta data
    max_depth_batch = torch.tensor(max_depths, dtype=torch.float32)
    dem_id_batch = torch.tensor(dem_ids, dtype=torch.long)
    
    return (flood_batch, vx_batch, vy_batch, dem_batch,
            mask_flood_batch, mask_vx_batch, mask_vy_batch,
            rainfall_batch, img_paths, vx_paths, vy_paths, ca4d_batch,
            next_timestep_datas, max_depth_batch, dem_id_batch)

def setup_loader(dataset, batch_size, num_workers=4):
    loader = DataLoaderX(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=custom_collate_fn,
    )

    while True:
        yield from loader

class BaseWriter(object):
    def __init__(self, opt):
        self.rank = opt.global_rank
    def add_scalar(self, step, key, val):
        pass # do nothing
    def add_image(self, step, key, image):
        pass # do nothing
    def close(self): pass

class WandBWriter(BaseWriter):
    def __init__(self, opt):
        super(WandBWriter,self).__init__(opt)
        if self.rank == 0:
            assert wandb.login(key=opt.wandb_api_key)
            wandb.init(dir=str(opt.log_dir), project="i2sb", entity=opt.wandb_user, name=opt.name, config=vars(opt))

    def add_scalar(self, step, key, val):
        if self.rank == 0: wandb.log({key: val}, step=step)

    def add_image(self, step, key, image):
        if self.rank == 0:
            image = image.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            wandb.log({key: wandb.Image(image)}, step=step)


class TensorBoardWriter(BaseWriter):
    def __init__(self, opt):
        super(TensorBoardWriter,self).__init__(opt)
        if self.rank == 0:
            run_dir = str(opt.log_dir / opt.name)
            os.makedirs(run_dir, exist_ok=True)
            self.writer=SummaryWriter(log_dir=run_dir, flush_secs=20)

    def add_scalar(self, global_step, key, val):
        if self.rank == 0: self.writer.add_scalar(key, val, global_step=global_step)

    def add_image(self, global_step, key, image):
        if self.rank == 0:
            image = image.mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8)
            self.writer.add_image(key, image, global_step=global_step)

    def close(self):
        if self.rank == 0: self.writer.close()

def build_log_writer(opt):
    if opt.log_writer == 'wandb': return WandBWriter(opt)
    elif opt.log_writer == 'tensorboard': return TensorBoardWriter(opt)
    else: return BaseWriter(opt)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def space_indices(num_steps, count):
    assert count <= num_steps

    if count <= 1:
        frac_stride = 1
    else:
        frac_stride = (num_steps - 1) / (count - 1)

    cur_idx = 0.0
    taken_steps = []
    for _ in range(count):
        taken_steps.append(round(cur_idx))
        cur_idx += frac_stride

    return taken_steps

def unsqueeze_xdim(z, xdim):
    bc_dim = (...,) + (None,) * len(xdim)
    return z[bc_dim]