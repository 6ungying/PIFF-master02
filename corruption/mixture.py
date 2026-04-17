# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import numpy as np
import enum
import pandas as pd 
import cv2
from PIL import Image
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from .jpeg import jpeg_decode, jpeg_encode
from .blur import Deblurring
from .superresolution import build_sr_bicubic, build_sr_pool
from .inpaint import get_center_mask, load_freeform_masks

from ipdb import set_trace as debug


class AllCorrupt(enum.IntEnum):
    JPEG_5 = 0
    JPEG_10 = 1
    BLUR_UNI = 2
    BLUR_GAUSS = 3
    SR4X_POOL = 4
    SR4X_BICUBIC = 5
    INPAINT_CENTER = 6
    INPAINT_FREE1020 = 7
    INPAINT_FREE2030 = 8

class MixtureCorruptMethod:
    def __init__(self, opt, device="cpu"):

        # ===== blur ====
        self.blur_uni = Deblurring(torch.Tensor([1/9] * 9).to(device), 3, opt.image_size, device)

        sigma = 10
        pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma)**2]))
        g_kernel = torch.Tensor([pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2)]).to(device)
        self.blur_gauss = Deblurring(g_kernel / g_kernel.sum(), 3, opt.image_size, device)

        # ===== sr4x ====
        factor = 4
        self.sr4x_pool = build_sr_pool(factor, device, opt.image_size)
        self.sr4x_bicubic = build_sr_bicubic(factor, device, opt.image_size)
        self.upsample = torch.nn.Upsample(scale_factor=factor, mode='nearest')

        # ===== inpaint ====
        self.center_mask = get_center_mask([opt.image_size, opt.image_size])[None,...] # [1, 1, 256, 256]
        self.free1020_masks = torch.from_numpy((load_freeform_masks("freeform1020"))) # [10000, 1, 256, 256]
        self.free2030_masks = torch.from_numpy((load_freeform_masks("freeform2030"))) # [10000, 1, 256, 256]

    def jpeg(self, img, qf):
        return jpeg_decode(jpeg_encode(img, qf), qf)

    def blur(self, img, kernel):
        img = (img + 1) / 2
        if kernel == "uni":
            _img = self.blur_uni.H(img).reshape(*img.shape)
        elif kernel == "gauss":
            _img = self.blur_gauss.H(img).reshape(*img.shape)
        # [0,1] -> [-1,1]
        return _img * 2 - 1

    def sr4x(self, img, filter):
        b, c, w, h = img.shape
        if filter == "pool":
            _img = self.sr4x_pool.H(img).reshape(b, c, w // 4, h // 4)
        elif filter == "bicubic":
            _img = self.sr4x_bicubic.H(img).reshape(b, c, w // 4, h // 4)

        # scale to original image size for I2SB
        return self.upsample(_img)

    def inpaint(self, img, mask_type, mask_index=None):
        if mask_type == "center":
            mask = self.center_mask
        elif mask_type == "free1020":
            if mask_index is None:
                mask_index = np.random.randint(len(self.free1020_masks))
            mask = self.free1020_masks[[mask_index]]
        elif mask_type == "free2030":
            if mask_index is None:
                mask_index = np.random.randint(len(self.free2030_masks))
            mask = self.free2030_masks[[mask_index]]
        return img * (1. - mask) + mask * torch.randn_like(img)

    def mixture(self, img, corrupt_idx, mask_index=None):
        if corrupt_idx == AllCorrupt.JPEG_5:
            corrupt_img = self.jpeg(img, 5)
        elif corrupt_idx == AllCorrupt.JPEG_10:
            corrupt_img = self.jpeg(img, 10)
        elif corrupt_idx == AllCorrupt.BLUR_UNI:
            corrupt_img = self.blur(img, "uni")
        elif corrupt_idx == AllCorrupt.BLUR_GAUSS:
            corrupt_img = self.blur(img, "gauss")
        elif corrupt_idx == AllCorrupt.SR4X_POOL:
            corrupt_img = self.sr4x(img, "pool")
        elif corrupt_idx == AllCorrupt.SR4X_BICUBIC:
            corrupt_img = self.sr4x(img, "bicubic")
        elif corrupt_idx == AllCorrupt.INPAINT_CENTER:
            corrupt_img = self.inpaint(img, "center")
        elif corrupt_idx == AllCorrupt.INPAINT_FREE1020:
            corrupt_img = self.inpaint(img, "free1020", mask_index=mask_index)
        elif corrupt_idx == AllCorrupt.INPAINT_FREE2030:
            corrupt_img = self.inpaint(img, "free2030", mask_index=mask_index)
        return corrupt_img


class MixtureCorruptDatasetTrain(Dataset):
    def __init__(self, opt, dataset):
        super(MixtureCorruptDatasetTrain, self).__init__()
        self.dataset = dataset
        self.method = MixtureCorruptMethod(opt)

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, index):
        clean_img, y = self.dataset[index] # clean_img: tensor [-1,1]

        rand_idx = np.random.choice(AllCorrupt)
        corrupt_img = self.method.mixture(clean_img.unsqueeze(0), rand_idx).squeeze(0)

        assert corrupt_img.shape == clean_img.shape, (clean_img.shape, corrupt_img.shape)
        return clean_img, corrupt_img, y


class MixtureCorruptDatasetVal(Dataset):
    def __init__(self, opt, dataset):
        super(MixtureCorruptDatasetVal, self).__init__()
        self.dataset = dataset
        self.method = MixtureCorruptMethod(opt)

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, index):
        clean_img, y = self.dataset[index] # clean_img: tensor [-1,1]

        idx = index % len(AllCorrupt)
        corrupt_img = self.method.mixture(clean_img.unsqueeze(0), idx, mask_index=idx).squeeze(0)

        assert corrupt_img.shape == clean_img.shape, (clean_img.shape, corrupt_img.shape)
        return clean_img, corrupt_img, y


class floodDataset(Dataset):
    def __init__(self, opt, val=False, test=False):
        super(floodDataset, self).__init__()
        self.opt = opt
        dem_path = 'C:\\Users\\THINKLAB\\Desktop\\PIFF-master02\\data\\dems\\dem_png'
        
        # [MODIFIED] 同時支援 SPM 和 CA4D 路徑
        self.spm_folder = 'C:\\Users\\THINKLAB\\Desktop\\PIFF-master02\\data\\SPM_71dem_output'
        # CA4D 資料夾: 根據測試/訓練模式選擇不同資料集
        # - Multi: 多 DEM 訓練 (30 DEMs × 86 scenarios)
        # - single_train: 單 DEM 訓練 (1 DEM × 182 scenarios)
        # - single_test: 單 DEM 測試 (1 DEM × 15 scenarios)
        if test:
            self.ca4d_folder = 'C:\\Users\\THINKLAB\\Desktop\\PIFF-master02\\data\\ca4d\\Multi'
        else:
            # 訓練模式: 使用 Multi (多 DEM) 或 single_train (單 DEM)
            # 可以根據 opt.ca4d_dataset 參數選擇
            ca4d_dataset = getattr(opt, 'ca4d_dataset', 'Multi') if opt is not None else 'Multi'
            self.ca4d_folder = f'C:\\Users\\THINKLAB\\Desktop\\PIFF-master02\\data\\ca4d\\{ca4d_dataset}'
        
        # 控制載入哪個物理模型 (從 opt 參數讀取)
        self.use_spm = getattr(opt, 'spm', False) if opt is not None else False
        self.use_ca4d = getattr(opt, 'ca4d', False) if opt is not None else False
        self.use_ca4d = True

        self.dem_stat = 'C:\\Users\\THINKLAB\\Desktop\\PIFF-master02\\data\\dems\\dem_png\\elevation_stats.csv'
        self.dem_stat = pd.read_csv(self.dem_stat)
        
        # ===== 整合 CSV 讀取：載入地形深度數據 =====
        # 從 maxmin_duv.csv 讀取每個地形的最大深度（用於物理損失）
        maxmin_csv_path = 'C:\\Users\\THINKLAB\\Desktop\\PIFF-master02\\data\\maxmin_duv.csv'
        if os.path.exists(maxmin_csv_path):
            terrain_df = pd.read_csv(maxmin_csv_path)
            self.terrain_depths = dict(zip(terrain_df['terrain'].astype(int), terrain_df['depth_max']))
            print(f"[Dataset] [OK] Loaded terrain depths from CSV: {len(self.terrain_depths)} terrains")
        else:
            self.terrain_depths = {}
            print(f"[Dataset] [WARNING] maxmin_duv.csv not found, using default depth 4.0m")
        
        self.test = test

        self.flood_path = 'C:\\Users\\THINKLAB\\Desktop\\PIFF-master02\\data\\dems\\train\\d'
        self.vx_path = 'C:\\Users\\THINKLAB\\Desktop\\PIFF-master02\\data\\dems\\train\\vx'
        self.vy_path = 'C:\\Users\\THINKLAB\\Desktop\\PIFF-master02\\data\\dems\\train\\vy'
        rainfall_path = 'C:\\Users\\THINKLAB\\Desktop\\PIFF-master02\\data\\dems\\scenario_rainfall.csv'
        
        # 使用所有可用的 DEM 資料夾進行訓練
        all_available_dems = [int(f) for f in os.listdir(self.flood_path) if os.path.isdir(os.path.join(self.flood_path, f))]
        
        if test:
            # 測試模式: 優先使用命令列指定的 test_dem_list
            if hasattr(opt, 'test_dem_list') and opt.test_dem_list is not None:
                dem_folder = opt.test_dem_list
                print(f"[Test mode] Using specified DEMs: {dem_folder}")
            else:
                # 沒指定就使用預設
                dem_folder = [8, 29, 62]
                print(f"[Test mode] Using default test DEMs: {dem_folder}")
        else:
            # 訓練模式: 使用所有可用 DEM,但排除測試用的 DEM
            if hasattr(opt, 'test_dem_list') and opt.test_dem_list is not None:
                # 從所有可用 DEM 中排除測試 DEM
                dem_folder = [dem for dem in all_available_dems if dem not in opt.test_dem_list]
                print(f"[Training mode] Excluding test DEMs {opt.test_dem_list}")
                print(f"[Training mode] Using DEMs: {sorted(dem_folder)}")
            else:
                # 沒指定測試 DEM,使用全部
                dem_folder = all_available_dems
                print(f"[Training mode] Using all available DEMs: {sorted(dem_folder)}")

        rainfall = pd.read_csv(rainfall_path)
        rainfall = rainfall.iloc[:, :]

        # Initialize lists to store cell values and their positions
        rainfall_cum_value = []
        cell_positions = []
        spm_values = []  # [RESTORED] 儲存 SPM 累積降雨值

        val = False
        # Iterate through each column
        for dem_num in dem_folder:
            for col in rainfall.columns:
                if col == 'time':
                    continue
                col_num = int(col.split("_")[1])
                if (val and col_num not in [2]) or (not val and col_num in []):
                    continue
                cell_values = []
                # Iterate through each row in the current column
                for row in range(25):
                    cell_value = rainfall.iloc[row][col]
                    cell_values.append(np.ceil(cell_value))
                    # make it a len 24 list if not append 0 in front
                    temp = [0] * (24 - len(cell_values))
                    temp.extend(cell_values)
                    if len(temp) == 25:
                        temp = temp[1:]
                    sum_rainfall = sum(temp[:])
                    
                    if not test and sum_rainfall <= 5:
                        if np.random.rand() < 0.8:
                            continue
                    
                    # [MODIFIED] 彈性檢查：根據 use_spm / use_ca4d 決定要檢查哪個
                    skip_sample = False
                    
                    # 檢查 SPM (如果啟用)
                    spm_value = int(np.ceil(sum_rainfall / 5) * 5)
                    spm_path = os.path.join(self.spm_folder, str(dem_num), f'SPM_{dem_num}_{spm_value}.png')
                    if not os.path.exists(spm_path):
                        skip_sample = True
                    
                    # 檢查 CA4D (如果啟用)
                
                    dem_str = str(dem_num)
                        # CA4D 使用小寫 rf (不是 RF)
                    rf_scenario = f"rf{col_num:02d}"
                        # CA4D timestep 從 001 開始 (row+1)
                    time_str = f"{row:03d}"
                        
                        # CA4D 資料夾結構: ca4d_folder/d/{DEM}/rf{scenario}/ca4d_{DEM}_rf{scenario}_d_{timestep}.png
                    path_d = os.path.join(self.ca4d_folder, 'd', dem_str, rf_scenario,
                                            f"ca4d_{dem_str}_{rf_scenario}_d_{time_str}.png")
                    path_vx = os.path.join(self.ca4d_folder, 'vx', dem_str, rf_scenario,
                                             f"ca4d_{dem_str}_{rf_scenario}_vx_{time_str}.png")
                    path_vy = os.path.join(self.ca4d_folder, 'vy', dem_str, rf_scenario,
                                             f"ca4d_{dem_str}_{rf_scenario}_vy_{time_str}.png")
                        
                    if not all([os.path.exists(p) for p in [path_d, path_vx, path_vy]]):
                        skip_sample = True
                    
                    if skip_sample:
                        continue
                    
                    # 檢查 Ground Truth
                    dem_str = str(dem_num)
                    if col_num < 100:
                        folder_name = f"RF{col_num:02d}"
                    else:
                        folder_name = f"RF{col_num}"
                    
                    flood_name = f"{dem_str}_{folder_name}_d_{row:03d}_00.png"
                    vx_name = f"{dem_str}_{folder_name}_vx_{row:03d}_00.png"
                    vy_name = f"{dem_str}_{folder_name}_vy_{row:03d}_00.png"
                    
                    flood_path = os.path.join(self.flood_path, dem_str, folder_name, flood_name)
                    vx_path = os.path.join(self.vx_path, dem_str, folder_name, vx_name)
                    vy_path = os.path.join(self.vy_path, dem_str, folder_name, vy_name)
                    
                    if not all([os.path.exists(p) for p in [flood_path, vx_path, vy_path]]):
                        continue
                    
                    # [RESTORED] 儲存 SPM 值 (如果啟用)
                    spm_values.append(spm_value)
                    
                    rainfall_cum_value.append(temp[:])
                    # col_num is the rainfall index, and row is the time index
                    cell_positions.append((dem_num, col_num, row))   

        self.rainfall = rainfall_cum_value
        self.cell_positions = cell_positions
        self.spm_values = spm_values  # [RESTORED] 儲存 SPM 值列表
        
        # 打印訓練資料統計
        total_scenarios = len(dem_folder) * len([c for c in rainfall.columns if c != 'time']) * len(rainfall)
        
        # 顯示載入的物理模型類型
        model_types = []
        if self.use_spm:
            model_types.append("SPM")
        if self.use_ca4d:
            model_types.append("CA4D")
        model_str = " + ".join(model_types) if model_types else "None"
        
        print(f"[OK] Dataset initialized: {len(self.cell_positions)} valid samples")
        print(f"     Physical Models: {model_str}")
        print(f"     (Filtered {total_scenarios - len(self.cell_positions)} samples with missing files)")
        
        # add transform , to tensor and normalize
        self.transform = T.Compose([
            T.ToTensor(),
        ])

    def __find_flood_image(self, cell_position, flood_path):
        dem, col, row = cell_position
        dem_folder_name = str(dem)
        if col < 100:
            folder_name = f"RF{col:02d}"
        else:
            folder_name = f"RF{col}"
        # 正確的檔案名稱格式: {dem}_RF{scenario}_d_{timestep}_00.png
        image_name = f"{dem_folder_name}_{folder_name}_d_{row:03d}_00.png"
        image_path = os.path.join(flood_path, dem_folder_name, folder_name, image_name)
        return image_path
    
    def __find_vx_image(self, cell_position):
        dem, col, row = cell_position
        dem_folder_name = str(dem)
        if col < 100:
            folder_name = f"RF{col:02d}"
        else:
            folder_name = f"RF{col}"
        # 正確的檔案名稱格式: {dem}_RF{scenario}_vx_{timestep}_00.png
        image_name = f"{dem_folder_name}_{folder_name}_vx_{row:03d}_00.png"
        image_path = os.path.join(self.vx_path, dem_folder_name, folder_name, image_name)
        return image_path
    
    def __find_vy_image(self, cell_position):
        dem, col, row = cell_position
        dem_folder_name = str(dem)
        if col < 100:
            folder_name = f"RF{col:02d}"
        else:
            folder_name = f"RF{col}"
        # 正確的檔案名稱格式: {dem}_RF{scenario}_vy_{timestep}_00.png
        image_name = f"{dem_folder_name}_{folder_name}_vy_{row:03d}_00.png"
        image_path = os.path.join(self.vy_path, dem_folder_name, folder_name, image_name)
        return image_path
    
    def __find_dem_image(self, cell_position):
        dem_num = cell_position[0]
        # 直接使用固定路徑
        dem_folder = Path('C:\\Users\\THINKLAB\\Desktop\\PIFF-master02\\data\\dems\\dem_png')
        dem_path = dem_folder / f'{dem_num}.png'
        return str(dem_path)

    def get_next_timestep_data(self, dem, col, row):
        """
        獲取下一個時間步的水深、Vx、Vy 資料 (row+1)
        回傳: (next_flood_image, next_vx_image, next_vy_image, None, None, None)
        """
        try:
            next_row = row + 1
            next_cell_position = (dem, col, next_row)
            
            # 檢查所有檔案是否存在
            next_flood_path = self.__find_flood_image(next_cell_position, self.flood_path)
            next_vx_path = self.__find_vx_image(next_cell_position)
            next_vy_path = self.__find_vy_image(next_cell_position)
            
            if not all([os.path.exists(p) for p in [next_flood_path, next_vx_path, next_vy_path]]):
                return None
                
            # 讀取影像
            next_flood_image = cv2.imread(next_flood_path, cv2.IMREAD_UNCHANGED)
            next_vx_image = cv2.imread(next_vx_path, cv2.IMREAD_UNCHANGED)
            next_vy_image = cv2.imread(next_vy_path, cv2.IMREAD_UNCHANGED)
            
            # [MODIFIED] Mask 設為 None
            # next_binary_mask = (next_flood_image <= 250).astype('uint8') ...
            next_binary_mask = None
            next_vx_binary_mask = None
            next_vy_binary_mask = None
            
            # 轉換為 uint8
            next_flood_image = np.array(next_flood_image, dtype=np.uint8)
            next_vx_image = np.array(next_vx_image, dtype=np.uint8)
            next_vy_image = np.array(next_vy_image, dtype=np.uint8)
            
            # 轉換為 tensor 並標準化
            next_flood_image = self.transform(next_flood_image)
            next_vx_image = self.transform(next_vx_image)
            next_vy_image = self.transform(next_vy_image)
            
            next_flood_image = (next_flood_image - 0.98) / 0.056
            next_vx_image = (next_vx_image - 0.497) / 0.0043
            next_vy_image = (next_vy_image - 0.497) / 0.0047
            
            return (next_flood_image, next_vx_image, next_vy_image, 
                    next_binary_mask, next_vx_binary_mask, next_vy_binary_mask)
                   
        except Exception as e:
            return None
    
    def __len__(self):
        return len(self.cell_positions)

    def __getitem__(self, index):
        cell_position = self.cell_positions[index]
        rainfall = self.rainfall[index]
        spm_value = self.spm_values[index]  # [RESTORED] 取得 SPM 值
        
        dem_path = self.__find_dem_image(cell_position)
        
        # ===== 獲取地形深度（用於物理損失）=====
        dem_id = cell_position[0]
        max_depth = self.terrain_depths.get(dem_id, 4.0)  # 預設 4.0m
        
        # 讀取 DEM
        dem_image = cv2.imread(dem_path, cv2.IMREAD_UNCHANGED)
        if dem_image is None:
            raise FileNotFoundError(f"Failed to load DEM image: {dem_path}")
        
        if len(dem_image.shape) == 3:
            dem_image = dem_image[:,:,0]
        
        dem_cur_state = self.dem_stat[self.dem_stat['Filename'] == cell_position[0]]
        min_elev = int(dem_cur_state['Min Elevation'].iloc[0])
        max_elev = int(dem_cur_state['Max Elevation'].iloc[0])
        real_height = dem_image / 255 * (max_elev - min_elev) + min_elev
        dem_image = (real_height - (-3)) / (125 + 3) * 255
        dem_image = np.clip(dem_image, 0, 255)
        dem_image = np.array(dem_image, dtype=np.uint8)
        
        # [MODIFIED] 根據設定載入 SPM 或 CA4D
        dem_str = str(cell_position[0])
        col_num = cell_position[1]
        row_num = cell_position[2]
        
        # 初始化為 None
        spm_image = None
        ca4d_image = None
        
        # 載入 SPM (如果啟用)
            # SPM 格式: SPM_{DEM}_{rainfall}.png
        spm_path = os.path.join(self.spm_folder, dem_str, f'SPM_{dem_str}_{spm_value}.png')
        spm_image = cv2.imread(spm_path, cv2.IMREAD_GRAYSCALE)
            
        if spm_image is None:
            raise FileNotFoundError(f"SPM image not found: {spm_path}")
            
        # SPM 轉為 Tensor (保持 [0, 1])
        spm_image = self.transform(spm_image)
        
        # 載入 CA4D (如果啟用)
        # CA4D 格式: ca4d_{DEM}_rf{scenario}_{channel}_{timestep}.png
            # 注意: scenario 是小寫 rf (不是 RF), 且沒有 _00 後綴
        rf_scenario = f"rf{col_num:02d}"  # 小寫 rf
        time_str = f"{row_num:03d}"   # timestep (001, 002, ...)
            
            # CA4D 資料夾結構: ca4d_folder/d/{DEM}/rf{scenario}/ca4d_{DEM}_rf{scenario}_d_{timestep}.png
        path_ca4d_d = os.path.join(self.ca4d_folder, 'd', dem_str, rf_scenario, 
                                       f"ca4d_{dem_str}_{rf_scenario}_d_{time_str}.png")
        path_ca4d_vx = os.path.join(self.ca4d_folder, 'vx', dem_str, rf_scenario,
                                        f"ca4d_{dem_str}_{rf_scenario}_vx_{time_str}.png")
        path_ca4d_vy = os.path.join(self.ca4d_folder, 'vy', dem_str, rf_scenario,
                                        f"ca4d_{dem_str}_{rf_scenario}_vy_{time_str}.png")
            
        ca4d_d = cv2.imread(path_ca4d_d, cv2.IMREAD_GRAYSCALE)
        ca4d_vx = cv2.imread(path_ca4d_vx, cv2.IMREAD_GRAYSCALE)
        ca4d_vy = cv2.imread(path_ca4d_vy, cv2.IMREAD_GRAYSCALE)
            
        if any(img is None for img in [ca4d_d, ca4d_vx, ca4d_vy]):
            print(f"[WARNING] CA4D not found for {path_ca4d_d}, using zero tensor")
            ca4d_image = torch.zeros((3, 256, 256))
        else:
            # CA4D Normalize [0, 255] -> [-1, 1]
            ca4d_d = self.transform(ca4d_d)
            ca4d_vx = self.transform(ca4d_vx)
            ca4d_vy = self.transform(ca4d_vy)
            
            ca4d_d = (ca4d_d * 2) - 1
            ca4d_vx = (ca4d_vx * 2) - 1
            ca4d_vy = (ca4d_vy * 2) - 1
            
            # [3, H, W]
            ca4d_image = torch.cat([ca4d_d, ca4d_vx, ca4d_vy], dim=0)

        # 讀取 Ground Truth
        rainfall = np.array(rainfall, dtype=np.int64)

        flood_path = self.__find_flood_image(cell_position, self.flood_path)
        vx_path = self.__find_vx_image(cell_position)
        vy_path = self.__find_vy_image(cell_position)

        flood_image = cv2.imread(flood_path, cv2.IMREAD_UNCHANGED)
        vx_image = cv2.imread(vx_path, cv2.IMREAD_UNCHANGED)
        vy_image = cv2.imread(vy_path, cv2.IMREAD_UNCHANGED)
        
        if any(img is None for img in [flood_image, vx_image, vy_image]):
             raise FileNotFoundError("GT images not found")
        
        # [MODIFIED] Mask 設為 None
        # flood_binary_mask = (flood_image <= 250).astype('uint8') ...
        flood_binary_mask = None
        vx_binary_mask = None
        vy_binary_mask = None
        
        # Convert to proper dtype
        flood_image = np.array(flood_image, dtype=np.uint8)
        vx_image = np.array(vx_image, dtype=np.uint8)
        vy_image = np.array(vy_image, dtype=np.uint8)

        # Apply transforms
        dem_image = self.transform(dem_image)
        flood_image = self.transform(flood_image)
        vx_image = self.transform(vx_image)
        vy_image = self.transform(vy_image)

        # Normalize
        dem_image = (dem_image - 0.18) / 0.22
        flood_image = (flood_image - 0.986) / 0.0405
        vx_image = (vx_image - 0.562) / 0.078
        vy_image = (vy_image - 0.495) / 0.0789
        
        # Get next timestep data
        next_data = self.get_next_timestep_data(cell_position[0], cell_position[1], cell_position[2])

        return (flood_image, vx_image, vy_image, dem_image, 
                flood_binary_mask, vx_binary_mask, vy_binary_mask, 
                rainfall, flood_path, vx_path, vy_path, 
                spm_image, 
                ca4d_image, 
                next_data,  # ← 同時包含 SPM 和 CA4D
                max_depth, dem_id)

class singleDEMFloodDataset(Dataset):
    def __init__(self, opt, val=False, test=False):
        super(singleDEMFloodDataset, self).__init__()
        self.opt = opt
        self.test = test
        
        dem_path = 'C:\\Users\\THINKLAB\\Desktop\\PIFF-master02\\data\\tainan_dem.png'
        self.spm_folder = 'C:\\Users\\THINKLAB\\Desktop\\PIFF-master02\\data\\SPM_output'
        self.dem = cv2.imread(dem_path, cv2.IMREAD_UNCHANGED)[:,:,0]
        
        # [MODIFIED] CA4D Folder - 使用 ca4d 資料夾結構
        if test:
            self.ca4d_folder = 'C:\\Users\\THINKLAB\\Desktop\\PIFF-master02\\data\\ca4d\\single_test'
        else:
            # 訓練模式使用 single_train
            self.ca4d_folder = 'C:\\Users\\THINKLAB\\Desktop\\PIFF-master02\\data\\ca4d\\single_train'

        self.flood_path = 'C:\\Users\\THINKLAB\\Desktop\\PIFF-master02\\data\\train\\depth'
        self.VX_path = 'C:\\Users\\THINKLAB\\Desktop\\PIFF-master02\\data\\train\\Vx'
        self.VY_path = 'C:\\Users\\THINKLAB\\Desktop\\PIFF-master02\\data\\train\\Vy'

        rainfall_path = 'C:\\Users\\THINKLAB\\Desktop\\PIFF-master02\\data\\train_rf.csv'
        if test:
            rainfall_path = 'C:\\Users\\THINKLAB\\Desktop\\PIFF-master02\\data\\test_rf.csv'
            self.flood_path = 'C:\\Users\\THINKLAB\\Desktop\\PIFF-master02\\data\\test\\depth'
            self.VX_path = 'C:\\Users\\THINKLAB\\Desktop\\PIFF-master02\\data\\test\\Vx'
            self.VY_path = 'C:\\Users\\THINKLAB\\Desktop\\PIFF-master02\\data\\test\\Vy'

        rainfall = pd.read_csv(rainfall_path)
        rainfall = rainfall.iloc[:, :]

        rainfall_cum_value = []
        cell_positions = []
        spm_values = []
        
        val = False
        for col in rainfall.columns:
            if col == 'time':
                continue
            col_num = int(col.split("_")[1])
            if (val and col_num not in [2]) or (not val and col_num in []):
                continue
            cell_values = []
            for row in range(len(rainfall)):
                cell_value = rainfall.iloc[row][col]
                cell_values.append(np.floor(cell_value))
                temp = [0] * (24 - len(cell_values))
                temp.extend(cell_values)
                if len(temp) == 25:
                    temp = temp[1:]

                sum_rainfall = sum(temp[:])
                spm_value = int(np.ceil(sum_rainfall / 5) * 5)
                dem_num = 1
                dem_str = str(dem_num)
                rf_scenario = f"rf{col_num:02d}"
                time_str = f"{row:03d}"
                
                # 檢查 SPM
                spm_path = os.path.join(self.spm_folder, f'SPM_{dem_str}_{spm_value}.png')
                if not os.path.exists(spm_path):
                    continue
                
                # 檢查 CA4D (所有 3 通道)
                path_ca4d_d = os.path.join(self.ca4d_folder, 'd', dem_str, rf_scenario,
                                           f"ca4d_{dem_str}_{rf_scenario}_d_{time_str}.png")
                path_ca4d_vx = os.path.join(self.ca4d_folder, 'vx', dem_str, rf_scenario,
                                            f"ca4d_{dem_str}_{rf_scenario}_vx_{time_str}.png")
                path_ca4d_vy = os.path.join(self.ca4d_folder, 'vy', dem_str, rf_scenario,
                                            f"ca4d_{dem_str}_{rf_scenario}_vy_{time_str}.png")
                
                if not all([os.path.exists(p) for p in [path_ca4d_d, path_ca4d_vx, path_ca4d_vy]]):
                    continue
                
                # 檢查 Ground Truth
                if col_num < 100:
                    folder_name = f"RF{col_num:02d}"
                else:
                    folder_name = f"RF{col_num}"
                
                flood_name = f"{folder_name}_d_{row:03d}_00.png"
                vx_name = f"{folder_name}_vx_{row:03d}_00.png"
                vy_name = f"{folder_name}_vy_{row:03d}_00.png"
                
                flood_file = os.path.join(self.flood_path, folder_name, flood_name)
                vx_file = os.path.join(self.VX_path, folder_name, vx_name)
                vy_file = os.path.join(self.VY_path, folder_name, vy_name)
                
                if not all([os.path.exists(p) for p in [flood_file, vx_file, vy_file]]):
                    continue
                
                # [RESTORED] 所有檔案都存在，記錄此樣本
                spm_values.append(spm_value)
                rainfall_cum_value.append(temp)
                cell_positions.append((col_num, row))

        self.rainfall = rainfall_cum_value
        self.cell_positions = cell_positions
        self.spm_values = spm_values
        print(f"Training data length: {len(self.cell_positions)}")
        
        self.transform = T.Compose([
            T.ToTensor(),
        ])

    def __find_image(self, cell_position, flood_path):
        col, row = cell_position
        if col < 100:
            folder_name = f"RF{col:02d}"
        else:
            folder_name = f"RF{col}"
        if self.test:
            if col < 100:
                folder_name = f"RF{col:02d}"
            else:
                folder_name = f"RF{col}"
        image_name = f"{folder_name}_d_{row:03d}_00.png"
        image_path = os.path.join(flood_path, folder_name, image_name)
        return image_path
    
    def __find_Vx_image(self, cell_position, VX_path):
        col, row = cell_position
        if col < 100:
            vx_folder_name = f"RF{col:02d}"
        else:
            vx_folder_name = f"RF{col}"
        if self.test:
            if col < 100:
                vx_folder_name = f"RF{col:02d}"
            else:
                vx_folder_name = f"RF{col}"
        vx_image_image_name = f"{vx_folder_name}_vx_{row:03d}_00.png"
        vx_image_path = os.path.join(VX_path, vx_folder_name, vx_image_image_name)
        return vx_image_path

    def __find_Vy_image(self, cell_position, VY_path):
        col, row = cell_position
        if col < 100:
            vy_folder_name = f"RF{col:02d}"
        else:
            vy_folder_name = f"RF{col}"
        if self.test:
            if col < 100:
                vy_folder_name = f"RF{col:02d}"
            else:
                vy_folder_name = f"RF{col}"
        vy_image_name = f"{vy_folder_name}_vy_{row:03d}_00.png"
        vy_image_path = os.path.join(VY_path, vy_folder_name, vy_image_name)
        return vy_image_path

    def get_next_timestep_data(self, col, row):
        try:
            next_row = row + 1
            next_cell_position = (col, next_row)
            
            next_flood_path = self.__find_image(next_cell_position, self.flood_path)
            
            if not os.path.exists(next_flood_path):
                return None
                
            next_flood_image = cv2.imread(next_flood_path, cv2.IMREAD_UNCHANGED)
            
            # [MODIFIED] Mask 設為 None
            # next_binary_mask = (next_flood_image <= 250).astype('uint8') ...
            next_binary_mask = None
            
            next_flood_image = np.array(next_flood_image, dtype=np.uint8)
            
            next_flood_image = self.transform(next_flood_image)
            next_flood_image = (next_flood_image - 0.987) / 0.0343
            
            return (next_flood_image, next_binary_mask)
                   
        except Exception as e:
            return None

    def __len__(self):
        return len(self.cell_positions)

    def __getitem__(self, index):
        dem_num = 1  # singleDEM 固定為 1
        dem_str = str(dem_num)
        
        cell_position = self.cell_positions[index]
        rainfall = self.rainfall[index]
        spm_value = self.spm_values[index]  # [RESTORED] 取得 SPM 值
        rainfall = np.array(rainfall, dtype=np.int64)
        
        col_num = cell_position[0]
        row_num = cell_position[1]
        
        # ===== 讀取 DEM =====
        dem_image = self.dem
        dem_image = self.transform(dem_image)
        dem_image = (dem_image - 0.547) / 0.282
        
        # ===== 讀取 SPM (1 通道) =====
        spm_path = os.path.join(self.spm_folder, f'SPM_{dem_str}_{spm_value}.png')
        try:
            spm_image = cv2.imread(spm_path, cv2.IMREAD_GRAYSCALE)
            if spm_image is not None:
                spm_image = self.transform(spm_image)  # [0, 1]
            else:
                print(f"[WARNING] SPM image not found: {spm_path}")
                spm_image = torch.zeros((1, 256, 256))
        except Exception as e:
            print(f"[WARNING] Failed to load SPM: {spm_path}, {e}")
            spm_image = torch.zeros((1, 256, 256))
        
        # ===== 讀取 CA4D (3 通道: d, vx, vy) =====
        rf_scenario = f"rf{col_num:02d}"
        time_str = f"{row_num:03d}"
        
        path_ca4d_d = os.path.join(self.ca4d_folder, 'd', dem_str, rf_scenario,
                                   f"ca4d_{dem_str}_{rf_scenario}_d_{time_str}.png")
        path_ca4d_vx = os.path.join(self.ca4d_folder, 'vx', dem_str, rf_scenario,
                                    f"ca4d_{dem_str}_{rf_scenario}_vx_{time_str}.png")
        path_ca4d_vy = os.path.join(self.ca4d_folder, 'vy', dem_str, rf_scenario,
                                    f"ca4d_{dem_str}_{rf_scenario}_vy_{time_str}.png")
        
        ca4d_image = None
        try:
            if all([os.path.exists(p) for p in [path_ca4d_d, path_ca4d_vx, path_ca4d_vy]]):
                ca4d_d = cv2.imread(path_ca4d_d, cv2.IMREAD_GRAYSCALE)
                ca4d_vx = cv2.imread(path_ca4d_vx, cv2.IMREAD_GRAYSCALE)
                ca4d_vy = cv2.imread(path_ca4d_vy, cv2.IMREAD_GRAYSCALE)
                
                if all(img is not None for img in [ca4d_d, ca4d_vx, ca4d_vy]):
                    ca4d_d = self.transform(ca4d_d)
                    ca4d_vx = self.transform(ca4d_vx)
                    ca4d_vy = self.transform(ca4d_vy)
                    
                    # [0, 1] -> [-1, 1]
                    ca4d_image = torch.cat([
                        (ca4d_d * 2) - 1,
                        (ca4d_vx * 2) - 1,
                        (ca4d_vy * 2) - 1
                    ], dim=0)  # [3, H, W]
        except Exception as e:
            print(f"[WARNING] Failed to load CA4D: {e}")
        
        if ca4d_image is None:
            ca4d_image = torch.zeros((3, 256, 256))
        
        # ===== 讀取 Ground Truth (d, vx, vy) =====
        if col_num < 100:
            folder_name = f"RF{col_num:02d}"
        else:
            folder_name = f"RF{col_num}"
        
        image_path = self.__find_image(cell_position, self.flood_path)
        vx_image_path = self.__find_Vx_image(cell_position, self.VX_path)
        vy_image_path = self.__find_Vy_image(cell_position, self.VY_path)

        flood_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        vx_image = cv2.imread(vx_image_path, cv2.IMREAD_UNCHANGED)
        vy_image = cv2.imread(vy_image_path, cv2.IMREAD_UNCHANGED)

        # Mask 設為 None
        binary_mask = None
        vx_binary_mask = None
        vy_binary_mask = None
        
        flood_image = np.array(flood_image, dtype=np.uint8)
        vx_image = np.array(vx_image, dtype=np.uint8)
        vy_image = np.array(vy_image, dtype=np.uint8)
        
        # 轉換為 Tensor
        flood_image = self.transform(flood_image)
        vx_image = self.transform(vx_image)
        vy_image = self.transform(vy_image)

        # 標準化
        flood_image = (flood_image - 0.987) / 0.0343
        vx_image = (vx_image - 0.497) / 0.0043
        vy_image = (vy_image - 0.497) / 0.0047

        # 取得下一時間步資料
        col, row = cell_position
        next_data = self.get_next_timestep_data(col, row)
        
        # 回傳: 同時包含 SPM 和 CA4D
        max_depth = 4.0
        dem_id = 1
        
        return (flood_image, vx_image, vy_image, dem_image, 
                binary_mask, vx_binary_mask, vy_binary_mask, 
                rainfall, image_path, vx_image_path, vy_image_path, 
                spm_image,
                ca4d_image,
                next_data, max_depth, dem_id)


class yilanDataset(Dataset):
    """
    Yilan 新地形資料集 - 預測專用 (3個地形)
    
    地形結構:
      - yilan1: 中高丘陵 (高程 4-117m)
      - yilan2: 平原 (高程 0-16m)  
      - yilan3: 高山 (高程 1-230m)
    
    包含資料:
      - DEM: yilan1/2/3.png
      - 模擬數據: tuflow/{d,vx,vy}/yilan{1,2,3}/
      - CA4D 引導: yilan_ca4d/{d,vx,vy}/yilan{1,2,3}/
      - SPM 引導: yilan_spm/yilan{1,2,3}/SPM_*.png
      - 降雨: scenario_rainfall.csv (10個入流點)
      - 統計: maxmin_dem.csv, maxmin_duv.csv
    """
    
    def __init__(self, opt, test_dem_list=None):
        super(yilanDataset, self).__init__()
        self.opt = opt
        
        # 資料根路徑
        base_path = 'C:\\Users\\THINKLAB\\Desktop\\PIFF-master02\\data\\yilan'
        
        # ===== 路徑設定 =====
        self.dem_folder = os.path.join(base_path, 'dem')
        self.tuflow_d_folder = os.path.join(base_path, 'tuflow', 'd')
        self.tuflow_vx_folder = os.path.join(base_path, 'tuflow', 'vx')
        self.tuflow_vy_folder = os.path.join(base_path, 'tuflow', 'vy')
        self.ca4d_folder = os.path.join(base_path, 'yilan_ca4d')
        self.spm_folder = os.path.join(base_path, 'yilan_spm')
        
        # ===== 載入統計資訊 =====
        # 1. 高程統計 (maxmin_dem.csv)
        maxmin_dem_path = os.path.join(base_path, 'maxmin_dem.csv')
        self.dem_stats = pd.read_csv(maxmin_dem_path)
        dem_dict = dict(zip(self.dem_stats['terrain'].astype(str), 
                           zip(self.dem_stats['min_elevation'], self.dem_stats['max_elevation'])))
        
        # 2. 模擬數據統計 (maxmin_duv.csv)
        maxmin_duv_path = os.path.join(base_path, 'maxmin_duv.csv')
        duv_stats = pd.read_csv(maxmin_duv_path)
        self.terrain_depths = dict(zip(duv_stats['terrain'].astype(str), duv_stats['depth_max']))
        print(f"[yilanDataset] Loaded terrain depths: {self.terrain_depths}")
        
        # 3. 降雨情景 (scenario_rainfall.csv)
        rainfall_path = os.path.join(base_path, 'scenario_rainfall.csv')
        rainfall_df = pd.read_csv(rainfall_path)
        
        # ===== 建立資料列表 =====
        # 降雨有 10 個入流點 (inflow_1 ~ inflow_10)
        # TUFLOW 模擬每個地形有 10 個時間步
        
        self.cell_positions = []  # (dem_name, rf_scenario, timestep)
        self.rainfall_values = []  # 降雨累積值
        self.dem_info = {}  # DEM 高程範圍資訊
        
        dem_names = ['yilan1', 'yilan2', 'yilan3']
        
        for dem_name in dem_names:
            # 檢查 DEM 資料是否存在
            dem_png_path = os.path.join(self.dem_folder, f'{dem_name}.png')
            if not os.path.exists(dem_png_path):
                print(f"[WARNING] DEM not found: {dem_png_path}")
                continue
            
            # 儲存 DEM 高程資訊
            if dem_name in dem_dict:
                min_elev, max_elev = dem_dict[dem_name]
                self.dem_info[dem_name] = {'min': min_elev, 'max': max_elev}
            
            # 遍歷降雨情景 (inflow_1 ~ inflow_10)
            for col_idx, col in enumerate(rainfall_df.columns):
                if col == 'time':
                    continue
                
                # 提取入流編號 (1-10)
                try:
                    flow_num = int(col.split('_')[1])
                except:
                    continue
                
                # 遍歷時間步 (0-24, 共 25 步 = 25 小時)
                for row_idx in range(25):
                    self.cell_positions.append((dem_name, flow_num, row_idx))
                    
                    # 累積降雨值
                    rainfall_val = rainfall_df.iloc[row_idx][col]
                    self.rainfall_values.append(rainfall_val)
        
        print(f"[yilanDataset] Initialized with {len(self.cell_positions)} samples")
        print(f"              Terrain: yilan1, yilan2, yilan3")
        print(f"              Scenarios: 10 inflow points × 25 timesteps (0-24 hours)")
        
        self.transform = T.Compose([
            T.ToTensor(),
        ])
    
    def __find_tuflow_image(self, dem_name, flow_num, timestep, variable):
        """
        找到 TUFLOW 模擬檔案
        variable: 'd' (depth), 'vx', 'vy'
        
        路徑格式: tuflow/{var}/yilan{i}/RF{flow:02d}/{dem_name}_RF{flow:02d}_{var}_{timestep:03d}_00.png
        """
        if variable == 'd':
            base_folder = self.tuflow_d_folder
        elif variable == 'vx':
            base_folder = self.tuflow_vx_folder
        elif variable == 'vy':
            base_folder = self.tuflow_vy_folder
        else:
            raise ValueError(f"Unknown variable: {variable}")
        
        # 構造路徑: tuflow/{var}/yilan{i}/RF{flow:02d}/
        dem_folder = os.path.join(base_folder, dem_name)
        rf_folder = os.path.join(dem_folder, f"RF{flow_num:02d}")
        
        # 檔案名: yilan1_RF01_d_000_00.png
        filename = f"{dem_name}_RF{flow_num:02d}_{variable}_{timestep:03d}_00.png"
        filepath = os.path.join(rf_folder, filename)
        
        return filepath
    
    def __find_dem_image(self, dem_name):
        """找到 DEM PNG 檔案"""
        dem_path = os.path.join(self.dem_folder, f'{dem_name}.png')
        return dem_path
    
    def __find_ca4d_images(self, dem_name, flow_num, timestep):
        """
        找到 CA4D 引導檔案 (3通道: d, vx, vy)
        
        路徑格式: yilan_ca4d/{var}/yilan{i}/ca4d_{dem_name}_{flow_num:02d}_{var}_{timestep:03d}.png
        """
        ca4d_images = {}
        
        for var in ['d', 'vx', 'vy']:
            var_folder = os.path.join(self.ca4d_folder, var, dem_name)
            # 檔案名: ca4d_yilan1_rf01_d_001.png
           
            filename = f"ca4d_{dem_name}_rf{flow_num:02d}_{var}_{timestep:03d}.png"
            rf_dirname = f"rf{flow_num:02d}"
            filepath = os.path.join(var_folder, rf_dirname, filename)
            ca4d_images[var] = filepath
        
        return ca4d_images
    
    def __find_spm_image(self, dem_name, spm_index):
        """
        找到 SPM (空間先驗地圖) 檔案
        
        路徑格式: yilan_spm/yilan{i}/SPM_yilan{i}_{spm_index}.png
        spm_index 範圍: 0-720 (時間序列編號)
        """
        spm_folder = os.path.join(self.spm_folder, dem_name)
        filename = f"SPM_{dem_name}_{spm_index}.png"
        filepath = os.path.join(spm_folder, filename)
        
        return filepath
    
    def __len__(self):
        return len(self.cell_positions)
    
    def __getitem__(self, index):
        dem_name, flow_num, timestep = self.cell_positions[index]
        rainfall_val = self.rainfall_values[index]
        
        # ===== 讀取 DEM =====
        dem_path = self.__find_dem_image(dem_name)
        dem_image = cv2.imread(dem_path, cv2.IMREAD_UNCHANGED)
        
        if dem_image is None:
            raise FileNotFoundError(f"Failed to load DEM: {dem_path}")
        
        # DEM 轉為灰度
        if len(dem_image.shape) == 3:
            dem_image = dem_image[:, :, 0]
        
        # DEM 高程標準化
        if dem_name in self.dem_info:
            min_elev = self.dem_info[dem_name]['min']
            max_elev = self.dem_info[dem_name]['max']
            elev_range = max_elev - min_elev if max_elev != min_elev else 1
            # [0, 255] -> [min_elev, max_elev] -> [-1, 1]
            real_height = dem_image / 255.0 * elev_range + min_elev
            dem_normalized = (real_height - (-3)) / (125 + 3) * 255
            dem_image = np.clip(dem_normalized, 0, 255).astype(np.uint8)
            dem_image = np.array(dem_image, dtype=np.uint8)
        
        # ===== 讀取 TUFLOW 真值 (d, vx, vy) =====
        d_path = self.__find_tuflow_image(dem_name, flow_num, timestep, 'd')
        vx_path = self.__find_tuflow_image(dem_name, flow_num, timestep, 'vx')
        vy_path = self.__find_tuflow_image(dem_name, flow_num, timestep, 'vy')
        
        flood_image = cv2.imread(d_path, cv2.IMREAD_UNCHANGED)
        vx_image = cv2.imread(vx_path, cv2.IMREAD_UNCHANGED)
        vy_image = cv2.imread(vy_path, cv2.IMREAD_UNCHANGED)
        
        if any(img is None for img in [flood_image, vx_image, vy_image]):
            raise FileNotFoundError(f"Failed to load TUFLOW images:\n  d: {d_path}\n  vx: {vx_path}\n  vy: {vy_path}")
        
        flood_image = np.array(flood_image, dtype=np.uint8)
        vx_image = np.array(vx_image, dtype=np.uint8)
        vy_image = np.array(vy_image, dtype=np.uint8)
        
        # ===== 讀取 CA4D 引導 (3 通道: d, vx, vy) =====
        ca4d_paths = self.__find_ca4d_images(dem_name, flow_num, timestep)
        ca4d_image = None
        
        try:
            if all(os.path.exists(p) for p in ca4d_paths.values()):
                ca4d_d = cv2.imread(ca4d_paths['d'], cv2.IMREAD_GRAYSCALE)
                ca4d_vx = cv2.imread(ca4d_paths['vx'], cv2.IMREAD_GRAYSCALE)
                ca4d_vy = cv2.imread(ca4d_paths['vy'], cv2.IMREAD_GRAYSCALE)
                
                if all(img is not None for img in [ca4d_d, ca4d_vx, ca4d_vy]):
                    ca4d_d = self.transform(ca4d_d)
                    ca4d_vx = self.transform(ca4d_vx)
                    ca4d_vy = self.transform(ca4d_vy)
                    
                    # [0, 1] -> [-1, 1]
                    ca4d_image = torch.cat([
                        (ca4d_d * 2) - 1,
                        (ca4d_vx * 2) - 1,
                        (ca4d_vy * 2) - 1
                    ], dim=0)  # [3, H, W]
        except Exception as e:
            print(f"[WARNING] Failed to load CA4D for {dem_name}_RF{flow_num:02d}_{timestep:03d}: {e}")
        
        if ca4d_image is None:
            print(f"[WARNING] CA4D not found for {ca4d_paths}, using zero tensor")
            ca4d_image = torch.zeros((3, 256, 256))
        
        # ===== 讀取 SPM 引導 (1 通道) =====
        # SPM 編號對應時間步: 0-720 (總共 145 個)
        spm_index = timestep * 72  # 簡單映射: timestep * 72 (0, 72, 144, ...)
        spm_path = self.__find_spm_image(dem_name, spm_index)
        spm_image = None
        
        try:
            if os.path.exists(spm_path):
                spm_image = cv2.imread(spm_path, cv2.IMREAD_GRAYSCALE)
                if spm_image is not None:
                    spm_image = self.transform(spm_image)  # [0, 1]
        except Exception as e:
            print(f"[WARNING] Failed to load SPM: {spm_path}, {e}")
        
        if spm_image is None:
            spm_image = torch.zeros((1, 256, 256))
        
        # ===== 影像標準化 =====
        dem_image = self.transform(dem_image)
        flood_image = self.transform(flood_image)
        vx_image = self.transform(vx_image)
        vy_image = self.transform(vy_image)
        
        # 標準化 (使用預設統計值)
        dem_image = (dem_image - 0.1623) / 0.1836
        flood_image = (flood_image - 0.9880) / 0.0279
        vx_image = (vx_image - 0.3571) / 0.0999
        vy_image = (vy_image - 	0.4413) / 0.0846
        
        # ===== 取得下一時間步資料 =====
        next_data = None
        if timestep < 24:  # 最多 25 步 (0-24)
            try:
                next_d_path = self.__find_tuflow_image(dem_name, flow_num, timestep + 1, 'd')
                if os.path.exists(next_d_path):
                    next_flood = cv2.imread(next_d_path, cv2.IMREAD_UNCHANGED).astype(np.uint8)
                    next_flood = self.transform(next_flood)
                    # 使用 Yilan 統計參數標準化
                    next_flood = (next_flood - 0.9880) / 0.0279
                    next_data = (next_flood, None)
            except:
                pass
        
        # ===== 準備回傳值 =====
        max_depth = self.terrain_depths.get(dem_name, 4.0)
        
        # 記錄降雨值
        rainfall_array = np.array([rainfall_val] * 24, dtype=np.int64)
        
        # 將地形名稱轉換為整數 ID (yilan1->1, yilan2->2, yilan3->3)
        dem_id = int(dem_name.replace('yilan', ''))
        
        return (flood_image, vx_image, vy_image, dem_image,
                None, None, None,  # binary masks (都為 None)
                rainfall_array, d_path, vx_path, vy_path,
                spm_image,
                ca4d_image, 
                next_data,
                max_depth, dem_id)