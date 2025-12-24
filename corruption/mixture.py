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

from pathlib import Path
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
        self.spm_folder = 'C:\\Users\\THINKLAB\\Desktop\\PIFF-master02\\data\\SPM_71dem_output'
        # list all the subfolder in the dem_path, for example subfolder is 1, 2, 3 then return me [1, 2, 3]

        self.dem_stat = 'C:\\Users\\THINKLAB\\Desktop\\PIFF-master02\\data\\dems\\dem_png\\elevation_stats.csv'
        self.dem_stat = pd.read_csv(self.dem_stat)
        # self.dem = cv2.imread(dem_path)
        # self.dem = cv2.cvtColor(self.dem, cv2.COLOR_BGR2GRAY)
        # self.dem = cv2.cvtColor(self.dem, cv2.COLOR_GRAY2BGR)
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
            #dem_folder = [8, 29, 62]
            #self.flood_path = 'C:\\Users\\THINKLAB\\Desktop\\PIFF-master02\\data\\dems\\train\\d'
            #self.vx = "C:\\Users\\THINKLAB\\Desktop\\PIFF-master02\\data\\dems\\train\\vx"
            #self.vy = "C:\\Users\\THINKLAB\\Desktop\\PIFF-master02\\data\\dems\\train\\vy"

        rainfall = pd.read_csv(rainfall_path)
        rainfall = rainfall.iloc[:, :]

        # Initialize lists to store cell values and their positions
        rainfall_cum_value = []
        cell_positions = []
        spm = []

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
                for row in range(len(rainfall)):
                    cell_value = rainfall.iloc[row][col]
                    cell_values.append(np.ceil(cell_value))
                    # make it a len 24 list if not append 0 in front
                    temp = [0] * (24 - len(cell_values))
                    temp.extend(cell_values)
                    if len(temp) == 25:
                        temp = temp[1:]
                    sum_rainfall = sum(temp[:])
                    # ceil the number to the nearest 5 multiple
                    # if all the value is 0, then skip
                    if not test and sum_rainfall <= 5:
                        if np.random.rand() < 0.8:
                            continue
                    
                    # 檢查必要的檔案是否存在
                    spm_value = int(np.ceil(sum_rainfall / 5) * 5)
                    
                    # 構建檔案路徑進行檢查
                    dem_folder_name = str(dem_num)
                    if col_num < 100:
                        folder_name = f"RF{col_num:02d}"
                    else:
                        folder_name = f"RF{col_num}"
                    
                    # 檢查 SPM 圖像 (路徑: SPM_71dem_output/{dem}/SPM_{dem}_{rainfall}.png)
                    spm_path = os.path.join(self.spm_folder, str(dem_num), f'SPM_{dem_num}_{spm_value}.png')
                    if not os.path.exists(spm_path):
                        continue  # 跳過沒有 SPM 的資料
                    
                    # 檢查淹水/流速圖像 (使用正確的檔案名稱格式)
                    flood_name = f"{dem_folder_name}_{folder_name}_d_{row:03d}_00.png"
                    vx_name = f"{dem_folder_name}_{folder_name}_vx_{row:03d}_00.png"
                    vy_name = f"{dem_folder_name}_{folder_name}_vy_{row:03d}_00.png"
                    
                    flood_path = os.path.join(self.flood_path, dem_folder_name, folder_name, flood_name)
                    vx_path = os.path.join(self.vx_path, dem_folder_name, folder_name, vx_name)
                    vy_path = os.path.join(self.vy_path, dem_folder_name, folder_name, vy_name)
                    
                    if not all([os.path.exists(p) for p in [flood_path, vx_path, vy_path]]):
                        continue  # 跳過缺失檔案的資料
                    
                    # 全部檢查通過,加入資料集
                    spm.append(spm_value)
                    rainfall_cum_value.append(temp[:])
                    # col_num is the rainfall index, and row is the time index
                    cell_positions.append((dem_num, col_num, row))   

        self.rainfall = rainfall_cum_value
        self.cell_positions = cell_positions
        self.spm = spm
        
        # 打印訓練資料統計
        total_scenarios = len(dem_folder) * len([c for c in rainfall.columns if c != 'time']) * len(rainfall)
        print(f"[OK] Dataset initialized: {len(self.cell_positions)} valid samples")
        print(f"     (Filtered {total_scenarios - len(self.cell_positions)} samples with missing files)")
        
        # add transform , to tensor and normalize
        self.transform = T.Compose([
            T.ToTensor(),
            # T.Lambda(lambda t: (t * 2) - 1)
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
        # 直接使用固定路徑,不依賴 opt.dataset_dir
        dem_folder = Path('C:\\Users\\THINKLAB\\Desktop\\PIFF-master02\\data\\dems\\dem_png')
        dem_path = dem_folder / f'{dem_num}.png'
        return str(dem_path)
    def get_next_timestep_data(self, dem, col, row):
        """
        獲取下一個時間步的水深、Vx、Vy 資料 (row+1)
        回傳: (next_flood_image, next_vx_image, next_vy_image, next_binary_mask) 或 None
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
            
            # 處理 mask
            next_binary_mask = (next_flood_image <= 250).astype('uint8')
            next_binary_mask = np.expand_dims(next_binary_mask, axis=0)
            
            next_vx_binary_mask = (next_vx_image != 125).astype('uint8')
            next_vx_binary_mask = np.expand_dims(next_vx_binary_mask, axis=0)
            
            next_vy_binary_mask = (next_vy_image != 125).astype('uint8')
            next_vy_binary_mask = np.expand_dims(next_vy_binary_mask, axis=0)
            
            # 轉換為 uint8
            next_flood_image = np.array(next_flood_image, dtype=np.uint8)
            next_vx_image = np.array(next_vx_image, dtype=np.uint8)
            next_vy_image = np.array(next_vy_image, dtype=np.uint8)
            
            # 轉換為 tensor 並標準化 (與 __getitem__ 相同)
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
        spm = self.spm[index]
        dem_path = self.__find_dem_image(cell_position)
        
        # 讀取 DEM 並檢查是否成功
        dem_image = cv2.imread(dem_path, cv2.IMREAD_UNCHANGED)
        if dem_image is None:
            raise FileNotFoundError(f"Failed to load DEM image: {dem_path}")
        
        # 確保是 3 通道圖像,取第一個通道
        if len(dem_image.shape) == 3:
            dem_image = dem_image[:,:,0]
        
        # dem_image = cv2.cvtColor(dem_image, cv2.COLOR_GRAY2BGR)
        dem_cur_state = self.dem_stat[self.dem_stat['Filename'] == cell_position[0]]
        min_elev = int(dem_cur_state['Min Elevation'].iloc[0])
        max_elev = int(dem_cur_state['Max Elevation'].iloc[0])
        # do a normalization with max = 410 min = -3, with current max = max_elev, min = min_elev
        real_height = dem_image / 255 * (max_elev - min_elev) + min_elev
        dem_image = (real_height - (-3)) / (125 + 3) * 255
        # clamp dem_image to 0-255
        dem_image = np.clip(dem_image, 0, 255)
        dem_image = np.array(dem_image, dtype=np.uint8)
        
        # SPM 路徑: SPM_71dem_output/{dem}/SPM_{dem}_{rainfall}.png
        spm_path = os.path.join(self.spm_folder, str(cell_position[0]), f'SPM_{cell_position[0]}_{spm}.png')
        spm_image = cv2.imread(spm_path, cv2.IMREAD_GRAYSCALE)
        
        # 檢查 SPM 圖像是否存在
        if spm_image is None:
            raise FileNotFoundError(f"SPM image not found: {spm_path}")
        
        # dem_image -= dem_image.min()
        rainfall = np.array(rainfall, dtype=np.int64)
        # rainfall = rainfall.reshape(1, 24)

        flood_path = self.__find_flood_image(cell_position, self.flood_path)
        vx_path = self.__find_vx_image(cell_position)
        vy_path = self.__find_vy_image(cell_position)

        # Read flood image
        flood_image = cv2.imread(flood_path, cv2.IMREAD_UNCHANGED)
        
        # 檢查檔案是否存在
        if flood_image is None:
            raise FileNotFoundError(f"Flood image not found: {flood_path}")
        
        # Read vx and vy images
        vx_image = cv2.imread(vx_path, cv2.IMREAD_UNCHANGED)
        vy_image = cv2.imread(vy_path, cv2.IMREAD_UNCHANGED)
        
        # 檢查 vx/vy 圖像是否存在
        if vx_image is None:
            raise FileNotFoundError(f"Vx image not found: {vx_path}")
        if vy_image is None:
            raise FileNotFoundError(f"Vy image not found: {vy_path}")
        flood_image = cv2.imread(flood_path, cv2.IMREAD_UNCHANGED)
        
        # Read vx and vy images
        vx_image = cv2.imread(vx_path, cv2.IMREAD_UNCHANGED)
        vy_image = cv2.imread(vy_path, cv2.IMREAD_UNCHANGED)
        
        # Create binary masks
        flood_binary_mask = (flood_image <= 250).astype('uint8')
        flood_binary_mask = np.expand_dims(flood_binary_mask, axis=0)
        
        vx_binary_mask = (vx_image != 125).astype('uint8')
        vx_binary_mask = np.expand_dims(vx_binary_mask, axis=0)
        
        vy_binary_mask = (vy_image != 125).astype('uint8')
        vy_binary_mask = np.expand_dims(vy_binary_mask, axis=0)
        
        # Convert to proper dtype
        flood_image = np.array(flood_image, dtype=np.uint8)
        vx_image = np.array(vx_image, dtype=np.uint8)
        vy_image = np.array(vy_image, dtype=np.uint8)

        # Apply transforms
        dem_image = self.transform(dem_image)
        flood_image = self.transform(flood_image)
        vx_image = self.transform(vx_image)
        vy_image = self.transform(vy_image)
        spm_image = self.transform(spm_image)  # Convert SPM to tensor

        # Normalize
        dem_image = (dem_image - 0.18) / 0.22
        flood_image = (flood_image - 0.98) / 0.039
        vx_image = (vx_image - 0.552) / 0.093
        vy_image = (vy_image - 0.489) / 0.0818
        
        # Get next timestep data
        next_data = self.get_next_timestep_data(cell_position[0], cell_position[1], cell_position[2])

        return (flood_image, vx_image, vy_image, dem_image, 
                flood_binary_mask, vx_binary_mask, vy_binary_mask, 
                rainfall, flood_path, vx_path, vy_path, spm_image, next_data)

class singleDEMFloodDataset(Dataset):
    def __init__(self, opt, val=False, test=False):
        super(singleDEMFloodDataset, self).__init__()
        self.opt = opt
        dem_path = 'C:\\Users\\THINKLAB\\Desktop\\PIFF-master02\\data\\tainan_dem.png'
        self.spm_folder = 'C:\\Users\\THINKLAB\\Desktop\\PIFF-master02\\data\\SPM_output'
        self.dem = cv2.imread(dem_path, cv2.IMREAD_UNCHANGED)[:,:,0]
        self.test = test

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
        # remove first row, no 0 row 
        rainfall = rainfall.iloc[:, :]

        # Initialize lists to store cell values and their positions
        rainfall_cum_value = []
        cell_positions = []
        spm = []
        val = False
        # Iterate through each column
        for col in rainfall.columns:
            if col == 'time':
                continue
            col_num = int(col.split("_")[1])
            if (val and col_num not in [2]) or (not val and col_num in []):
                continue
            cell_values = []
            # Iterate through each row in the current column
            for row in range(len(rainfall)):
                cell_value = rainfall.iloc[row][col]
                cell_values.append(np.floor(cell_value))
                # make it a len 24 list if not append 0 in front
                temp = [0] * (24 - len(cell_values))
                temp.extend(cell_values)
                if len(temp) == 25:
                    temp = temp[1:]
                sum_rainfall = sum(temp[:])
                spm.append(int(np.ceil(sum_rainfall / 5) * 5))
                rainfall_cum_value.append(temp)
                cell_positions.append((col_num, row))

        self.rainfall = rainfall_cum_value
        self.cell_positions = cell_positions
        self.spm = spm
        print(f"Training data length: {len(self.cell_positions)}")
        # add transform , to tensor and normalize
        self.transform = T.Compose([
            T.ToTensor(),
            # T.Lambda(lambda t: (t * 2) - 1)
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
        """
        獲取下一個時間步的水深資料 (row+1)
        只需要水深用於計算 ∂h/∂t，不需要流速
        回傳: (next_flood_image, next_binary_mask) 或 None
        """
        try:
            next_row = row + 1
            next_cell_position = (col, next_row)
            
            # 只檢查水深檔案是否存在
            next_flood_path = self.__find_image(next_cell_position, self.flood_path)
            
            if not os.path.exists(next_flood_path):
                return None
                
            # 只讀取水深影像
            next_flood_image = cv2.imread(next_flood_path, cv2.IMREAD_UNCHANGED)
            
            # 處理 mask (與原 __getitem__ 相同邏輯)
            next_binary_mask = (next_flood_image <= 250).astype('uint8')
            next_binary_mask = np.expand_dims(next_binary_mask, axis=0)
            next_flood_image = np.array(next_flood_image, dtype=np.uint8)
            
            # 轉換為 tensor 並標準化 (與原 __getitem__ 相同)
            next_flood_image = self.transform(next_flood_image)
            next_flood_image = (next_flood_image - 0.987) / 0.0343
            
            return (next_flood_image, next_binary_mask)
                   
        except Exception as e:
            return None

    def __len__(self):
        return len(self.cell_positions)

    def __getitem__(self, index):
        dem_image = self.dem
        rainfall = self.rainfall[index]
        spm = self.spm[index]
        rainfall = np.array(rainfall, dtype=np.int64)
        # rainfall = rainfall.reshape(1, 24)
        spm_path = os.path.join(self.spm_folder, f'SPM_1_{spm}.png')
        spm_image = cv2.imread(spm_path, cv2.IMREAD_GRAYSCALE)
        cell_position = self.cell_positions[index]

        image_path = self.__find_image(cell_position, self.flood_path)
        vx_image_path = self.__find_Vx_image(cell_position, self.VX_path)
        vy_image_path = self.__find_Vy_image(cell_position, self.VY_path)

        flood_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        vx_image = cv2.imread(vx_image_path, cv2.IMREAD_UNCHANGED)
        vy_image = cv2.imread(vy_image_path, cv2.IMREAD_UNCHANGED)

        binary_mask = (flood_image <= 250).astype('uint8')
        binary_mask = np.expand_dims(binary_mask, axis=0)
        flood_image = np.array(flood_image, dtype=np.uint8)

        vx_binary_mask = (vx_image != 125).astype('uint8')
        vx_binary_mask = np.expand_dims(vx_binary_mask, axis=0)
        vx_image = np.array(vx_image, dtype=np.uint8)

        vy_binary_mask = (vy_image != 125).astype('uint8')
        vy_binary_mask = np.expand_dims(vy_binary_mask, axis=0)
        vy_image = np.array(vy_image, dtype=np.uint8)
        
        # 取得下一時間步資料 (用於物理損失計算)
        col, row = cell_position
        next_data = self.get_next_timestep_data(col, row)
        
        dem_image = self.transform(dem_image)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
        flood_image = self.transform(flood_image)
        spm_image = self.transform(spm_image)
        vx_image = self.transform(vx_image)
        vy_image = self.transform(vy_image)

        dem_image = (dem_image - 0.547) / 0.282
        flood_image = (flood_image - 0.987) / 0.0343
        vx_image = (vx_image - 0.497) / 0.0043
        vy_image = (vy_image - 0.497) / 0.0047

        #dem_image = (dem_image *2) - 1
        #flood_image = (flood_image *2) - 1
        #spm_image = (spm_image *2) - 1
        #vx_image = (vx_image *2) - 1
        #vy_image = (vy_image *2) - 1

        return flood_image, vx_image, vy_image, dem_image, binary_mask, vx_binary_mask, vy_binary_mask, rainfall, image_path, vx_image_path, vy_image_path, spm_image, next_data