import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class PhysicsInformedLoss(nn.Module):
    """
    質量守恆方程 (連續性方程):
      ∂h/∂t + ∂(hu)/∂x + ∂(hv)/∂y = R
    
    轉換流程:
      標準化值 -> [0,1] 像素值 -> 物理單位 (米, 米/秒)
    
    全局轉換係數 (從 maxmin_duv.csv 計算):
      - 流速: ±125 像素 = ±8.30 m/s (全局最大流速)
      - 深度: 動態 max_depth 參數 (每個 DEM 不同, 範圍 1.58-8.77m)
    """
    def __init__(self, dx: float = 20.0, dy: float = 20.0, dt: float = 3600.0,
                 h_mean: float = 0.986, h_std: float = 0.0405,
                 u_mean: float = 0.561, u_std: float = 0.078,
                 v_mean: float = 0.495, v_std: float = 0.0789,
                 pixel_to_mps: float = 0.066369):  # 全局流速轉換係數
        super().__init__()
        self.dx = dx
        self.dy = dy  
        self.dt = dt
        
        # 標準化參數 (用於反標準化)
        self.register_buffer("h_mean", torch.tensor(h_mean))
        self.register_buffer("h_std", torch.tensor(h_std))
        self.register_buffer("u_mean", torch.tensor(u_mean))
        self.register_buffer("u_std", torch.tensor(u_std))
        self.register_buffer("v_mean", torch.tensor(v_mean))
        self.register_buffer("v_std", torch.tensor(v_std))
        
        # 全局流速轉換係數 (從 CSV 計算: max_velocity / 125)
        self.pixel_to_mps = pixel_to_mps
        
        # 空間導數卷積核
        kx = torch.tensor([[[[0.5, 0.0, -0.5]]]], dtype=torch.float32) / dx
        ky = torch.tensor([[[[0.5], [0.0], [-0.5]]]], dtype=torch.float32) / dy
        self.register_buffer("kernel_dx", kx)
        self.register_buffer("kernel_dy", ky)

    def _denormalize(self, tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """反標準化: normalized -> [0, 1] pixel value"""
        return tensor * std + mean
    
    def _pixel_to_physical(self, h_pixel: torch.Tensor, vx_pixel: torch.Tensor, vy_pixel: torch.Tensor, 
                          max_depth: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        將像素值轉換為物理單位 (簡化版本)
        
        參數:
          h_pixel: [0, 1] 水深像素值
          vx_pixel, vy_pixel: [0, 1] 流速像素值
          max_depth: 每個樣本的最大深度 (米), 預設 4.0m
        
        轉換邏輯:
          1. 水深 (反向編碼): h_meter = (1 - h_pixel) * max_depth
             像素 1.0 = 0m, 像素 0.0 = max_depth
          
          2. 流速 (中性點 125): v_mps = (v_pixel * 255 - 125) * pixel_to_mps
             像素 125/255 ≈ 0.49 = 0 m/s
             像素 0 = -8.30 m/s, 像素 255 = +8.30 m/s
        """
        # 水深轉換 (預設 4.0m)
        if max_depth is None:
            max_depth = 4.0
        
        # 確保 max_depth 形狀正確
        if isinstance(max_depth, torch.Tensor):
            if max_depth.dim() == 1:
                max_depth = max_depth.view(-1, 1, 1, 1)
            max_depth = max_depth.to(h_pixel.device, h_pixel.dtype)
        
        # 反向編碼: 像素↑ → 深度↓
        h_meter = (1.0 - h_pixel) * max_depth
        
        # 流速轉換 (全局係數)
        vx_mps = (vx_pixel * 255.0 - 125.0) * self.pixel_to_mps
        vy_mps = (vy_pixel * 255.0 - 125.0) * self.pixel_to_mps
        
        return h_meter, vx_mps, vy_mps

    def _d(self, f: torch.Tensor, direction: str) -> torch.Tensor:
        """空間導數 (中心差分)"""
        if direction == 'x':
            return F.conv2d(f, self.kernel_dx, padding='same')
        else:
            return F.conv2d(f, self.kernel_dy, padding='same')

    def forward(
        self,
        h_t: torch.Tensor,        # (N,1,H,W) 時間 t 的水深 (標準化)
        u_t: torch.Tensor,        # (N,1,H,W) 時間 t 的 x 方向流速 (標準化)
        v_t: torch.Tensor,        # (N,1,H,W) 時間 t 的 y 方向流速 (標準化)
        h_t1: torch.Tensor,       # (N,1,H,W) 時間 t+1 的水深 (標準化)
        rainfall_t: torch.Tensor, # (N,) 或 (N,1,1,1) 降雨強度 (mm/hr)
        mask: Optional[torch.Tensor] = None,       # (N,1,H,W) 有效區域遮罩
        max_depth: Optional[torch.Tensor] = None,  # (N,) 每個樣本的最大深度 (米)
    ) -> Tuple[torch.Tensor, dict]:
        """
        計算質量守恆方程殘差: ∂h/∂t + ∂(hu)/∂x + ∂(hv)/∂y = R
        
        轉換流程: 標準化值 -> [0,1] 像素值 -> 物理單位 (米, 米/秒)
        """
        eps = 1e-12
        
        # 步驟 1: 反標準化 (標準化值 -> [0, 1] 像素值)
        h_t_pixel = self._denormalize(h_t, self.h_mean, self.h_std)
        u_t_pixel = self._denormalize(u_t, self.u_mean, self.u_std)
        v_t_pixel = self._denormalize(v_t, self.v_mean, self.v_std)
        h_t1_pixel = self._denormalize(h_t1, self.h_mean, self.h_std)
        
        # 步驟 2: 轉換為物理單位 (像素值 -> 米, 米/秒)
        h_t_m, u_t_mps, v_t_mps = self._pixel_to_physical(h_t_pixel, u_t_pixel, v_t_pixel, max_depth)
        h_t1_m, _, _ = self._pixel_to_physical(h_t1_pixel, u_t_pixel, v_t_pixel, max_depth)
        
        # Clamp 避免負值
        h_t_m = torch.clamp(h_t_m, min=eps)
        h_t1_m = torch.clamp(h_t1_m, min=eps)
        
        # 遮罩處理
        if mask is None:
            m = torch.ones_like(h_t_m)
        else:
            m = mask.to(dtype=h_t_m.dtype, device=h_t_m.device)

        # 步驟 3: 計算物理方程 (全部使用物理單位)
        # 時間導數: ∂h/∂t ≈ (h_{t+1} - h_t) / dt (米/秒)
        dhdt = (h_t1_m - h_t_m) / self.dt

        # 流量 hu, hv (米²/秒)
        hu_t = h_t_m * u_t_mps
        hv_t = h_t_m * v_t_mps

        # 流量散度: ∂(hu)/∂x + ∂(hv)/∂y (米/秒)
        dhu_dx = self._d(hu_t, 'x')
        dhv_dy = self._d(hv_t, 'y')

        # 降雨項: mm/hr -> m/s
        if rainfall_t.dim() == 1:
            rainfall_t = rainfall_t.view(-1, 1, 1, 1)
        elif rainfall_t.dim() == 0:
            rainfall_t = rainfall_t.view(1, 1, 1, 1)
        
        R = rainfall_t / 3600000.0  # mm/hr -> m/s

        # 質量守恆殘差 (米/秒)
        residual = dhdt + dhu_dx + dhv_dy - R

        # 套用遮罩
        residual = residual * m

        # 計算損失
        if m.sum() > 0:
            loss = (residual.abs() * m).sum() / m.sum().clamp(min=eps)
        else:
            loss = residual.abs().mean()

        # 診斷資訊
        summary = {
            "mass_conservation_loss": loss.detach(),
            "mean_dhdt": dhdt.mean().detach(),
            "mean_dhu_dx": dhu_dx.mean().detach(), 
            "mean_dhv_dy": dhv_dy.mean().detach(),
            "mean_R": R.mean().detach(),
            "mean_residual": residual.abs().mean().detach(),
        }
        
        return loss, summary