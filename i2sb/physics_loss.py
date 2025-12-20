import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple

class PhysicsInformedLoss(nn.Module):
    """
    質量守恆方程 (連續性方程):
      ∂h/∂t + ∂(hu)/∂x + ∂(hv)/∂y = R
    其中:
      ∂h/∂t ≈ (h_{t+1} - h_t) / dt
      ∂(hu)/∂x, ∂(hv)/∂y: 使用時間 t 的流量計算散度
      R: 降雨強度 (mm/hr -> m/s)
      dt = 1 小時 = 3600 秒
    """
    def __init__(self, dx: float = 20.0, dy: float = 20.0, dt: float = 3600.0):
        super().__init__()
        self.dx = dx
        self.dy = dy  
        self.dt = dt  # 時間間隔 (秒)
        kx = torch.tensor([[[[0.5, 0.0, -0.5]]]], dtype=torch.float32) / dx
        ky = torch.tensor([[[[0.5], [0.0], [-0.5]]]], dtype=torch.float32) / dy
        self.register_buffer("kernel_dx", kx)
        self.register_buffer("kernel_dy", ky)

    def _d(self, f: torch.Tensor, direction: str) -> torch.Tensor:
        if direction == 'x':
            return F.conv2d(f, self.kernel_dx, padding='same')
        else:
            return F.conv2d(f, self.kernel_dy, padding='same')

    def forward(
        self,
        h_t: torch.Tensor,        # (N,1,H,W) 時間 t 的水深
        u_t: torch.Tensor,        # (N,1,H,W) 時間 t 的 x 方向流速
        v_t: torch.Tensor,        # (N,1,H,W) 時間 t 的 y 方向流速
        h_t1: torch.Tensor,       # (N,1,H,W) 時間 t+1 的水深
        rainfall_t: torch.Tensor, # (N,) 或 (N,1,1,1) 當前時間步降雨強度 (mm/hr)
        mask: Optional[torch.Tensor] = None,  # (N,1,H,W) 有效區域遮罩
    ) -> Tuple[torch.Tensor, dict]:
        """
        計算質量守恆方程殘差:
          ∂h/∂t + ∂(hu)/∂x + ∂(hv)/∂y = R
        
        注意：只需要 h_{t+1}，不需要 u_{t+1}, v_{t+1}
        因為散度項 ∂(hu)/∂x + ∂(hv)/∂y 是在時間 t 計算
        """
        eps = 1e-12
        h_t = torch.clamp(h_t, min=eps)
        
        if mask is None:
            m = torch.ones_like(h_t)
        else:
            m = mask.to(dtype=h_t.dtype, device=h_t.device)

        # 時間導數: ∂h/∂t ≈ (h_{t+1} - h_t) / dt (m/s)
        dhdt = (h_t1 - h_t) / self.dt

        # 流量 hu, hv 在時間 t
        hu_t = h_t * u_t
        hv_t = h_t * v_t

        # 流量散度: ∂(hu)/∂x + ∂(hv)/∂y (m/s)
        dhu_dx = self._d(hu_t, 'x')
        dhv_dy = self._d(hv_t, 'y')

        # 降雨項: mm/hr -> m/s
        # rainfall_t 形狀處理
        if rainfall_t.dim() == 1:
            rainfall_t = rainfall_t.view(-1, 1, 1, 1)  # (N,1,1,1)
        elif rainfall_t.dim() == 0:
            rainfall_t = rainfall_t.view(1, 1, 1, 1)
        
        R = rainfall_t / 3600000.0  # mm/hr -> m/s

        # 質量守恆殘差
        residual = dhdt + dhu_dx + dhv_dy - R

        # 套用遮罩
        residual = residual * m

        # 計算損失 (使用平均)
        if m.sum() > 0:
            loss = (residual.abs() * m).sum() / m.sum().clamp(min=eps)
        else:
            loss = residual.abs().mean()

        summary = {
            "mass_conservation_loss": loss.detach(),
            "mean_dhdt": dhdt.mean().detach(),
            "mean_dhu_dx": dhu_dx.mean().detach(), 
            "mean_dhv_dy": dhv_dy.mean().detach(),
            "mean_R": R.mean().detach(),
            "mean_residual": residual.abs().mean().detach(),
        }
        
        return loss, summary