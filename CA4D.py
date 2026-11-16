#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CA_4D 淹水模擬 - 單一地形(tif) x 單一雨型，GPU 融合 kernel 版

- 網格：256x256，每格 20 m
- 只輸出水深 d：色階不固定 (自動範圍)，附 color bar
- 淹水公式與邏輯同原版 (Manning、CFL、中央更新、鄰格流入、降雨、邊界排水)
  (原版 mask 恆為 1、Manning 為常數，故省略 mask 陣列，結果等價)

需求：numba, numpy, pandas, matplotlib, pillow，建議 tifffile (pip install tifffile)
"""

from __future__ import annotations

import os
import re
import glob
import math
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from numba import cuda, float64
from numba.core.errors import NumbaPerformanceWarning
from PIL import Image

warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

# ==========================================
#      參數與路徑
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEM_DIR = os.path.join(BASE_DIR, "yilan_dem")
DEM_FILE = None                      # 指定檔名，例如 "terrain01.tif"；None = 取 DEM_DIR 內第一個 tif
RAINFALL_PATH = os.path.join(BASE_DIR, "yilan_rainfall.csv")
RAIN_COLUMN = None                   # 指定欄名，例如 "inflow_01"；None = 取第一個含 "inflow_" 的欄位
OUTPUT_DIR = os.path.join(BASE_DIR, "yilan_Results_WSL")
LOG_FILE = os.path.join(BASE_DIR, "yilan_log_WSL.txt")

CELL_SIZE_M = 20.0                   # 每格邊長 (m)
MANNING_N = 0.03
TMAX_S = 24 * 3600.0
SAVE_EVERY_S = 3600.0
DT_MAX_S = 10.0
ALPHA_CFL = 0.7
HLIN = 0.01
SINK_FRAC = 0.5                      # 邊界排水比例 (原版 0.5)

# --- 出圖設定 (僅影響顯示，不影響模擬) ---
SHARED_COLOR_RANGE = False           # False: 每張圖各自自動範圍；True: 24 張共用同一色階 (0 ~ 全時段最大水深)
DISPLAY_MIN_DEPTH = 0.001            # 小於此水深 (m) 視為無水，顯示為灰白
COLORMAP = "Blues"
DPI = 150

# --- GPU block ---
TPB_X = 32                           # x 方向 = 欄 (j)，讓同一 warp 讀連續記憶體
TPB_Y = 8                            # 32*8 = 256 threads；k_flux 內的 block reduction 依賴此值


def write_to_log(message: str):
    print(message)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(message + "\n")
    except Exception:
        pass


# ==========================================
#      GPU Kernels (2 個融合 kernel)
# ==========================================
@cuda.jit(device=True)
def _dir_flow(wsi, zi, ws_n, z_n, dx):
    """回傳 (v, h, slope)；公式同原版 k_velocity。"""
    ds = wsi - ws_n
    if ds <= 0.0:
        return 0.0, 0.0, 0.0
    slope = ds / dx
    base_z = z_n if z_n > zi else zi
    htmp = wsi - base_z
    if htmp <= 0.0:
        return 0.0, 0.0, slope
    v = (htmp ** (2.0 / 3.0)) * math.sqrt(slope) / MANNING_N
    return v, htmp, slope


@cuda.jit
def k_flux(ws, water, z, dx, hlin, root_dx_over_hlin, x_off, y_off, Q, T, HM, vmax):
    """算各方向通量 Q、總流出 T、最大水深 HM，並用 block reduce + atomic max 求全域 vmax。"""
    j, i = cuda.grid(2)                      # j (欄) 在 x 軸 -> coalesced
    nrows, ncols = ws.shape
    tid = cuda.threadIdx.y * cuda.blockDim.x + cuda.threadIdx.x
    sh = cuda.shared.array(256, float64)

    lv = 0.0
    if i >= 1 and i < nrows - 1 and j >= 1 and j < ncols - 1:
        tot = 0.0
        hm = 0.0
        active = water[i, j] > 0.0
        wsi = ws[i, j]
        zi = z[i, j]
        for k in range(4):
            qk = 0.0
            if active:
                ni = i + y_off[k]
                nj = j + x_off[k]
                v, h, s = _dir_flow(wsi, zi, ws[ni, nj], z[ni, nj], dx)
                if v > lv:
                    lv = v
                if h > 0.0:
                    if h <= hlin:
                        qk = v * h * math.sqrt(s) * dx * root_dx_over_hlin
                    else:
                        qk = v * h * dx
                    if h > hm:
                        hm = h
            Q[i, j, k] = qk
            tot += qk
        T[i, j] = tot
        HM[i, j] = hm

    sh[tid] = lv
    cuda.syncthreads()
    stride = 128
    while stride > 0:
        if tid < stride:
            if sh[tid + stride] > sh[tid]:
                sh[tid] = sh[tid + stride]
        cuda.syncthreads()
        stride //= 2
    if tid == 0:
        cuda.atomic.max(vmax, 0, sh[0])


@cuda.jit
def k_update(Q, T, HM, water_in, water_out, ws, z, dx, dt, rain_amt, sink_frac,
             x_off, y_off, vmax):
    """中央更新 + 鄰格流入 + 降雨 + 水深更新 + 邊界排水 + 同步水位。"""
    j, i = cuda.grid(2)
    nrows, ncols = z.shape
    if i >= nrows or j >= ncols:
        return
    if i == 0 and j == 0:
        vmax[0] = 0.0                        # 為下一步重設 (host 已在本 kernel 啟動前讀走)

    wsi = ws[i, j]
    zi = z[i, j]
    ws2 = wsi

    if i >= 1 and i < nrows - 1 and j >= 1 and j < ncols - 1:
        coef = dt / (dx * dx)
        w_own = water_in[i, j]
        tot = T[i, j]
        if w_own > 0.0 and tot > 0.0:
            dH = tot * coef
            if dH <= w_own:
                ws2 = wsi - dH
            else:
                ws2 = zi + w_own - HM[i, j]
        inflow = 0.0
        for k in range(4):
            pi = i - y_off[k]
            pj = j - x_off[k]
            q = Q[pi, pj, k]
            if q <= 0.0:
                continue
            tp = T[pi, pj]
            wp = water_in[pi, pj]
            if tp <= 0.0 or wp <= 0.0:
                continue
            if tp * coef <= wp:
                inflow += q * coef
            else:
                hmx = HM[pi, pj]
                if hmx > 0.0:
                    inflow += hmx * q / tp
        ws2 += inflow

    if rain_amt > 0.0:
        ws2 += rain_amt

    w = ws2 - zi
    if w < 0.0:
        w = 0.0
    if (i == 0 or i == nrows - 1 or j == 0 or j == ncols - 1) and w > 0.0:
        w *= (1.0 - sink_frac)
    water_out[i, j] = w
    ws[i, j] = zi + w


# ==========================================
#      I/O
# ==========================================
def read_tif(path: str) -> np.ndarray:
    """讀 tif 高程。與原版 asc 一致做 flipud，使第 0 列為南邊。"""
    try:
        import tifffile
        arr = tifffile.imread(path)
    except ImportError:
        with Image.open(path) as img:
            arr = np.array(img)
    arr = np.squeeze(arr).astype(np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{path} 不是單波段 tif")
    return np.flipud(arr).copy()


def save_depth_png(depth: np.ndarray, path: str, title: str, vmax: float):
    """水深圖 + color bar。row 0 為南邊，用 origin='lower' 讓圖面北在上。"""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import matplotlib

    ny, nx = depth.shape
    data = np.ma.masked_less(depth, DISPLAY_MIN_DEPTH)
    cmap = matplotlib.colormaps[COLORMAP].copy()
    cmap.set_bad("#f0f0f0")

    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0

    fig = Figure(figsize=(6.6, 5.6))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    im = ax.imshow(data, origin="lower", cmap=cmap, vmin=0.0, vmax=vmax,
                   interpolation="nearest",
                   extent=[0, nx * CELL_SIZE_M, 0, ny * CELL_SIZE_M])
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Water depth (m)")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)


# ==========================================
#      Main
# ==========================================
def main():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("=== CA_4D Single-run GPU Simulation Log ===\n")
        f.write(f"Start Time: {datetime.now():%Y-%m-%d %H:%M:%S}\n")

    # ---- 地形 ----
    if DEM_FILE:
        dem_path = os.path.join(DEM_DIR, DEM_FILE)
    else:
        cands = sorted(glob.glob(os.path.join(DEM_DIR, "*.tif")) + glob.glob(os.path.join(DEM_DIR, "*.tiff")))
        if not cands:
            print(f"No .tif found in {DEM_DIR}")
            return
        dem_path = cands[0]
    terrain_name = os.path.splitext(os.path.basename(dem_path))[0]
    z_cpu = read_tif(dem_path)
    nrows, ncols = z_cpu.shape

    # ---- 雨型 ----
    rain_df = pd.read_csv(RAINFALL_PATH)
    col = RAIN_COLUMN
    if col is None:
        cols = [c for c in rain_df.columns if "inflow_" in c]
        if not cols:
            print("No 'inflow_' column in rainfall csv")
            return
        col = cols[0]
    m = re.search(r'(\d+)', col)
    rf_name = f"rf{int(m.group(1)):02d}" if m else col
    t_rain_s = rain_df['time'].values.astype(np.float64) * 3600.0
    r_rain_mps = (rain_df[col].values.astype(np.float64) / 1000.0) / 3600.0
    n_rain = len(t_rain_s)

    write_to_log(f"DEM: {terrain_name}  ({nrows}x{ncols}, dx={CELL_SIZE_M} m)   Rain: {col} -> {rf_name}")

    out_dir = os.path.join(OUTPUT_DIR, "d", terrain_name, rf_name)
    os.makedirs(out_dir, exist_ok=True)

    # ---- GPU 配置 (全部以 0 初始化；原版 Q 邊界格未初始化，這裡明確設 0) ----
    dx = float(CELL_SIZE_M)
    z_gpu = cuda.to_device(z_cpu)
    ws_gpu = cuda.to_device(z_cpu)
    water_cur = cuda.to_device(np.zeros_like(z_cpu))
    water_nxt = cuda.to_device(np.zeros_like(z_cpu))
    Q_gpu = cuda.to_device(np.zeros((nrows, ncols, 4), dtype=np.float64))
    T_gpu = cuda.to_device(np.zeros((nrows, ncols), dtype=np.float64))
    HM_gpu = cuda.to_device(np.zeros((nrows, ncols), dtype=np.float64))
    vmax_dev = cuda.to_device(np.zeros(1, dtype=np.float64))
    vmax_host = cuda.pinned_array(1, dtype=np.float64)
    x_off = cuda.to_device(np.array([1, -1, 0, 0], dtype=np.int64))
    y_off = cuda.to_device(np.array([0, 0, 1, -1], dtype=np.int64))

    threads = (TPB_X, TPB_Y)
    blocks = (math.ceil(ncols / TPB_X), math.ceil(nrows / TPB_Y))
    root_dx_over_hlin = (dx / HLIN) ** 0.5

    t = 0.0
    tu = SAVE_EVERY_S
    rain_idx = 0
    steps = 0
    frames = []                                   # (hour, depth ndarray)
    t_wall = time.time()

    while t < TMAX_S:
        k_flux[blocks, threads](ws_gpu, water_cur, z_gpu, dx, HLIN, root_dx_over_hlin,
                                x_off, y_off, Q_gpu, T_gpu, HM_gpu, vmax_dev)
        vmax_dev.copy_to_host(vmax_host)          # 每步唯一的 host 同步點
        vmax = vmax_host[0]
        dt = DT_MAX_S if vmax <= 1e-9 else min(DT_MAX_S, ALPHA_CFL * dx / vmax)

        while rain_idx < n_rain - 1 and t >= t_rain_s[rain_idx + 1]:
            rain_idx += 1
        rain_amt = r_rain_mps[rain_idx] * dt

        k_update[blocks, threads](Q_gpu, T_gpu, HM_gpu, water_cur, water_nxt, ws_gpu,
                                  z_gpu, dx, dt, rain_amt, SINK_FRAC, x_off, y_off, vmax_dev)
        water_cur, water_nxt = water_nxt, water_cur

        t += dt
        steps += 1

        if t >= tu - 1e-9:
            hour_idx = int(round(tu / 3600.0))
            w_host = water_cur.copy_to_host()
            frames.append((hour_idx, w_host))
            print(f"  t = {hour_idx:3d} h | steps = {steps:7d} | max depth = {w_host.max():.3f} m "
                  f"| elapsed = {time.time() - t_wall:.1f} s")
            tu += SAVE_EVERY_S

    sim_time = time.time() - t_wall

    # ---- 出圖 ----
    g_vmax = max((f[1].max() for f in frames), default=0.0)
    for hour_idx, w in frames:
        vmax_plot = g_vmax if SHARED_COLOR_RANGE else float(w.max())
        fname = f"ca4d_{terrain_name}_{rf_name}_d_{hour_idx:03d}.png"
        save_depth_png(w, os.path.join(out_dir, fname),
                       f"{terrain_name}  {rf_name}  t = {hour_idx} h", vmax_plot)

    total = time.time() - t_wall
    write_to_log("==================================================\n"
                 f"Completed at     : {datetime.now():%Y-%m-%d %H:%M:%S}\n"
                 f"Simulation time  : {sim_time:.1f} s (含 JIT 編譯)\n"
                 f"Time steps       : {steps}  ({1000.0 * sim_time / max(steps, 1):.3f} ms/step)\n"
                 f"Total (with PNG) : {total:.1f} s\n"
                 f"Output           : {out_dir}\n"
                 "==================================================")


if __name__ == "__main__":
    main()