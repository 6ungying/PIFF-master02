# 🚀 洪水預測擴散模型完整指南

## 📑 目錄
1. [訓練指南](#訓練指南)
2. [推理完全指南](#推理完全指南)
3. [模型對比分析](#模型對比分析)
4. [快速參考](#快速參考)
5. [常見問題](#常見問題)

---

# 訓練指南

## 🎯 目標概述
測試在單一地形或多地形資料集中加入不同引導機制（SPM、CA4D、物理損失）的效果

## ⚠️ 現狀分析

### 數據集選擇
- `floodDataset`：多地形資料（30個DEM）
- `singleDEMFloodDataset`：單一地形資料（台南）

### 訓練參數
```
--use-single-dem      # 使用單一地形而非多地形
--spm                 # 啟用 SPM 引導 (1通道)
--ca4d                # 啟用 CA4D 引導 (3通道)
--use-physics         # 啟用物理損失
--physics-weight      # 物理損失權重 (default: 1.0)
```

## 📊 訓練數據路徑

### 單一地形（台南）訓練資料
```
data/
├── train_rf.csv                    # 訓練雨量 CSV
├── test_rf.csv                     # 測試雨量 CSV
├── tainan_dem.png                  # DEM 地形圖
├── train/
│   ├── depth/RF##/*.png            # 水深 Ground Truth
│   ├── Vx/RF##/*.png               # 速度 X Ground Truth
│   └── Vy/RF##/*.png               # 速度 Y Ground Truth
├── test/
│   ├── depth/RF##/*.png
│   ├── Vx/RF##/*.png
│   └── Vy/RF##/*.png
└── ca4d/
    ├── single_train/               # CA4D 訓練引導 (3通道)
    └── single_test/                # CA4D 測試引導 (3通道)
```

### 多地形訓練資料
```
data/dems/
├── scenario_rainfall.csv           # 多地形雨量配置
├── dem_png/{dem_id}.png            # 各地形 DEM
├── train/
│   ├── d/                          # 水深 Ground Truth
│   ├── vx/                         # 速度 X Ground Truth
│   └── vy/                         # 速度 Y Ground Truth
├── test/
│   ├── d/
│   ├── vx/
│   └── vy/
├── SPM_71dem_output/               # SPM 先驗地圖 (1通道)
└── ca4d/Multi/                     # CA4D 多地形引導 (3通道)
```

## ✅ 訓練命令示例

### 1️⃣ 單一地形 + SPM
```powershell
python train.py `
  --name "flood-single-spm" `
  --use-single-dem `
  --spm `
  --batch-size 32 `
  --num-itr 100000
```

### 2️⃣ 單一地形 + CA4D
```powershell
python train.py `
  --name "flood-single-ca4d" `
  --use-single-dem `
  --ca4d `
  --batch-size 32
```

### 3️⃣ 單一地形 + SPM + 物理損失
```powershell
python train.py `
  --name "flood-single-spm-physics" `
  --use-single-dem `
  --spm `
  --use-physics `
  --physics-weight 0.5 `
  --batch-size 32
```

### 4️⃣ 多地形 + SPM
```powershell
python train.py `
  --name "flood-multi-spm" `
  --spm `
  --batch-size 128
```

### 5️⃣ 多地形 + CA4D
```powershell
python train.py `
  --name "flood-multi-ca4d" `
  --ca4d `
  --batch-size 128
```

### 6️⃣ 多地形 + CA4D + 物理損失
```powershell
python train.py `
  --name "flood-multi-ca4d-physics" `
  --ca4d `
  --use-physics `
  --batch-size 128
```

---

# 推理完全指南

## 🎯 推理的 3 個步驟

### 步驟 1️⃣：生成預測結果

使用 `sample.py` 進行推理：

```powershell
python sample.py `
  --ckpt "flood-single-spm-physics" `
  --batch-size 8 `
  --nfe 50 `
  --gpu 0
```

#### 參數對照

| 參數 | 說明 | 推薦值 |
|------|------|--------|
| `--ckpt` | 檢查點名稱（訓練時的 `--name`） | `flood-single-spm-physics` |
| `--batch-size` | 推理批量大小 | `8` 或 `16` |
| `--nfe` | 求解器步數 | `50`（平衡），`100`（精確） |
| `--gpu` | GPU 設備編號 | `0` |
| `--use-single-dem` | **推理單一地形** | （無值，只需存在） |
| `--test-dem-list` | **推理多地形** | `"8,29,62"` |

#### 推理結果位置
```
results/{模型名}/
├── test3_nfe{NFE}_euler-maruyama-dcvar/    ← 推理結果資料夾
│   ├── sample_000_pred_d.png               ← 預測水深
│   ├── sample_000_pred_vx.png              ← 預測速度 X
│   ├── sample_000_pred_vy.png              ← 預測速度 Y
│   ├── sample_000_gt_d.png                 ← Ground Truth 水深
│   ├── sample_000_gt_vx.png                ← GT 速度 X
│   └── sample_000_gt_vy.png                ← GT 速度 Y
├── latest.pt                               ← 訓練模型
├── options.pkl                             ← 訓練配置
└── metrics/                                ← 評估指標（下一步生成）
```

#### NFE 選擇指南

| NFE 值 | 耗時 | 品質 | 推薦場景 |
|--------|------|------|---------|
| 10 | 2-5 分鐘 | 👎 低 | 快速測試 |
| 50 | 5-15 分鐘 | 👍 中等 | **標準推薦** |
| 100 | 15-30 分鐘 | 👍👍 高 | 精確評估 |

### 步驟 2️⃣：計算評估指標

```powershell
python compute_metrices.py `
  --ckpt "flood-single-spm-physics"
```

#### 輸出指標

| 指標 | 說明 | 更好的值 |
|------|------|---------|
| RMSE | 均方根誤差 | 越小越好 |
| MAE | 平均絕對誤差 | 越小越好 |
| SSIM | 結構相似性 | 越接近 1 越好 |
| PSNR | 峰值信噪比 | 越大越好 |
| LPIPS | 感知距離 | 越小越好 |

#### 指標保存位置
```
results/{模型名}/metrics/
├── metrics_summary.csv             # 指標總結
├── metrics_per_sample.csv          # 每個樣本的指標
└── metrics_plot.png                # 指標可視化
```

### 步驟 3️⃣：視覺化結果

#### 快速查看 PNG 圖片
```powershell
explorer.exe results/{模型名}/test3_nfe50_euler-maruyama-dcvar/
```

#### Python 加載分析
```python
import numpy as np
import matplotlib.pyplot as plt

# 加載推理結果
data = np.load('results/{模型名}/recon_imgs.npz')

# 獲取數據
pred = data['pred']        # [N, 3, 256, 256] (d, vx, vy)
gt = data['gt']            # [N, 3, 256, 256]
corrupt = data['corrupt']  # [N, 3, 256, 256]

# 可視化第一個樣本
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(corrupt[0, 0], cmap='viridis'); axes[0].set_title('Input')
axes[1].imshow(pred[0, 0], cmap='viridis'); axes[1].set_title('Prediction')
axes[2].imshow(gt[0, 0], cmap='viridis'); axes[2].set_title('Ground Truth')
plt.tight_layout()
plt.savefig('comparison.png', dpi=150)
```

## 🔍 單一地形 vs 多地形推理

### 推理單一地形（台南）
```powershell
python sample.py `
  --ckpt "flood-single-spm-physics" `
  --use-single-dem `
  --batch-size 8 `
  --nfe 50 `
  --gpu 0
```
✅ 推理 **singleDEMFloodDataset**（1 地形 × 15 場景 × ~25 時步 = 375 樣本）

### 推理多地形（指定 3 個地形）
```powershell
python sample.py `
  --ckpt "flood-multi-spm" `
  --test-dem-list "8,29,62" `
  --batch-size 8 `
  --nfe 50 `
  --gpu 0
```
✅ 推理 **floodDataset**（3 地形 × 多場景 × 多時步）

---

# 模型對比分析

## 四個多地形模型詳細對比

| 模型 | 實際配置名 | SPM | CA4D | 物理損失 | 預期效果 |
|-----|----------|-----|------|---------|---------|
| **flood-multi-spm** | flood-single-b128-sde-norm-novar-rand04 | ✅ | ❌ | ❌ | ⭐⭐⭐ |
| **flood-multi-spm-mass** | flood-single-b128-sde-norm-novar-rand04-PY | ✅ | ❌ | ✅ | ⭐⭐⭐⭐ |
| **flood-multi-ca4d** | flood-single-b128-sde-norm-novar-ca4d | ❌ | ✅ | ❌ | ⭐⭐⭐⭐ |
| **flood-multi-ca4d-mass** | flood-single-b128-sde-norm-novar-ca4d-mass | ❌ | ✅ | ✅ | ⭐⭐⭐⭐⭐ |

## 性能預測金字塔

```
      ┌──────────────────────┐
      │ flood-multi-ca4d-    │
      │ mass (CA4D+Physics)  │ ⭐⭐⭐⭐⭐ 最佳
      ├──────────────────────┤
      │ flood-multi-ca4d     │ ⭐⭐⭐⭐ 很好
      │ (CA4D Only)          │
      ├──────────────────────┤
      │ flood-multi-spm-     │ ⭐⭐⭐⭐ 很好
      │ mass (SPM+Physics)   │
      ├──────────────────────┤
      │ flood-multi-spm      │ ⭐⭐⭐ 基礎
      │ (SPM Only)           │
      └──────────────────────┘
```

## 改進軌跡

### 維度 1：引導信息量
- **SPM**（1通道）：靜態先驗地圖
- **CA4D**（3通道）：動態 h, vx, vy 信息

→ **CA4D > SPM**（3倍信息量）

### 維度 2：物理約束
- **無物理損失**：純深度學習
- **有物理損失**：加入物理方程約束

→ **有物理 > 無物理**（引入先驗知識）

## 關鍵發現

### 🔍 命名解釋
- **flood-multi-spm**：SPM 引導
- **flood-multi-spm-mass**：SPM + Physics Loss（mass = 質量守恆）
- **flood-multi-ca4d**：CA4D 引導
- **flood-multi-ca4d-mass**：CA4D + Physics Loss

### ✅ 共同特徵
所有四個模型都是：
- ✅ 多地形模式
- ✅ 測試地形列表：[8, 29, 62]
- ✅ 批大小：128
- ✅ 使用 SDE 配置

## 推薦優先級

| 優先級 | 模型 | 原因 |
|------|------|------|
| 🥇 最高 | flood-multi-ca4d-mass | 最完整配置：CA4D(3ch) + Physics |
| 🥈 高 | flood-multi-ca4d | 驗證 CA4D 貢獻度 |
| 🥉 中 | flood-multi-spm-mass | 驗證 Physics 貢獻度 |
| 低 | flood-multi-spm | 基準方案（對比用） |

---

# 快速參考

## 🚀 最常用命令

### 訓練
```powershell
# 單一地形 + SPM
python train.py --name "flood-single-spm" --use-single-dem --spm --batch-size 32

# 多地形 + CA4D
python train.py --name "flood-multi-ca4d" --ca4d --batch-size 128
```

### 推理
```powershell
# 推理單一地形
python sample.py --ckpt "flood-single-spm" --use-single-dem --batch-size 8 --nfe 50

# 推理多地形
python sample.py --ckpt "flood-multi-ca4d" --test-dem-list "8,29,62" --batch-size 8 --nfe 50
```

### 評估
```powershell
# 計算指標
python compute_metrices.py --ckpt "flood-single-spm"
```

## ⏱️ 耗時估算

| 任務 | NFE 10 | NFE 50 | NFE 100 |
|------|--------|--------|----------|
| 推理 (375 樣本) | 2-5 分鐘 | 5-15 分鐘 | 15-30 分鐘 |
| 評估指標計算 | ~3 分鐘 | ~3 分鐘 | ~3 分鐘 |
| **總計** | ~5-8 分鐘 | ~10-20 分鐘 | ~20-35 分鐘 |

## 💾 重要檔案位置

| 檔案 | 位置 | 說明 |
|------|------|------|
| 訓練模型 | `results/{name}/latest.pt` | PyTorch checkpoint |
| 訓練配置 | `results/{name}/options.pkl` | 所有訓練參數 |
| 推理結果 | `results/{name}/test3_nfe{N}_*/*.png` | 預測圖片 |
| 評估指標 | `results/{name}/metrics/metrics_summary.csv` | 定量評估 |

---

# 常見問題

## ❓ Q: 如何知道訓練是否成功？
**A:** 查看以下信息：
```
✅ Training data length: XXXX
✅ train_it 1/1000000 | loss: +XX.XX
✅ Saved latest(it=0) checkpoint to results/{name}
```

## ❓ Q: 推理需要多久？
**A:** 375 個樣本，NFE=50，約 **5-15 分鐘**（取決於 GPU）

## ❓ Q: 推理用的是哪個資料集？
**A:** 
- 有 `--use-single-dem`：單一地形（台南，375 樣本）
- 有 `--test-dem-list`：多地形（指定地形）
- 都沒有：默認單一地形（向後兼容）

## ❓ Q: CA4D 和 SPM 的差異是什麼？
**A:** 
- **SPM**：1 通道靜態先驗（地形相關）
- **CA4D**：3 通道動態引導（h, vx, vy，時間相關）

## ❓ Q: 物理損失是什麼？
**A:** 在訓練中加入物理方程約束（質量守恆、動量方程等），讓模型學習物理規律

## ❓ Q: 如何批量推理多個模型？
**A:**
```powershell
python sample.py --ckpt "model1" --nfe 50 &
python sample.py --ckpt "model2" --nfe 50 &
python sample.py --ckpt "model3" --nfe 50 &
```

## ❓ Q: 評估指標哪個最重要？
**A:** 根據應用場景：
- **RMSE/MAE**：衡量預測誤差大小
- **SSIM**：衡量結構相似性（視覺效果）
- **PSNR**：衡量信噪比

---

## 📞 快速聯繫

- 訓練問題 → 檢查 `results/{name}/` 是否存在
- 推理問題 → 檢查 `--use-single-dem` 或 `--test-dem-list` 參數
- 評估問題 → 確保推理已完成，檢查 `recon/` 資料夾

---

**最後更新**: 2026-04-10  
**版本**: 1.0 (統合版)
