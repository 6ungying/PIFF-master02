# 訓練模型儲存位置說明

## 📁 主要儲存位置

訓練的模型會儲存在 **`results/`** 資料夾中:

```
PIFF-master02/
└── results/
    ├── flood-single-b128-sde-norm-novar-rand01-PY/    # ← 你目前的訓練 (2025/11/29)
    │   ├── latest.pt                                   # 最新的模型權重
    │   ├── options.pkl                                 # 訓練配置
    │   └── py/                                         # Python 腳本備份
    │
    ├── flood-single-b128-sde-norm-novar-rand-phy/     # ← 新的物理損失訓練
    │   ├── latest.pt                                   # (如果已開始訓練)
    │   └── options.pkl
    │
    └── flood-single-b128-sde-norm-novar-rand01/       # 其他訓練實驗
        └── ...
```

---

## 🎯 儲存規則

### 1. 資料夾名稱來源
- 由 `train.py` 的 `--name` 參數決定:
```python
parser.add_argument("--name", type=str, 
    default='flood-single-b128-sde-norm-novar-rand01-PY')
```

- 完整路徑:
```python
opt.ckpt_path = RESULT_DIR / opt.name
# 結果: results/flood-single-b128-sde-norm-novar-rand01-PY/
```

### 2. 儲存頻率
- **每 100 個 iteration** 自動儲存一次 (Line 688 in runner.py)
```python
if it % 100 == 0:  # 改為每 100 步保存一次
    torch.save({...}, opt.ckpt_path / "latest.pt")
```

- 之前是每 1000 步,已修改為 100 步以便更頻繁備份

### 3. 儲存內容

**`latest.pt`** (PyTorch checkpoint 檔案):
```python
{
    "net": self.net.state_dict(),           # 模型權重
    "embedding": self.rainfall_emb.state_dict(),  # 降雨 embedding 權重
    "ema": ema.state_dict(),                # EMA (Exponential Moving Average) 權重
    "optimizer": optimizer.state_dict(),     # 優化器狀態
    "sched": sched.state_dict()             # Learning rate scheduler 狀態
}
```

**`options.pkl`** (訓練配置):
```python
# 儲存所有訓練參數
{
    'batch_size': 128,
    'lr': 5e-5,
    'use_physics': True,
    'physics_weight': 1.0,
    ...
}
```

---

## 🔍 查看訓練進度

### 方法 1: 查看最新的 checkpoint
```powershell
# 查看檔案修改時間
Get-ChildItem "results\flood-single-b128-sde-norm-novar-rand01-PY\latest.pt" | Select-Object Name, LastWriteTime
```

### 方法 2: 查看訓練 log
訓練 log 會顯示:
```
Saved latest(it=100) checkpoint to opt.ckpt_path='results\flood-single-b128-sde-norm-novar-rand01-PY'!
Saved latest(it=200) checkpoint to opt.ckpt_path='results\flood-single-b128-sde-norm-novar-rand01-PY'!
...
```

### 方法 3: 使用 Python 檢查 checkpoint
```python
import torch

ckpt = torch.load("results/flood-single-b128-sde-norm-novar-rand01-PY/latest.pt", 
                  map_location="cpu")
print("Checkpoint keys:", ckpt.keys())
print("Model parameters count:", sum(p.numel() for p in ckpt['net'].values()))
```

---

## 📊 目前的訓練實驗

### 1. **flood-single-b128-sde-norm-novar-rand01-PY** (主要訓練)
- **最後更新**: 2025/11/29 下午 4:57
- **狀態**: ✅ 有完整的 checkpoint
- **用途**: 基礎模型,用於 sampling 測試

### 2. **flood-single-b128-sde-norm-novar-rand-phy** (新訓練)
- **狀態**: 🔄 可能正在訓練或尚未開始
- **用途**: 使用物理損失的新模型

---

## 🔄 繼續訓練 (Resume Training)

如果訓練中斷,可以從最新的 checkpoint 繼續:

```powershell
python train.py --ckpt flood-single-b128-sde-norm-novar-rand01-PY
```

這會:
1. 載入 `results/flood-single-b128-sde-norm-novar-rand01-PY/latest.pt`
2. 恢復模型權重、優化器狀態、learning rate
3. 從上次的 iteration 繼續訓練

---

## 💾 備份建議

### 定期備份重要的 checkpoint:
```powershell
# 創建備份資料夾
New-Item -ItemType Directory -Force -Path "backups"

# 備份當前訓練
Copy-Item -Recurse `
    "results\flood-single-b128-sde-norm-novar-rand01-PY" `
    "backups\flood-single-b128-sde-norm-novar-rand01-PY_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
```

### 只備份 checkpoint (不含 log):
```powershell
# 只複製重要檔案
$dest = "backups\checkpoint_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
New-Item -ItemType Directory -Force -Path $dest
Copy-Item "results\flood-single-b128-sde-norm-novar-rand01-PY\latest.pt" $dest
Copy-Item "results\flood-single-b128-sde-norm-novar-rand01-PY\options.pkl" $dest
```

---

## 🚀 使用訓練好的模型

### Sampling (推論):
```powershell
python sample.py `
    --ckpt "results/flood-single-b128-sde-norm-novar-rand01-PY" `
    --batch-size 30 `
    --nfe 10
```

### 載入模型進行評估:
```python
from i2sb import Runner
from pathlib import Path

# 載入 checkpoint
ckpt_path = Path("results/flood-single-b128-sde-norm-novar-rand01-PY")
runner = Runner(opt, log, save_opt=False)

# 模型已自動載入 latest.pt
# 可以直接使用 runner.net 進行預測
```

---

## 📝 檔案大小參考

典型的 checkpoint 大小:
- **`latest.pt`**: ~500-800 MB (取決於模型架構)
- **`options.pkl`**: ~10 KB (配置檔案很小)

---

## ⚠️ 注意事項

1. **覆寫風險**: 
   - 使用相同的 `--name` 會覆寫舊的 checkpoint
   - 建議使用不同的實驗名稱 (例如加上日期或版本號)

2. **磁碟空間**:
   - 每 100 步儲存一次,但只保留 `latest.pt`
   - 不會自動保留歷史版本
   - 如果需要保留里程碑模型,需要手動備份

3. **分散式訓練**:
   - 只有 `global_rank == 0` 的 GPU 會儲存 checkpoint
   - 其他 GPU 會等待 (barrier)

---

## 🎓 總結

| 項目 | 說明 |
|------|------|
| **儲存位置** | `results/{experiment_name}/` |
| **主要檔案** | `latest.pt` (模型), `options.pkl` (配置) |
| **儲存頻率** | 每 100 個 iteration |
| **自動覆寫** | 是 (只保留最新版本) |
| **繼續訓練** | `python train.py --ckpt {experiment_name}` |
| **使用模型** | `python sample.py --ckpt "results/{experiment_name}"` |

**當前主要模型位置:**
```
results/flood-single-b128-sde-norm-novar-rand01-PY/latest.pt
```
