# 更新日誌

## 2026-11-16

### 模型、訓練與預測

- 將訓練目標與模型輸出由水深／流速三通道改為單通道水深 `d`；DEM 使用單通道，保留雨量序列輸入。
- 單地形、多地形、宜蘭與新竹資料集不再讀取 Ground Truth 流速圖片；CA4D 引導也只讀取水深。
- SPM 與 CA4D adapter 改為 `2 → 1` 通道，加入輸入維度檢查；UNet 正確採用指定的輸入、輸出通道數。
- 直接建立水深 UNet，移除外部模型架構 pickle 的固定路徑依賴。
- 預測依各資料集的平均值與標準差反正規化，只輸出 `recon_*_d_*.png` 水深灰階圖。
- 修正 batch 整理時的 SPM 初始化與 evaluation 回傳值解包，移除不再使用的流速處理及訓練依賴。

### 資料路徑與執行環境

- 訓練與預測新增 `--dataset-dir`，統一資料根目錄設定。
- 單地形 DEM 改用 `tainan_dem.png`，支援灰階／彩色圖，並在讀取失敗時提供路徑與設定提示。
- 更新 VS Code 啟動設定，採用 Windows `flooddiff` Conda 環境及獨立的水深實驗名稱。
- 新增 `requirements-windows.txt`、`SETUP_WINDOWS.md` 與 `DEPTH_ONLY.md`，記錄 Windows／RTX 4090 環境及水深模式操作方式。

### CA4D 模擬

- 新增 `CA4D.py`：使用 Numba CUDA 執行單一地形、單一雨型的淹水模擬，輸出附色條的水深圖，支援逐圖或共用色階。
- 腳本的資料路徑、雨型、模擬與出圖參數集中於檔案開頭；所需 Numba 等套件請參閱腳本說明。

### 相容性

- 舊三通道模型與三通道 CA4D 引導 checkpoint 無法直接沿用；載入時會明確報錯，需使用新實驗名稱重新訓練。
- 水深模式不支援需要流速預測的 `--use-physics`，亦不支援既有三通道 VQGAN latent 模式。

### 驗證

- 新增 `tests/test_depth_only.py`，涵蓋資料載入、無流速 batch、SPM／CA4D 梯度、單步訓練、checkpoint 重載、採樣、反正規化、灰階輸出與不相容設定檢查。
- 在 `flooddiff` 環境執行 `python -m unittest discover -s tests -v`：7 項測試全部通過。
- `git diff --check` 通過。上述測試使用合成資料，未執行完整真實資料訓練或 CA4D GPU 模擬驗證。
