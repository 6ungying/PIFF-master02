# 單通道水深模型

現在 `train.py` 與 `sample.py` 的目標皆為水深 `d`，張量形狀為
`[batch, 1, height, width]`。DEM 為單通道擴散橋起點，雨量序列保留。
資料集不再要求或讀取 Ground Truth 的 `vx、vy` 圖片。

## 訓練與預測

資料根目錄為 `C:\Users\Hank\Desktop\data`，訓練、預測與 VS Code
啟動設定皆已更新。可使用 `--dataset-dir` 指定其他資料根目錄。
單地形 DEM 使用其中的 `tainan_dem.png`，支援灰階或彩色圖；
找不到或無法讀取時會顯示完整路徑。本次未搬移資料或執行正式訓練。

```powershell
conda activate flooddiff
python train.py --name flood-depth-only --use-single-dem
python sample.py --ckpt flood-depth-only --use-single-dem
```

多地形訓練不加 `--use-single-dem`；預測時依原有規則使用
`--test-dem-list`、`--use-yilan` 或 `--use-hsinchu` 選擇資料集。

- `--spm`：保留單通道 SPM 輔助輸入，adapter 為 `2 → 1`。
- `--ca4d`：只讀取 CA4D 的水深 `d` 作為單通道輔助輸入，adapter 為
  `2 → 1`。Ground Truth、CA4D 引導與預測輸出都不需要流速檔案。
  VS Code 的 CA4D 實驗名稱加上 `-d-guidance`，避免覆寫先前使用三通道
  CA4D 引導的 checkpoint；請以新的設定重新訓練。
- 預測會依對應資料集的水深平均值與標準差反正規化，只儲存
  `recon_*_d_*.png` 水深灰階圖，不再產生流速預測圖。
- 原質量守恆損失需要預測的流速，因此單水深模式不支援
  `--use-physics`，會顯示錯誤。既有三通道 VQGAN 的 latent 模式同樣不支援。
- 請使用新的實驗名稱重新訓練。原本三通道 checkpoint 與單水深模型
  的權重形狀不相容，無法直接續訓或用於此版本的預測。

## 驗證

```powershell
python -m unittest discover -s tests -v
```

測試涵蓋四種水深資料集、不含流速標籤的 batch、SPM／CA4D 梯度、
單步訓練、checkpoint 儲存與重載、採樣、反正規化與灰階輸出。
測試使用合成資料，不代表已驗證真實資料上的預測精度。
