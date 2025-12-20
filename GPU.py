import torch

# 確認 PyTorch 是否可用
print(f"PyTorch 版本: {torch.__version__}")
print(f"是否安裝 CUDA: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"GPU 數量: {torch.cuda.device_count()}")
    print(f"GPU 名稱: {torch.cuda.get_device_name(0)}")
