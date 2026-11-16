# Windows environment (RTX 4090)

Environment name: `flooddiff`. Python 3.10, PyTorch 2.0.1, torchvision
0.15.2, CUDA runtime 11.8. These replace the original torchvision 0.13
requirement for this GPU setup. NumPy stays below 2 for compatibility.

## Create

In a PowerShell session with Conda available:

```powershell
conda create -n flooddiff python=3.10 pip -y
conda activate flooddiff
python -m pip install -r requirements-windows.txt
```

The Windows requirements include dependencies imported by the project but
missing from the original requirements files. A separate CUDA Toolkit
installation is unnecessary for the PyTorch wheels.

## Activate

```powershell
conda activate flooddiff
```

If PowerShell has not loaded Conda, load its hook first:

```powershell
(& 'C:\Users\Hank\miniconda3\Scripts\conda.exe' 'shell.powershell' 'hook') | Out-String | Invoke-Expression
conda activate flooddiff
```

## Verify

```powershell
python -m pip check
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); x = torch.randn(128, 128, device='cuda'); print((x @ x).mean().item())"
python train.py --help
python sample.py --help
```

Training and inference additionally require datasets and model checkpoints.
Pass paths for this machine; some defaults in the source reference another
user's directories.
