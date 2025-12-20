# FloodDiff
This is the code base of FloodDiff, the paper is currently reviewed and the code is verified by ICCV 2025.
## Requirements
```bash
conda env create --file requirements.yaml python=3
conda activate flooddiff
```
This code was built and tested with python 3.8 and torch 2.1.0
## Dataset
The training, testing dataset (include time series rainfall) and evaluation images for the results in the paper are provided in [link](https://drive.google.com/drive/folders/1N9ZAvTmtkQih-eYWm47XlJhUIwKmya3U?usp=sharing).
## FloodDiff Training

```
python train.py --name test_name \
                --dataset_dir <the folder of the downloaded dataset>
```
To modify the hyperparameters, please check train.py. 
## FloodDiff Sampling
```
python sample.py --ckpt <folder name of the test> \ 
                --dataset_dir <the folder of the downloaded dataset>
```
To modify the hyperparameters, please check sample.py. 
## Pretrained Models
All pretrained models (latent space model and pixel space model) are in [pretrain_models_zip_link](https://www.dropbox.com/scl/fo/ueh7u6848uz04ks5uiawu/AFz863yyXUiiaaIs16yyDT4?rlkey=cpb3y8hinz7cjzos3uec1llk7&e=1&dl=0).
## Sourse
The FloodDiff code was adapted from the following [BBDM](https://github.com/xuekt98/BBDM) and [I2SB](https://github.com/NVlabs/I2SB)
