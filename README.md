# [Alias-Free Convnets](https://google.com)

Official PyTorch implementation

--- 

## Requirements
We provide installation instructions for ImageNet classification experiments here,
based on [ConvNeXt](https://github.com/facebookresearch/ConvNeXt/tree/9a7b47bd6a6c156a8018dbd0c3b36303d4e564af)
instructions.

### Dependency Setup
Create an new conda virtual environment
```
conda create -n convnext python=3.8 -y
conda activate convnext
```

Install [Pytorch](https://pytorch.org/)>=1.8.0, [torchvision](https://pytorch.org/vision/stable/index.html)>=0.9.0 following official instructions. For example:
```
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install timm==0.3.2 tensorboardX six
```

The results in the paper are produced with `torch==1.8.0+cu111 torchvision==0.9.0+cu111 timm==0.3.2`.

### Dataset Preparation

Download the [ImageNet-1K](http://image-net.org/) classification dataset and structure the data as follows:
```
/path/to/imagenet-1k/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class2/
      img4.jpeg
```


## Train models

### Original ConvNeXt-Tiny
```
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1440 main.py \
--model convnext_tiny \
--drop_path 0.1 \
--batch_size 32  --update_freq 16 \
--lr 4e-3 \
--model_ema true --model_ema_eval true \
--data_set IMNET \
--data_path </path/to/imagenet> \
--output_dir </path/to/output/dir> \
 --epochs 300 --warmup_epochs 20 \

```

###  ConvNeXt-Tiny Baseline (circular convolutions)

```
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1440 main.py \
--model convnext_afc_tiny \
--blurpool_kwargs "{\"filt_size\": 1, \"scale_l2\":false}" \
--activation gelu \
--normalization_type C \
--drop_path 0.1 \
--batch_size 32  --update_freq 16 \
--lr 4e-3 \
--model_ema true --model_ema_eval true \
--data_set IMNET \
--data_path </path/to/imagenet> \
--output_dir </path/to/output/dir> \
 --epochs 300 --warmup_epochs 20 \

```


###  ConvNeXt-Tiny-AFC

```
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1440 main.py \
--model convnext_afc_tiny \
--drop_path 0.1 \
--blurpool_kwargs "{\"filter_type\": \"ideal\", \"scale_l2\":false}" \
--activation up_poly_per_channel \
--activation_kwargs "{\"in_scale\":7, \"out_scale\":7, \"train_scale\":true}" \
--model_kwargs "{\"stem_mode\":\"activation_residual\", \"stem_activation\": \"lpf_poly_per_channel\"}" \
--stem_activation_kwargs "{\"in_scale\":7, \"out_scale\":7, \"train_scale\":true, \"cutoff\":0.75}" \
--normalization_type CHW2 \
--batch_size 32  --update_freq 16 \
--lr 4e-3 \
--model_ema true --model_ema_eval true \
--data_set IMNET \
--data_path </path/to/imagenet> \
--output_dir </path/to/output/dir> \
 --epochs 300 --warmup_epochs 20 \

```

### Train ConvNeXt-Tiny-APS

```
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1440 main.py \
--model convnext_aps_tiny \
--drop_path 0.1 \
 --blurpool_kwargs "{\"filt_size\": 1}" \
--activation gelu \
--normalization_type C \
--batch_size 32  --update_freq 16 \
--lr 4e-3 \
--model_ema true --model_ema_eval true \
--data_set IMNET \
--data_path </path/to/imagenet> \
--output_dir </path/to/output/dir> \
 --epochs 300 --warmup_epochs 20 \

```


## Checkpoints
 
Trained models can be downloaded from:
https://drive.google.com/drive/folders/1IsqMWL8OVKNDQ7CNaHe8F2ox7GDmwMUs?usp=share_link


## Experiments
We provide instructions for running shift-invariance experiments to reproduce the paper results.
We use Convnext-Tiny-AFC as an example. the arguments regarding the model should be changed accordingly.

### Integer and fractional shift consistency

```
python -m torch.distributed.launch --nproc_per_node=2 --master_port=1440  main.py \
--model $ARCH  \
--blurpool_kwargs "{\"filter_type\": \"ideal\", \"scale_l2\":false}" \
--activation up_poly_per_channel \
--activation_kwargs "{\"in_scale\":7, \"out_scale\":7, \"train_scale\":true}" \
--model_kwargs "{\"stem_mode\":\"activation_residual\", \"stem_activation\": \"lpf_poly_per_channel\"}" \
--stem_activation_kwargs "{\"in_scale\":7, \"out_scale\":7, \"train_scale\":true, \"cutoff\":0.75}" \
--normalization_type CHW2 \
--batch_size 64 \
--data_set IMNET \
--data_path </path/to/imagenet> \
--output_dir </path/to/output/dir> \
--finetune $CKPT \
--model_key model_ema \
--eval true 

```

### Adversarial integer grid
### Adversarial half-pixel grid
### Adversarial fractional grid

### ImageNet-C

### Crop-shift

### Zero-padding bilinear interpolation
We use [spatial-pytorch](https://github.com/MadryLab/spatial-pytorch/tree/14ffe976a2387669f183a3fbe45fffd82b992c83)
repository for the experiments.
Copy the modified files and models from this repository. 

```
# clone spatial-pytorch to be in the same directory as this repository
git clone https://github.com/MadryLab/spatial-pytorch.git
cd spatial-pytorch/
# git checkout 14ffe976a2387669f183a3fbe45fffd82b992c83
# copy convnext models
cp -r ../alias_free_convnets/models/ ./robustness/imagenet_models/convnext
# copy modified files
cp -r ../alias_free_convnets/spatial-pytorch/* .

# install aditional requirements
pip install -r requirements.txt
pip install cox
```
spatial-pytorch repository requires loading checkpoint in format of its own models.
Use the script to convert the checkpoint to the format of spatial-pytorch models.
```
alias_free_convnets/spatial-pytorch/create_attacker_state_dict.py
```



---

## Acknowledgement
This repository is built using [Truly shift invariant CNNs](https://github.com/achaman2/truly_shift_invariant_cnns/tree/9c319a2f4734745b1a8f2375981750867db1078a) 
and [ConvNeXt](https://github.com/facebookresearch/ConvNeXt/tree/9a7b47bd6a6c156a8018dbd0c3b36303d4e564af) repositories.

* Truly shift invariant CNNs: 
  * https://arxiv.org/abs/2011.14214

* ConvNeXt
    * https://arxiv.org/abs/2201.03545
    * [LICENSE](alias_free_convnets/license/convnext_LICENSE.txt)

[//]: # (    * conda version )
[//]: # (Python 3.8	Miniconda3 Linux 64-bit	98.8 MiB	935d72deb16e42739d69644977290395561b7a6db059b316958d97939e9bdf3d)

---

## Citation
If you find this repository helpful, please consider citing:
```

```
