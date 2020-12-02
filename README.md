## [STATUS - ACTIVE]
## [GLDA] - Greedy Localized Data Augmentation for Image Enhancement in presence of Non Homogeneous Noise 
Official Pytorch based implementation.

### To-Do List  

- [ ] Paper Link
- [ ] Training Code
- [ ] Trained Models
- [ ] Trained Models corresponding to experiments
- [ ] Citation


### Dependencies and Installation

* python=3.8
* PyTorch=1.6
* NVIDIA GPU+CUDA
* numpy
* matplotlib

#
## MODEL ZOO - Checkpoints of Evaluated Networks on different datasets
### SoTA Dehazing
| Model  | [Ntire-19](https://data.vision.ee.ethz.ch/cvl/ntire19//dense-haze/) | [Ntire-20](https://competitions.codalab.org/competitions/22236) | 
|--------|:------------:|:------------:|
| UNet           |  [Trained_Model]         |  [Trained_Model]|
|                |  (PSNR / SSIM)           |  (PSNR / SSIM)  |
| DIDH[]         |  [Trained_Model]         |  [Trained_Model]|
|                |  (PSNR / SSIM)           |  (PSNR / SSIM)  |
| SoTA-2[]       |  [Trained_Model]         |  [Trained_Model]|
|                |  (PSNR / SSIM)           |  (PSNR / SSIM)  | 



### SoTA LLIE
|  Model        |[SID](https://github.com/cchen156/Learning-to-See-in-the-Dark) | [SICE](https://github.com/csjcai/SICE) |
|--------|:------------:|:------------:| 
| SoTA-1[]       |  [Trained_Model]         | [Trained_Model]          | [Trained_Model]   |
|                |  (PSNR / SSIM)           | (PSNR / SSIM)            |   (PSNR / SSIM)   |  
| SoTA-2[]       |  [Trained_Model]         | [Trained_Model]          | [Trained_Model]   |
|                |  (PSNR / SSIM)           | (PSNR / SSIM)            |   (PSNR / SSIM)   |  


### Ablation Experiments : 

| Model - UNet | [DeepUPE](https://github.com/Jia-Research-Lab/DeepUPE) | [Retinex](https://daooshee.github.io/BMVC2018website) | [SICE](https://github.com/csjcai/SICE) |
|--------|:----------:|:---------:|:---------:|
| [Vanilla]               |  [Trained_Model]         |  [Trained_Model]        |  [Trained_Model]        |
|                         |  (PSNR / SSIM)           |  (PSNR / SSIM)          |  (PSNR / SSIM)          |  
| [Cut_Blur]              |  [Trained_Model]         |  [Trained_Model]        |  [Trained_Model]        |
|                         |  (PSNR / SSIM)           |  (PSNR / SSIM)          |  (PSNR / SSIM)          |  
| [Cut_out]               |  [Trained_Model]         |  [Trained_Model]        |  [Trained_Model]        |
|                         |  (PSNR / SSIM)           |  (PSNR / SSIM)          |  (PSNR / SSIM)          |  
| [Cut_Mix]               |  [Trained_Model]         |  [Trained_Model]        |  [Trained_Model]        |
|                         |  (PSNR / SSIM)           |  (PSNR / SSIM)          |  (PSNR / SSIM)          |  
| [Attentive_Cut_Mix]     |  [Trained_Model]         |  [Trained_Model]        |  [Trained_Model]        |
|                         |  (PSNR / SSIM)           |  (PSNR / SSIM)          |  (PSNR / SSIM)          |  
| [LDA]                   |  [Trained_Model]         |  [Trained_Model]        |  [Trained_Model]        |
|                         |  (PSNR / SSIM)           |  (PSNR / SSIM)          |  (PSNR / SSIM)          |  
| [G-LDA]                 |  [Trained_Model]         |  [Trained_Model]        |  [Trained_Model]        |
|                         |  (PSNR / SSIM)           |  (PSNR / SSIM)          |  (PSNR / SSIM)          |  


#
## Model Training on UNet

### Data Preprocessing 

For training purpose run -

```shell
python /helper/{dataset_name}.py
```

1. non overlapping crops of size 512 x 512 
2. generate paired lst in npy format to paired_samples folder


