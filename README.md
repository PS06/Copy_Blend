## [STATUS - ACTIVE]
## [CB] - EVALUATING COPY-BLEND AUGMENTATION FOR LOW LEVEL VISION TASKS
Official Pytorch implementation.

#
### To-Do List  

- [] Paper Link
- [] Training Code
- [] Trained Models
- [] Citation

#
### Different Augmentation Techniques 
| ![Input](Images/IN.png)| ![Cut Mix](Images/cut_mix.png) | ![Cut Out](Images/cut_out.png)|![MixUp](Images/mix_up.png)| ![Cut Blur](Images/cut_blur.png) |![Copy Blend](Images/copy_blend.png) || ![Ground Truth](Images/GT.png) |  
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|


#
### Results 
| ![](Images/msnet_in.png)| ![](Images/msnet.png) | ![](Images/dln_in.png)| ![](Images/dln.png) | ![](Images/deblurganv2_in.png)| ![](Images/deblurganv2.png) |
|:---:|:---:|:---:|:---:|:---:|:---:|

#
### Dependencies and Installation

* python=3.8
* PyTorch=1.6
* tqdm
* numpy
* matplotlib
* PIL
* Matlab (For NIQE)
* OpenCV=4.4
* [albumentations](https://github.com/albumentations-team/albumentations)
* [piq](https://github.com/photosynthesis-team/piq)

#
## MODEL ZOO - Checkpoints of Evaluated Networks on different datasets
&nbsp;

### Task - Single Image Dehazing - [Trained Models]()
| Model  | [Ntire-19](https://data.vision.ee.ethz.ch/cvl/ntire19//dense-haze/) | [Ntire-20](https://competitions.codalab.org/competitions/22236) | 
|--------|:------------:|:------------:|
| [MSNet]()        |  13.32 / 0.53 / 4.21 | 12.04 / 0.50 / 4.08 |
| MSNet + CB()     |  14.71 / 0.58 / 3.87 | 13.97 / 0.57 / 3.77 | 
| [DIDH]()         |  15.71 / 0.54 / 4.71 | 14.71 / 0.45 / 5.34 |
| DIDH + CB        |  17.18 / 0.62 / 3.47 | 18.16 / 0.69 / 3.28 |
| [Dual_Res]()     |  PSNR / SSIM            |  PSNR / SSIM   | 
| Dual_Res + CB    |  PSNR / SSIM            |  PSNR / SSIM   | 
&nbsp;

### Task - Low Light Image Enhancement - [Trained Models]()
|  Model        |[Retinex](https://daooshee.github.io/BMVC2018website) | [SICE](https://github.com/csjcai/SICE) |
|--------|:------------:|:------------:| 
| [DLN]()       | 21.34 / 0.82 / 3.05 | 16.44 / 0.60 / 2.32 |  
| DLN + CB      | 21.34 / 0.82 / 3.05 | 16.44 / 0.60 / 2.32 |  
| [AFNet]()     | 20.17 / 0.81 / 3.17 | 18.75 / 0.64 / 2.42 |  
| AFNet + CB    | 20.84 / 0.84 / 2.73 | 18.91 / 0.65 / 2.49 |  
| [URIE]()      |  |  
| URIE + CB     |  |  
&nbsp;

### Task - Deblurring - [Trained Models]()
|  Model        |[GO PRO]() | [Real Blur]() |
|--------|:------------:|:------------:| 
| [DeblurGANv2]()       | 29.55 / 0.93 / 3.13 | 28.70 / 0.86 / 3.49 |  
| DeblurGANv2 + CB      | 29.91 / 0.93 / 3.07 | 31.26 / 0.92 / 3.19 |  
| [DMPHN]()             | 30.21 / 0.93 / 2.64 | 29.71 / 0.93 / 2.76 |  
| DMPHN + CB            | 30.21 / 0.94 / 2.73 | 31.18 / 0.94 / 2.50 |  
| []()      |  |  
|  + CB     |  |  
&nbsp;

#
### Performance Evaluation with Prior Local Region based Augmentations : 

| Evalaution Dataset | Retinex | NTIRE-19 | GO PRO |
|--------|:----------:|:----------:|:----------:|
| Baseline       | 21.33 / 0.81 | 13.32 / 0.53 | 29.55 / 0.93 |
| [Cut Mix]()    | 20.78 / 0.83 | 13.51 / 0.54 | 29.17 / 0.91 |
| [Mix Up]()     | 20.57 / 0.81 | 13.05 / 0.49 | 29.23 / 0.91 |
| [Cut Blur]()   | 21.39 / 0.83 | 13.77 / 0.60 | 29.99 / 0.94 |
| [Cut out]()    | 21.42 / 0.85 | 13.75 / 0.58 | 29.51 / 0.92 |
| Copy Blend     | 21.47 / 0.84 | 14.27 / 0.62 | 29.91 / 0.93 |
&nbsp;

#
## Execution Scripts
Fig 1
```shell
python fig1_exp.py
```

Fig 4
```shell
python fig4_exp.py
```

Fig 5
```shell
python fig5_exp.py
```

Eval All
```shell
python eval_all.py
```
#
### Citation


