import numpy as np
import torchvision
import torch
from torchvision import datasets, models, transforms
import torchvision.transforms.functional as functional
from PIL import Image, ImageFilter, ImageDraw
import os
from tqdm import tqdm
import cv2 as cv
from torch.utils.data.dataloader import DataLoader
import time
import random
from torchvision.transforms import ColorJitter
from albumentations import IAAAdditiveGaussianNoise, GaussNoise, Flip, RandomRotate90, Compose, Transpose, MultiplicativeNoise
import itertools
from albumentations.pytorch import ToTensorV2

def get_crop_params(img, crop_size):
    w, h = img.size
    if isinstance(crop_size, int):
        th, tw = crop_size, crop_size
    elif isinstance(crop_size, list):
        th = crop_size[0] 
        tw = crop_size[1]
    else:
        raise TypeError

    if w == tw and h == th:
        return 0, 0, h, w
    else:
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw
     
def cut_blur(low_image, high_image, patch_size, num_patch=1):
    for _ in range(num_patch):
        i, j, height, width = get_crop_params(low_image, patch_size)
        patch = functional.crop(high_image, i, j, height, width)
        low_image.paste(patch, (j, i))
    return low_image 

def cut_out(low_image, patch_size, num_patch=1):
    for _ in range(num_patch):
        i, j, height, width = get_crop_params(low_image, patch_size)
        patch = Image.fromarray(128*np.ones((height, width)))
        low_image.paste(patch, (j, i))
    return low_image 

def cut_mix(low_image, alt_image, patch_size, num_patch=1):
    for _ in range(num_patch):
        i, j, height, width = get_crop_params(low_image, patch_size)
        region = np.asarray(functional.crop(alt_image, i, j, height, width))
        patch = Image.fromarray(region, 'RGB')
        low_image.paste(patch, (j, i))
    return low_image 

def copy_blend(low_image, high_image, patch_size, num_patch):
    shapes = ['square', 'rect', 'cicle', 'ellipse', 'polygon'] 

    for _ in range(num_patch):
        idx = random.randrange(0, len(shapes))
        shape = shapes[idx]
        i, j, height, width = get_crop_params(low_image, patch_size)
        
        if shape in ['cicle', 'ellipse', 'polygon']:
            mask = Image.new("L", low_image.size, 255)
            draw = ImageDraw.Draw(mask)

            if shape == 'ellipse':
                draw.ellipse((j, i, height, width), fill=0)
            elif shape == 'cicle':
                radius = max(height, width)
                draw.ellipse((j, i, radius, radius), fill=0)
            else:
                n_sides = random.randint(3, 15)
                draw.regular_polygon((j, i, min(height, width)), n_sides, fill=0)

            low_image = Image.composite(low_image, high_image, mask)            

        else:
            if shape == 'square':
                width = height
            
            low_patch = functional.crop(low_image, i, j, height, width)
            high_patch = functional.crop(high_image, i, j, height, width)
            patch = Image.blend(low_patch, high_patch, random.random())
            low_image.paste(patch, (j, i))

    return low_image

def apply_data_aug(low_image, high_image, mode, patch_size, num_patch):

    if mode == 'cut_blur':
        res_image = cut_blur(low_image, high_image, patch_size)
    elif mode == 'cut_out':
        res_image = cut_out(low_image, patch_size)
    elif mode == 'cut_mix':
        res_image = cut_mix(low_image, high_image, patch_size)
    elif mode == 'mix_up':
        assert patch_size <= 1.0
        res_image = Image.blend(low_image, high_image, patch_size)
    elif mode == 'copy_blend':
        res_image = copy_blend(low_image, high_image, patch_size, num_patch)
    else:
        res_image = low_image

    return res_image

class paired_image_loader(torch.utils.data.Dataset):
    def __init__(self, args, img_pair_path, istrain=False):

        self.aug_type = args.aug_type
        self.crop_size = args.patch_size

        self.patch_size = args.aug_scale
        self.num_patch = args.num_patch
        num_samples = len(img_pair_path)
        self.image_pair = img_pair_path[:int(args.dataset_ratio*num_samples)]
        self.istrain = istrain
        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()
        self.image_list = list(sorted(set(itertools.chain(*self.image_pair))))
            
        self.synthetic = []
        self.real = []
        for gt_file, haze_file in self.image_pair:
            if ('ITS' in gt_file) or ('OTS' in gt_file):
                self.synthetic.append([gt_file, haze_file])
            else:
                self.real.append([gt_file, haze_file])

        
    def __getitem__(self, index):

        if index % 2 == 0:
            index = random.randint(0, len(self.synthetic)-1)
            low_img_path, high_img_path = self.synthetic[index]
        else:
            index = random.randint(0, len(self.real)-1)
            low_img_path, high_img_path = self.real[index]

        high_img = Image.open(high_img_path).convert('RGB')
        low_img = Image.open(low_img_path).convert('RGB')
            
        # Training Data Augmentation 
        if self.istrain: 
            # Random Crop
            i, j, height, width = get_crop_params(high_img, self.crop_size)
            high_img = functional.crop(high_img, i, j, height, width)
            low_img = functional.crop(low_img, i, j, height, width)
            low_img_org = low_img.copy()
            high_img_org = high_img.copy()

            # Swapping low image with high image
            if random.random() > 0.5:
                low_img = high_img_org.copy()
                high_img = low_img_org.copy()

            transform = [RandomRotate90(), Flip(), Transpose()]

            if random.random() > 0.5:
                low_img = ColorJitter()(low_img)

            # Apply Noise Here
            # if self.noise_type != 'None' and random.random() > 0.5:
            #     if self.noise_type == 'add_gauss':
            #         transform.append(IAAAdditiveGaussianNoise())
            #     elif self.noise_type == 'gauss':
            #         transform.append(GaussNoise())
            #     else:
            #         transform.append(MultiplicativeNoise(multiplier=[0.5, 1.5], elementwise=True, per_channel=True, p=1)) 

            # Apply Data Augmentation Here
            if random.random() > 0.5: # patch size as a percentage of train image size
                if self.aug_type in ['cut_mix', 'mix_up']:
                    if self.aug_type == 'mix_up':
                        self.patch_size = self.patch_size/self.crop_size
                    alt_index = random.randrange(0, len(self.image_list))
                    alt_image_path = self.image_list[alt_index]
                    alt_image = Image.open(alt_image_path).convert('RGB')
                    alt_image = functional.crop(alt_image, i, j, height, width)
                    low_img = apply_data_aug(low_img, alt_image, self.aug_type, self.patch_size, self.num_patch)
                else:
                    low_img = apply_data_aug(low_img, high_img, self.aug_type, self.patch_size, self.num_patch)
   
            self.transform = Compose(transform)
            data = {"image": np.array(low_img), "mask": np.array(high_img_org)}
            augmented = self.transform(**data)
            low_img, high_img = augmented["image"], augmented["mask"]
    
        low_img = self.to_tensor(low_img)
        high_img = self.to_tensor(high_img)

        return low_img, high_img

    def __len__(self):
        return len(self.image_pair)

