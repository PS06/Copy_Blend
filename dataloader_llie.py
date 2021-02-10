import numpy as np
import torchvision
import torch
from torchvision import datasets, models, transforms
import torchvision.transforms.functional as functional
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance
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
        patch = Image.fromarray(np.zeros((height, width)))
        low_image.paste(patch, (j, i))
    return low_image 

def cut_mix(low_image, alt_image, patch_size, num_patch=1):
    for _ in range(num_patch):
        i, j, height, width = get_crop_params(low_image, patch_size)
        region = np.asarray(functional.crop(alt_image, i, j, height, width))
        patch = Image.fromarray(region, 'RGB')
        low_image.paste(patch, (j, i))
    return low_image 

def lda(low_image, high_image, patch_size, num_patch, shape):
    if shape == 'all':
        shape = random.choice(['cicle', 'polygon', 'square', 'rect'])

    for _ in range(num_patch):
        i, j, height, width = get_crop_params(low_image, patch_size)
        if shape in ['cicle', 'polygon']:
            mask = Image.new("L", low_image.size, 255)
            draw = ImageDraw.Draw(mask)

            if shape == 'cicle':
                if random.random() > 0.5:
                    draw.ellipse((j, i, height, width), fill=0)
                else:
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
    elif 'lda' in mode:
        shape = mode.split('_')[-1]
        res_image = lda(low_image, high_image, patch_size, num_patch, shape)
    else:
        res_image = low_image

    return res_image

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".bmp", ".png", ".jpg", ".jpeg"])

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
        
    def __getitem__(self, index):
        low_img_path, high_img_path = self.image_pair[index]
        high_img = Image.open(high_img_path).convert('RGB')
        low_img = Image.open(low_img_path).convert('RGB')

        # Random Crop
        i, j, height, width = get_crop_params(high_img, self.crop_size)
        high_img = functional.crop(high_img, i, j, height, width)
        low_img = functional.crop(low_img, i, j, height, width)
        low_img_org = low_img.copy()
        high_img_org = high_img.copy()

        # Training Data Augmentation 
        if self.istrain: 

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

class Lowlight_DatasetFromVOC(torch.utils.data.Dataset):
    def __init__(self, args, imgFolder, istrain=False):
        super(Lowlight_DatasetFromVOC, self).__init__()
        self.imgFolder = imgFolder
        self.image_filenames = [os.path.join(self.imgFolder, x) for x in os.listdir(self.imgFolder) if is_image_file(x)]
        self.patch_size = args.patch_size
        self.to_tensor = transforms.ToTensor()
        self.istrain = istrain
        transform = [RandomRotate90(), Flip(), Transpose()]
        self.transform = Compose(transform)
        self.aug_type = args.aug_type
        self.crop_size = args.patch_size
        self.patch_size = args.aug_scale
        self.num_patch = args.num_patch

    def __getitem__(self, index):

        ori_img = Image.open(self.image_filenames[index])  # PIL image
        width, height = ori_img.size

        # Random Crop
        i, j, height, width = get_crop_params(ori_img, self.patch_size)
        high_img = functional.crop(ori_img, i, j, height, width)
        low_img = functional.crop(ori_img, i, j, height, width)

        ## color and contrast *dim*
        color_dim_factor = 0.3 * random.random() + 0.7
        contrast_dim_factor = 0.3 * random.random() + 0.7
        low_img = ImageEnhance.Color(low_img).enhance(color_dim_factor)
        low_img = ImageEnhance.Contrast(low_img).enhance(contrast_dim_factor)

        low_img = cv2.cvtColor((np.asarray(low_img)), cv2.COLOR_RGB2BGR)  # cv2 image
        low_img = (low_img.clip(0, 255)).astype("uint8")
        low_img = low_img.astype('double') / 255.0

        # generate low-light image
        beta = 0.5 * random.random() + 0.5
        alpha = 0.1 * random.random() + 0.9
        gamma = 3.5 * random.random() + 1.5
        low_img = beta * np.power(alpha * low_img, gamma)

        low_img = low_img * 255.0
        low_img = (low_img.clip(0, 255)).astype("uint8")
        low_img = Image.fromarray(cv2.cvtColor(low_img, cv2.COLOR_BGR2RGB))

        if self.istrain:      
            # Apply Data Augmentation Here
            if random.random() > 0.5: # patch size as a percentage of train image size
                if self.aug_type in ['cut_mix', 'mix_up']:
                    if self.aug_type == 'mix_up':
                        self.patch_size = self.patch_size/self.crop_size
                    alt_index = random.randrange(0, len(self.image_filenames))
                    alt_image_path = self.image_filenames[alt_index]
                    alt_image = Image.open(alt_image_path).convert('RGB')
                    alt_image = functional.crop(alt_image, i, j, height, width)
                    low_img = apply_data_aug(low_img, alt_image, self.aug_type, self.patch_size, self.num_patch)
                else:
                    low_img = apply_data_aug(low_img, high_img, self.aug_type, self.patch_size, self.num_patch)

            data = {"image": np.array(low_img), "mask": np.array(high_img)}
            augmented = self.transform(**data)
            low_img, high_img = augmented["image"], augmented["mask"]

        img_in = self.to_tensor(low_img)
        img_tar = self.to_tensor(high_img)

        return img_in, img_tar

    def __len__(self):
        return len(self.image_filenames)
