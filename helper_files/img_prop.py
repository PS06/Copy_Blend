import os
import numpy as np
from piq import psnr, ssim
from multiprocessing.pool import Pool
from torchvision import transforms
from PIL import Image
import csv

transform = transforms.ToTensor()

def read_config_file(file_path):
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for data_base in csv_reader:
            frame = data_base
    return frame[0]

def calc_metrics(gt_image_path, processed_image_path):
    transform = transforms.ToTensor()
    
    gt_image = transform(Image.open(gt_image_path).convert('RGB'))
    processed_image = transform(Image.open(processed_image_path).convert('RGB'))
    ssim_value = ssim(gt_image, processed_image)
    psnr_value = psnr(gt_image, processed_image)
    return ssim_value.numpy(), psnr_value.numpy()

def eval_img_metrics(in_gt_img, processed_images):
    assert len(in_gt_img) == len(processed_images), '# gt images {} != # processed images {}'.format(len(in_gt_img), len(processed_images))
    file_list = []
    for in_file_name, processed_file_name in zip(in_gt_img, processed_images):
        file_list.append([in_file_name, processed_file_name])

    eval_metrics = []   
    pool = Pool(processes=12)
    eval_metrics.append(pool.starmap(calc_metrics, file_list))
    pool.close()
    pool.join()

    eval_metrics = eval_metrics[0]
    eval_metrics = np.mean(eval_metrics, axis=0)
    ssim_value = eval_metrics[0]
    psnr_value = eval_metrics[1]

    in_img_dir = os.path.dirname(processed_images[0])
    mat_cmd = "matlab -nodisplay -nosplash -nodesktop -r \"run(niqe_calc('{}/')); quit; \" | tail +11".format(in_img_dir)
    os.system(mat_cmd)
    niqe_value = float(read_config_file('./niqe_data.txt'))
    os.remove('./niqe_data.txt')

    return psnr_value, ssim_value, niqe_value

if __name__ == '__main__':

    for file_name in os.listdir('./paired_samples'):
        if '_test.npy' in file_name:
            file_name = os.path.join('./paired_samples', file_name)
            file_info = np.load(file_name)
            gt_images = []
            in_images = []
            for in_image, gt_image in file_info:
                in_images.append(in_image)
                gt_images.append(gt_image)
            psnr_value, ssim_value, niqe_value = eval_img_metrics(gt_images, in_images)
            print(f'{file_name} psnr_value : {psnr_value} ssim_value : {ssim_value} niqe_value : {niqe_value}')


