import os 
import shutil
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import csv
from PIL import Image
import numpy as np
import imageio
import os
import rawpy
import cv2

def read_file(file_path):
    data = []
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        for row in csv_reader:
            data.append(row)
    return data

def write_data(file_path, database):
    with open(file_path, 'w') as result_file:
        writer = csv.writer(result_file, delimiter=',')
        for data in database:
            writer.writerow(data)

def named_sliding_window(data):
    short_path, long_path, short_save_path, long_save_path, crop_size = data
    _, ext = os.path.splitext(short_path)
    ext = ext.replace('.', '')
    image_data = []
    if ext in ['jpg', 'png', 'JPG', 'PNG']:
        low_rgb_img = np.asarray(Image.open(short_path).convert('RGB'))
        high_rgb_img = np.asarray(Image.open(long_path).convert('RGB'))
    elif ext in ['ARW', 'RAF']:
        low_rgb_img = rawpy.imread(short_path)
        low_rgb_img = low_rgb_img.postprocess(output_bps=16)
        high_rgb_img = rawpy.imread(long_path)
        high_rgb_img = high_rgb_img.postprocess(output_bps=16)
    else:
        raise Exception
    
    assert (np.shape(low_rgb_img) == np.shape(high_rgb_img))
    w_1, h_1, _ = np.shape(low_rgb_img)

    sub_image = 0
    folder, scene = short_path.split('/')[-3], short_path.split('/')[-2]
    short_save_name = folder + '_' + scene + '_' + os.path.basename(short_path).replace('.{}'.format(ext), '')
    long_save_name = folder + '_' + scene + '_' + os.path.basename(long_path).replace('.{}'.format(ext), '')

    for w in range(0, w_1, crop_size):
        for h in range(0, h_1, crop_size):
            short_rgb_crop = low_rgb_img[w:w+crop_size, h:h+crop_size, :]         
            long_rgb_crop = high_rgb_img[w:w+crop_size, h:h+crop_size, :]
            
            if (short_rgb_crop.shape == (512, 512, 3)) and (long_rgb_crop.shape == (512, 512, 3)):
                short_crop_rgb_path = short_save_path + '/{}_{}.tif'.format(sub_image, short_save_name)
                long_crop_rgb_path = long_save_path + '/{}_{}.tif'.format(sub_image, long_save_name)

                imageio.imwrite(short_crop_rgb_path, short_rgb_crop)
                imageio.imwrite(long_crop_rgb_path, long_rgb_crop)
                sub_image += 1
                image_data.append([short_crop_rgb_path, long_crop_rgb_path])

    return image_data
    
def get_paired_data(gt_dir, in_dir):
    paired_list = []
    train_high_dir = sorted([os.path.join(gt_dir, file_name) for file_name in os.listdir(gt_dir)])
    for file_name in train_high_dir:
        high_image_path = file_name
        folder_name = os.path.basename(file_name).split('.')[0]
        low_folder = os.path.join(in_dir, folder_name)
        for low_file in os.listdir(low_folder):
            low_file_path = os.path.join(low_folder, low_file)

            if os.path.isfile(low_file_path) and os.path.isfile(high_image_path):
                paired_list.append([low_file_path, high_image_path])
            else:
                raise Exception('File not found {} or {}'.format(low_file_path, high_image_path))

    return paired_list


if __name__ == '__main__':
    root_dir = '/home/pranjay/Desktop/Research_Topics/Low_Light_Image_Enhancement/SICE'
    sice_1_gt = root_dir + '/Dataset_Part1/Label'
    sice_1_in = root_dir + '/Dataset_Part1/Dataset_Part1'
    sice_2_gt = root_dir + '/Dataset_Part2/Label'
    sice_2_in = root_dir + '/Dataset_Part2/Dataset_Part2'
    sice_test_gt = root_dir + '/Dataset_Part2/Test_Label'
    sice_test_in = root_dir + '/Dataset_Part2/Test'

    high_loc = root_dir + '/train/high_crop'
    low_loc = root_dir + '/train/low_crop' 
    save_loc_1 = './paired_samples'
    crop_size = 512
    
    for loc in [low_loc, high_loc]:
        if os.path.isdir(loc):
            shutil.rmtree(loc)
        os.makedirs(loc, exist_ok=True)
        
    train_img_data = get_paired_data(sice_1_gt, sice_1_in) 
    train_img_data.extend(get_paired_data(sice_1_gt, sice_1_in))
    test_img_data = get_paired_data(sice_test_gt, sice_test_in)

    print(np.shape(train_img_data), np.shape(test_img_data))

    crop_image_data = []
    for low_image, high_image in train_img_data:
        crop_image_data.append([low_image, high_image, low_loc, high_loc, crop_size])

    train_img_data = []
    pool = Pool(processes=12)
    # Sliding window for Training Data
    for image_data in tqdm(pool.imap_unordered(named_sliding_window, crop_image_data), total=len(crop_image_data)):
        train_img_data.extend(image_data)
    pool.close()
    pool.join()
    
    print('Created {} RGB image pairs in Train Set'.format(len(train_img_data)))
    print('Created {} RGB image pairs in Test Set'.format(len(test_img_data)))

    np.save(save_loc_1+'/sice.npy', train_img_data)
    np.save(save_loc_1+'/sice_test.npy', test_img_data)

    write_data(root_dir+'/sice.txt', train_img_data)
    write_data(root_dir+'/sice_test.txt', test_img_data)
