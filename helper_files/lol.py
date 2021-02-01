import os
import numpy as np
import csv

def write_data(file_path, database):
    with open(file_path, 'w') as result_file:
        writer = csv.writer(result_file, delimiter=',')
        for data in database:
            writer.writerow(data)

def get_dataset(in_dir, gt_dir):
    high_images = sorted([os.path.join(gt_dir, file_name) for file_name in os.listdir(gt_dir)])
    low_images = sorted([os.path.join(in_dir, file_name) for file_name in os.listdir(in_dir)])
    paired_dataset = []
    for low_image, high_image in zip(low_images, high_images):
        low_file_path = os.path.join(in_dir, low_image)
        high_image_path = os.path.join(gt_dir, high_image)
        if os.path.isfile(low_file_path) and os.path.isfile(high_image_path):
            paired_dataset.append([low_file_path, high_image_path])
        else:
            raise Exception('File not found {} or {}'.format(low_file_path, high_image_path))
    return paired_dataset

if __name__ == '__main__':
    root_dir = '/media/pranjay/SSD_2/LOLdataset/'
    in_dir = root_dir + 'train/low'
    gt_dir = root_dir + 'train/high'
    eval_in = root_dir + 'our485/low'
    eval_gt = root_dir + 'our485/high'
    test_in = root_dir + 'eval15/low'
    test_gt = root_dir + 'eval15/high'
    save_loc_1 = './paired_samples'
    save_loc_2 = root_dir

    train_data = get_dataset(in_dir, gt_dir)
    train_data.extend(get_dataset(eval_in, eval_gt))
    test_data = get_dataset(test_in, test_gt)

    np.save(save_loc_1+'/retinex.npy', train_data)
    np.save(save_loc_1+'/retinex_test.npy', test_data)

    write_data(save_loc_2+'/retinex.txt', train_data)
    write_data(save_loc_2+'/retinex_test.txt', test_data)

    print('Created {} RGB image pairs in Train Set'.format(len(train_data)))
    print('Created {} RGB image pairs in Test Set'.format(len(test_data)))