import os 
import shutil
import numpy as np
import csv

def read_file(file_path):
    data = []
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        for row in csv_reader:
            data.append(row)
    return data

if __name__ == '__main__':
    save_loc = './paired_samples'
    root_dir = '/media/pranjay/Datasets/Environment_Variations/Deblur/RealBlur/RealBlur' 

    train_list_path = root_dir + '/RealBlur_J_train_list.txt'
    test_list_path = root_dir + '/RealBlur_J_test_list.txt'

    train_list = read_file(train_list_path)  
    test_list = read_file(test_list_path) 

    train_file_paths = []
    for file_data in train_list:
        gt = file_data[0]
        blur = file_data[1]
        in_image = os.path.join(root_dir, blur)
        out_image = os.path.join(root_dir, gt)
        train_file_paths.append([in_image, out_image])

    test_file_paths = []
    for file_data in test_list:
        gt = file_data[0]
        blur = file_data[1]
        in_image = os.path.join(root_dir, blur)
        out_image = os.path.join(root_dir, gt)
        test_file_paths.append([in_image, out_image])
    
    print('Created {} RGB image pairs in Train Set'.format(len(train_file_paths)))
    print('Created {} RGB image pairs in Test Set'.format(len(test_file_paths)))

    np.save(save_loc+'/real_blur.npy', train_file_paths)
    np.save(save_loc+'/real_blur_test.npy', test_file_paths)
