import os 
import shutil
import numpy as np

if __name__ == '__main__':
    save_loc = './paired_samples'
    root_dir = '/media/pranjay/Datasets/Environment_Variations/Deblur/GOPRO_Large' 

    train_dir = root_dir + '/train'
    test_dir = root_dir + '/test'
            
    train_file_paths = []
    folders = [os.path.join(train_dir, folder_name) for folder_name in os.listdir(train_dir)]
    for folder in folders:
        blur_folder = os.path.join(folder, 'blur')
        gt_folder = os.path.join(folder, 'sharp')
        for in_image, out_image in zip(sorted(os.listdir(blur_folder)), sorted(os.listdir(gt_folder))):
            in_image = os.path.join(blur_folder, in_image)
            out_image = os.path.join(gt_folder, out_image)
            train_file_paths.append([in_image, out_image])

    test_file_paths = []
    folders = [os.path.join(test_dir, folder_name) for folder_name in os.listdir(test_dir)]
    for folder in folders:
        gt_folder = os.path.join(folder, 'blur')
        blur_folder = os.path.join(folder, 'sharp')
        for in_image, out_image in zip(sorted(os.listdir(blur_folder)), sorted(os.listdir(gt_folder))):
            in_image = os.path.join(blur_folder, in_image)
            out_image = os.path.join(gt_folder, out_image)
            test_file_paths.append([in_image, out_image])
    
    print('Created {} RGB image pairs in Train Set'.format(np.shape(train_file_paths)))
    print('Created {} RGB image pairs in Test Set'.format(np.shape(test_file_paths)))

    np.save(save_loc+'/go_pro.npy', train_file_paths)
    np.save(save_loc+'/go_pro_test.npy', test_file_paths)
