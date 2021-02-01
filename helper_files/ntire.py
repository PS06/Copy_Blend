import os 
import shutil
import numpy as np

    
if __name__ == '__main__':
    save_loc_1 = './paired_samples'

    root_dir = '/media/pranjay/SSD_2/Haze'
    # reside_ots_in = root_dir + '/Reside/OTS/OTS'
    # reside_ots_gt = root_dir + '/Reside/OTS/clear_images'
    reside_its_in = root_dir + '/Reside/ITS/train/ITS_haze'
    reside_its_gt = root_dir + '/Reside/ITS/train/ITS_clear'
    
    ntire_19_root = root_dir + '/NTIRE-19/'
    ntire_20_root = root_dir + '/NTIRE-20/'

    ntire_19_train_in = ntire_19_root + 'hazy' 
    ntire_19_train_gt = ntire_19_root + 'GT' 
    ntire_19_test_in = ntire_19_root + 'test_hazy' 
    ntire_19_test_gt = ntire_19_root + 'test_gt' 

    num_images = 60000
    dehaze_train = []
    dehaze_test = []
    for low_image, high_image in zip(sorted(os.listdir(ntire_19_train_in)), sorted(os.listdir(ntire_19_train_gt))):
        low_image_path = os.path.join(ntire_19_train_in, low_image)
        high_image_path = os.path.join(ntire_19_train_gt, high_image)
        dehaze_train.append([low_image_path, high_image_path])
    
    ntire_19_test = []
    for low_image, high_image in zip(sorted(os.listdir(ntire_19_test_in)), sorted(os.listdir(ntire_19_test_gt))):
        low_image_path = os.path.join(ntire_19_test_in, low_image)
        high_image_path = os.path.join(ntire_19_test_gt, high_image)
        ntire_19_test.append([low_image_path, high_image_path])
        dehaze_test.append([low_image_path, high_image_path])

    print('Created {} RGB image pairs in NTIRE-19 Test Set'.format(np.shape(ntire_19_test)))
    np.save(save_loc_1+'/ntire_19_test.npy', ntire_19_test)


    ntire_20_train_in = ntire_20_root + 'HAZY' 
    ntire_20_train_gt = ntire_20_root + 'GT' 
    ntire_20_test_in = ntire_20_root + 'val_hazy' 
    ntire_20_test_gt = ntire_20_root + 'val_gt' 

    for low_image, high_image in zip(sorted(os.listdir(ntire_20_train_in)), sorted(os.listdir(ntire_20_train_gt))):
        low_image_path = os.path.join(ntire_20_train_in, low_image)
        high_image_path = os.path.join(ntire_20_train_gt, high_image)
        dehaze_train.append([low_image_path, high_image_path])
    
    ntire_20_test = []
    for low_image, high_image in zip(sorted(os.listdir(ntire_20_test_in)), sorted(os.listdir(ntire_20_test_gt))):
        low_image_path = os.path.join(ntire_20_test_in, low_image)
        high_image_path = os.path.join(ntire_20_test_gt, high_image)
        ntire_20_test.append([low_image_path, high_image_path])
        dehaze_test.append([low_image_path, high_image_path])

    print('Created {} RGB image pairs in NTIRE-20 Test Set'.format(np.shape(ntire_20_test)))
    np.save(save_loc_1+'/ntire_20_test.npy', ntire_20_test)


    # gt_images = []
    # hazy_images = sorted([os.path.join(reside_its_in, file_name) for file_name in os.listdir(reside_its_in)])
    # for file_name in sorted(os.listdir(reside_its_in)):
    #     gt_file = file_name.split('_')[0] + '.png'
    #     gt_images.append(os.path.join(os.path.join(reside_its_gt, gt_file)))
    # for index, (gt_image, hazy_image) in enumerate(zip(gt_images, hazy_images)):
    #     if index == num_images:
    #         break
    #     if os.path.isfile(gt_image) and os.path.isfile(hazy_image):
    #         dehaze_train.append([hazy_image, gt_image])


    # gt_images = []
    # hazy_images = sorted([os.path.join(reside_ots_in, file_name) for file_name in os.listdir(reside_ots_in)])
    # for file_name in sorted(os.listdir(reside_ots_in)):
    #     gt_file = file_name.split('_')[0] + '.jpg'
    #     gt_images.append(os.path.join(os.path.join(reside_ots_gt, gt_file)))
    # for index, (gt_image, hazy_image) in enumerate(zip(gt_images, hazy_images)):
    #     if index == num_images:
    #         break
    #     if os.path.isfile(gt_image) and os.path.isfile(hazy_image):
    #         dehaze_train.append([hazy_image, gt_image])

    print('Created {} RGB image pairs in Train Set'.format(np.shape(dehaze_train)))
    print('Created {} RGB image pairs in Test Set'.format(np.shape(dehaze_test)))

    np.save(save_loc_1+'/combined.npy', dehaze_train)
    np.save(save_loc_1+'/combined_test.npy', dehaze_test)