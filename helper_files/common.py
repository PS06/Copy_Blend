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

def sliding_window(data):
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
    short_save_name = os.path.basename(short_path).replace('.{}'.format(ext), '')
    long_save_name = os.path.basename(long_path).replace('.{}'.format(ext), '')

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
    
def raw_2_rgb(data):
    low_image, high_image, short_test_loc, long_test_loc = data        
    low_raw = rawpy.imread(low_image)
    low_rgb = low_raw.postprocess(output_bps=16)
    high_raw = rawpy.imread(high_image)
    high_rgb = high_raw.postprocess(output_bps=16)
    _, ext = os.path.splitext(low_image)
    ext = ext.replace('.', '')
    low_rgb_image = os.path.basename(low_image).replace(".{}".format(ext), ".tif") 
    high_rgb_image = os.path.basename(high_image).replace(".{}".format(ext), ".tif") 

    short_rgb_test_loc = short_test_loc + '_rgb'
    long_rgb_test_loc = long_test_loc + '_rgb'

    for path in [short_rgb_test_loc, long_rgb_test_loc]:
        os.makedirs(path, exist_ok=True)

    low_rgb_image = os.path.join(short_rgb_test_loc, low_rgb_image)
    high_rgb_image = os.path.join(long_rgb_test_loc, high_rgb_image)

    imageio.imwrite(low_rgb_image, low_rgb)
    imageio.imwrite(high_rgb_image, high_rgb)

    return low_rgb_image, high_rgb_image 

def align_image(data):
    
    MAX_FEATURES = 5000
    GOOD_MATCH_PERCENT = 0.30

    # Convert images to grayscale
    im1 = cv2.imread(data[0])
    im2 = cv2.imread(data[1])

    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
    
    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    # cv2.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    try:
        # Find homography
        h, _ = cv2.findHomography(points1, points2, cv2.RANSAC)
        # Use homography
        height, width, _ = im2.shape
        im1Reg = cv2.warpPerspective(im1, h, (width, height))
        cv2.imwrite(data[0], im1Reg)
    except:
        print('Error for file -> {}'.format(data[0]))
