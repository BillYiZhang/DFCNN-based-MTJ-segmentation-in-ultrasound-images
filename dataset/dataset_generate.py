from __future__ import print_function

import os

import numpy as np

from skimage.io import imsave, imread

import matplotlib.pyplot as plt

from skimage import filters

data_path = 'raw/'
data_path_2 = 'raw_new/'

image_rows = 390
image_cols = 482


def create_mask_data():
    mask_data_path = os.path.join(data_path)
    images = os.listdir(mask_data_path)
    image_id = 1
    for image_name in images:
        if 'Mask' in image_name:
            continue
        image_mask_name = image_name.split('.')[0] + '_Mask.jpg'
        img = imread(os.path.join(data_path, image_name),  as_grey=True)
        img_mask = imread(os.path.join(data_path, image_mask_name),  as_grey=True)
        img_mask = img_mask/255
        img = img/255
        judge = 0
        for c_i in range(0, image_rows):
            if judge == 1:
                break
            for c_j in range(0, image_cols):
                if img_mask[c_i, c_j] == 1:
                    corner_row = c_i + 1
                    corner_col = c_j + 1
                    judge = 1
                    break
        center_row = corner_row + 30
        center_col = corner_col + 45
        img_mask_new = np.zeros((image_rows, image_cols))
        img_mask_new[(center_row-20-1):(center_row+20-1), (center_col-50-1):(center_col+50-1)] = 1

        thresh = filters.threshold_otsu(img)
        img = (img > thresh)*1.0

        image = img*img_mask_new

        #debug
        #plt.figure(1)
        #plt.imshow(img_mask,  cmap='Greys_r')
        #plt.show()
        #plt.figure(2)
        #plt.imshow(img_mask_new, cmap='Greys_r')
        #plt.show()
        #plt.figure(3)
        #plt.imshow(img, cmap='Greys_r')
        #plt.show()
        #plt.figure(4)
        #plt.imshow(image, cmap='Greys_r')
        #plt.show()

        imsave(os.path.join(data_path_2, 'S3_Img_' + '%04d' % image_id + '_Mask.jpg'), image)
        image_id = image_id + 1

if __name__ == '__main__':
    create_mask_data()