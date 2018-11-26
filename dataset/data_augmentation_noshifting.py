import os
import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imsave, imread
from skimage import transform

data_path = 'dataset_original/'
data_path_2 = 'dataset_augmented/'

images_row = 390
images_col = 482

rotation_count = 6

def create_augmented_data():
    data_original_path = os.path.join(data_path)
    images = os.listdir(data_original_path)

    #分别读取每一个phase map和其对应的mask
    image_id = 0
    image_count = 1
    for image_name in images:
        if 'Mask' in image_name:
            continue
        image_mask_name = image_name.split('.')[0] + '_Mask.jpg'
        img = imread(os.path.join(data_path, image_name), as_grey=True)
        img_mask = imread(os.path.join(data_path, image_mask_name), as_grey=True)
        img_mask = np.array(img_mask)
        img = np.array(img)

        #定义原输入图像的镜像图
        img_mirror = np.ndarray((images_row, images_col), dtype=np.uint8)
        img_mask_mirror = np.ndarray((images_row, images_col), dtype=np.uint8)
        img_mirror[:] = img
        img_mask_mirror[:] = img_mask
        for mr in range(0, images_row):
            for mc in range(0, int(images_col / 2)):
                inter = img_mirror[mr, mc]
                img_mirror[mr, mc] = img_mirror[mr, images_col - 1 - mc]
                img_mirror[mr, images_col - 1 - mc] = inter
                inter_mask = img_mask_mirror[mr, mc]
                img_mask_mirror[mr, mc] = img_mask_mirror[mr, images_col - 1 - mc]
                img_mask_mirror[mr, images_col - 1 - mc] = inter_mask

        # debug
       # plt.imshow(img, cmap='Greys_r')
        #plt.show()
       # plt.imshow(img_mask, cmap='Greys_r')
       # plt.show()

        #角度扩增
        for r_c in range(0, rotation_count):
            if r_c >= 3:
                degree_rotate = (-1 + (r_c - 3)) * 90
                img_r = transform.rotate(img_mirror, degree_rotate)
                img_mask_r = transform.rotate(img_mask_mirror, degree_rotate)
            else:
                degree_rotate = (-1 + r_c) * 90
                img_r = transform.rotate(img, degree_rotate)
                img_mask_r = transform.rotate(img_mask, degree_rotate)

            imsave(os.path.join(data_path_2, 'Img_augmented_' + '%04d' % image_count + '.jpg'), img_r)
            imsave(os.path.join(data_path_2, 'Img_augmented_' + '%04d' % image_count + '_Mask.jpg'), img_mask_r)
            image_count = image_count + 1

        #索引下一张图
        image_id = image_id + 1
        print('-' * 30)
        print('正在处理第：')
        print(image_id)
        print('张')
        print('-' * 30)


if __name__ == '__main__':
    create_augmented_data()
