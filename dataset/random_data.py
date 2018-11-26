import os
import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imsave, imread

data_path = 'random_data/'
test_data_path = 'test/'

image_rows = 390
image_cols = 482
sample_vector = np.ndarray((1, 1))
sample_dividing = 1

def create_random_training_data():

    count = 0
    for current_file in os.listdir(data_path):
        rt_data = os.path.join(data_path, current_file)
        images = os.listdir(rt_data)
        total = int(len(images) / 2)
        sample_vector[count] = round(total / sample_dividing)
        count = count + 1

    total_sum = int(sum(sample_vector))
    imgs_rand_sum = np.ndarray((total_sum, image_rows, image_cols), dtype=np.uint8)
    imgs_mask_rand_sum = np.ndarray((total_sum, image_rows, image_cols), dtype=np.uint8)

    count_2 = 0
    for current_file in os.listdir(data_path):
        rt_data = os.path.join(data_path, current_file)
        images = os.listdir(rt_data)
        total = int(len(images) / 2)
        sample_quant = round(total / sample_dividing)
        rand_number = np.arange(sample_quant)
        np.random.shuffle(rand_number)#生成随机索引号

        #将文件夹中所有文件分为两类，带mask的和不带mask的，得到imgs和imgs_mask
        imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
        imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
        i = 0
        for image_name in images:
            if 'Mask' in image_name:
              continue
            image_mask_name = image_name.split('.')[0] + '_Mask.jpg'
            img = imread(os.path.join(rt_data, image_name), as_grey=True)
            img_mask = imread(os.path.join(rt_data, image_mask_name), as_grey=True)

            img = np.array([img])
            img_mask = np.array([img_mask])

            imgs[i] = img
            imgs_mask[i] = img_mask

            if i % 100 == 0:
                print('Done: {0}/{1} images'.format(i, total))
            i += 1
        print('All data loading done.')

        #得到随机筛选的训练集
        imgs_rand = np.ndarray((sample_quant, image_rows, image_cols), dtype=np.uint8)
        imgs_mask_rand = np.ndarray((sample_quant, image_rows, image_cols), dtype=np.uint8)
        for j in range(0, sample_quant):
            imgs_rand[j] = imgs[rand_number[j]]
            imgs_mask_rand[j] = imgs_mask[rand_number[j]]

            if j % 20 == 0:
               print('Done: {0}/{1} images'.format(j+1, sample_quant))

        # 累加各个文件夹随机选出的图像组成最终的训练集
        if count_2 == 0:
            copy_number = int(sample_vector[0])
            for copy_index in range(0, copy_number):
                imgs_rand_sum[copy_index] = imgs_rand[copy_index]
                imgs_mask_rand_sum[copy_index] = imgs_mask_rand[copy_index]
        else:
            copy_start = 0
            for ci in range(0, count_2):
                copy_start = copy_start + int(sample_vector[ci])
            copy_end = copy_start + int(sample_vector[count_2])
            for copy_index_2 in range(copy_start, copy_end):
                imgs_rand_sum[copy_index_2] = imgs_rand[copy_index_2 - copy_start]
                imgs_mask_rand_sum[copy_index_2] = imgs_mask_rand[copy_index_2 - copy_start]
        print('The training dataset of current file loading done.')

        count_2 = count_2 + 1

        #随意显示任意一对儿图像
        # fig1 = plt.figure('fig1')
        # plt.imshow(imgs_rand[10], cmap='Greys_r')
        # fig1.show()
        # fig2 = plt.figure('fig2')
        # plt.imshow(imgs_mask_rand[10], cmap='Greys_r')
        # fig2.show()
        # print()

    np.save('imgs_rand_train.npy', imgs_rand_sum)
    np.save('imgs_rand_mask_train.npy', imgs_mask_rand_sum)
    print('Saving to .npy files done.')

def load_train_data():
    imgs_rand_train = np.load('imgs_rand_train.npy')
    imgs_rand_mask_train = np.load('imgs_rand_mask_train.npy')
    return imgs_rand_train, imgs_rand_mask_train

def create_test_data():
    train_data_path = os.path.join(test_data_path)
    images = os.listdir(train_data_path)
    images.sort(key= lambda x:int(x[:-4]))
    total = len(images)

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_id = np.ndarray((total, ), dtype=np.int32)

    i = 0
    print('-'*30)
    print('Creating test images...')
    print('-'*30)
    for image_name in images:
        img_id = int(image_name.split('.')[0])
        img = imread(os.path.join(train_data_path, image_name), as_grey=True)

        img = np.array([img])

        imgs[i] = img
        imgs_id[i] = img_id

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save('imgs_test.npy', imgs)
    np.save('imgs_id_test.npy', imgs_id)
    print('Saving to .npy files done.')


def load_test_data():
    imgs_test = np.load('imgs_test.npy')
    imgs_id = np.load('imgs_id_test.npy')
    return imgs_test, imgs_id

if __name__ == '__main__':
    create_random_training_data()
    create_test_data()





