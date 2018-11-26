from __future__ import print_function

import os
import tensorflow as tf
from skimage.transform import resize
from skimage.io import imsave, imread
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, BatchNormalization, Flatten, Dense
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras import backend as K
from keras import regularizers
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from random_dataset import load_train_data, load_test_data

images_row = 390
images_col = 482
img_cols = 384
img_rows = 384
smooth = 1
data_path = 'preds_mask/'
accuracy_calculated = 1


def preprocess(imgs):
    imgs_mp = np.ndarray((imgs.shape[0], img_rows, img_cols, 3), dtype=np.uint8)
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_cols, img_rows), preserve_range=True)
    imgs_mp[:, :, :, 0] = imgs_p
    imgs_mp[:, :, :, 1] = 0
    imgs_mp[:, :, :, 2] = 0
    return imgs_mp

def preprocess_2(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_cols, img_rows), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def focal_loss(y_true, y_pred):
    # defone the hyperparameters
    alpha = 0.25
    gamma = 2.0
    # compute the focal loss
    alpha_factor = K.ones_like(y_true) * alpha
    alpha_factor = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
    focal_weight = tf.where(K.equal(y_true, 1), 1 - y_pred, y_pred)
    focal_weight = alpha_factor * focal_weight ** gamma
    _focal = focal_weight * K.binary_crossentropy(y_true, y_pred)
    return _focal


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


if __name__ == '__main__':

    print('-'*30)
    print('Loading and preprocessing data...')
    print('-'*30)

    imgs_train, imgs_mask_train = load_train_data()

    imgs_train = preprocess(imgs_train)

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train[:, :, :, 0])  # mean for data centering
    std = np.std(imgs_train[:, :, :, 0])  # std for data normalization

    imgs_train[:, :, :, 0] -= mean
    imgs_train[:, :, :, 0] /= std

    imgs_test, imgs_id_test = load_test_data()
    imgs_test = preprocess(imgs_test)

    imgs_test = imgs_test.astype('float32')
    imgs_test[:, :, :, 0] -= mean
    imgs_test[:, :, :, 0] /= std

    #debug
    #plt.imshow(imgs_train[0, :, :, 0], cmap='Greys_r')
    #plt.show()
    #plt.imshow(imgs_mask_train[0, :, :, 0], cmap='Greys_r')
    #plt.show()

    print('-'*30)
    print('transfer the vgg16 to our own task...')
    print('-'*30)

    vgg16 = VGG16(weights='imagenet', include_top=False)

    print('-' * 30)
    print('Build the new model with transferred layers and new deconv layers...')
    print('-' * 30)

    inputs = Input((img_rows, img_cols, 3))

    conv1 = Conv2D(64, (11, 11), activation='relu', padding='same', name='conv1_1')(inputs)
    conv1 = Conv2D(64, (11, 11), activation='relu', padding='same', name='conv1_2')(conv1)
    conv1 = Conv2D(64, (11, 11), activation='relu', padding='same', name='conv1_3')(conv1)
    # 首层局部感受野为15*15，通道数与VGG16相同
    batchnorm1 = BatchNormalization(axis=3)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), name='pool1')(batchnorm1)

    conv2 = vgg16.get_layer('block2_conv1')(pool1)
    conv2 = vgg16.get_layer('block2_conv2')(conv2)
    batchnorm2 = BatchNormalization(axis=3)(conv2)
    pool2 = vgg16.get_layer('block2_pool')(batchnorm2)

    conv3 = vgg16.get_layer('block3_conv1')(pool2)
    conv3 = vgg16.get_layer('block3_conv2')(conv3)
    conv3 = vgg16.get_layer('block3_conv3')(conv3)
    batchnorm3 = BatchNormalization(axis=3)(conv3)
    pool3 = vgg16.get_layer('block3_pool')(batchnorm3)

    conv4 = vgg16.get_layer('block4_conv1')(pool3)
    conv4 = vgg16.get_layer('block4_conv2')(conv4)
    conv4 = vgg16.get_layer('block4_conv3')(conv4)
    batchnorm4 = BatchNormalization(axis=3)(conv4)
    pool4 = vgg16.get_layer('block4_pool')(batchnorm4)

    conv5 = vgg16.get_layer('block5_conv1')(pool4)
    conv5 = vgg16.get_layer('block5_conv2')(conv5)
    conv5 = vgg16.get_layer('block5_conv3')(conv5)
    batchnorm5 = BatchNormalization(axis=3)(conv5)
    pool5 = vgg16.get_layer('block5_pool')(batchnorm5)

    conv6 = Conv2D(1024, (3, 3), activation='relu', padding='same', name='conv6_1')(pool5)
    conv6 = Conv2D(1024, (3, 3), activation='relu', padding='same', name='conv6_2')(conv6)
    batchnorm6 = BatchNormalization(axis=3)(conv6)

    conv7 = concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same', name='conv7_up')(batchnorm6), batchnorm5],
                        axis=3)
    conv7 = Conv2D(512, (5, 5), activation='relu', padding='same', name='conv7_1')(conv7)
    conv7 = Conv2D(512, (5, 5), activation='relu', padding='same', name='conv7_2')(conv7)
    batchnorm7 = BatchNormalization(axis=3)(conv7)

    conv8 = concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same', name='conv8_up')(batchnorm7), batchnorm4],
                        axis=3)
    conv8 = Conv2D(512, (5, 5), activation='relu', padding='same', name='conv8_1')(conv8)
    conv8 = Conv2D(512, (5, 5), activation='relu', padding='same', name='conv8_2')(conv8)
    batchnorm8 = BatchNormalization(axis=3)(conv8)

    conv9 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same', name='conv9_up')(batchnorm8), batchnorm3],
                        axis=3)
    conv9 = Conv2D(256, (5, 5), activation='relu', padding='same', name='conv9_1')(conv9)
    conv9 = Conv2D(256, (5, 5), activation='relu', padding='same', name='conv9_2')(conv9)
    batchnorm9 = BatchNormalization(axis=3)(conv9)

    conv10 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', name='conv10_up')(batchnorm9), batchnorm2],
                         axis=3)
    conv10 = Conv2D(128, (5, 5), activation='relu', padding='same', name='conv10_1')(conv10)
    conv10 = Conv2D(128, (5, 5), activation='relu', padding='same', name='conv10_2')(conv10)
    batchnorm10 = BatchNormalization(axis=3)(conv10)

    conv11 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', name='conv11_up')(batchnorm10), batchnorm1],
                         axis=3)
    conv11 = Conv2D(64, (5, 5), activation='relu', padding='same', name='conv11_1')(conv11)
    conv11 = Conv2D(64, (5, 5), activation='relu', padding='same', name='conv11_2')(conv11)
    batchnorm11 = BatchNormalization(axis=3)(conv11)

    conv12 = Conv2D(1, (1, 1), activation='sigmoid', name='conv12')(batchnorm11)

    # construct the new model
    new_model = Model(inputs=[inputs], outputs=[conv12])

    new_model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss, metrics=[dice_coef])

    print('-' * 30)
    print('Loading saved weights...')
    print('-' * 30)
    new_model.load_weights('weights_YI_Net.h5')

    print('-' * 30)
    print('Predicting masks on test data...')
    print('-' * 30)
    imgs_mask_test = new_model.predict(imgs_test, verbose=1, batch_size=4)
    np.save('imgs_mask_test.npy', imgs_mask_test)

    # 给出测试集最终的正确率
    if accuracy_calculated == 1:
        predm_dir = os.path.join(data_path)
        imgs_mask_manual = os.listdir(predm_dir)
        total_mask = int(len(imgs_mask_manual))
        imgs_mask_test_manual = np.ndarray((total_mask, images_row, images_col), dtype=np.uint8)
        i = 0
        for image_name in imgs_mask_manual:
            imgm = imread(os.path.join(data_path, image_name), as_grey=True)
            imgm = np.array([imgm])
            imgs_mask_test_manual[i] = imgm
            i = i + 1

        imgs_mask_test_manual = preprocess_2(imgs_mask_test_manual)
        imgs_mask_test_manual = imgs_mask_test_manual.astype('float32')
        imgs_mask_test_manual /= 255.  # scale masks to [0, 1]

        test_accuracy_vec = []
        test_len = imgs_mask_test.shape[0]
        for test_c in range(0, test_len):
             intersection = np.sum(imgs_mask_test_manual[test_c, :, :, 0] * imgs_mask_test[test_c, :, :, 0])
             test_accuracy_vec.append((2 * intersection) / (
             np.sum(imgs_mask_test_manual[test_c, :, :, 0]) + np.sum(imgs_mask_test[test_c, :, :, 0])))
        test_accuracy = np.mean(test_accuracy_vec)
        print('the testing accuracy is: ')
        print(test_accuracy)

    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)
    pred_dir = 'preds'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for image, image_id in zip(imgs_mask_test, imgs_id_test):
        image = resize(image, (images_row, images_col), preserve_range=True)
        image = (image[:, :, 0] * 255.).astype(np.uint8)
        imsave(os.path.join(pred_dir, '%04d' % image_id + '_pred.jpg'), image)