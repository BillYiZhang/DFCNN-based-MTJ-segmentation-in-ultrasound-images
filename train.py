from __future__ import print_function

import os
import tensorflow as tf
from skimage.transform import resize
from skimage.io import imsave, imread
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras import backend as K
from keras import regularizers
import matplotlib.pyplot as plt

from random_dataset import load_train_data, load_test_data

data_path = 'preds_mask/'

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

images_row = 390
images_col = 482

img_rows = 384
img_cols = 384
k_fold = 1

smooth = 1.

more_training = 0
layer_visualization = 0

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_unet():
    inputs = Input((img_rows, img_cols, 1))
    conv1 = Conv2D(32, (5, 5), activation='relu', padding='same', name='conv1')(inputs)
    conv1 = Dropout(0.5)(conv1)
    conv1 = Conv2D(32, (5, 5), activation='relu', padding='same', name='conv2')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (5, 5), activation='relu', padding='same', name='conv3')(pool1)
    conv2 = Dropout(0.5)(conv2)
    conv2 = Conv2D(64, (5, 5), activation='relu', padding='same', name='conv4')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (5, 5), activation='relu', padding='same', name='conv5')(pool2)
    conv3 = Dropout(0.5)(conv3)
    conv3 = Conv2D(128, (5, 5), activation='relu', padding='same', name='conv6')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (5, 5), activation='relu', padding='same', name='conv7')(pool3)
    conv4 = Dropout(0.5)(conv4)
    conv4 = Conv2D(256, (5, 5), activation='relu', padding='same', name='conv8')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (5, 5), activation='relu', padding='same', name='conv9')(pool4)
    conv5 = Dropout(0.5)(conv5)
    conv5 = Conv2D(512, (5, 5), activation='relu', padding='same', name='conv10')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same', name='conv11')(conv5), conv4],
                      axis=3)
    conv6 = Conv2D(256, (5, 5), activation='relu', padding='same', name='conv12')(up6)
    conv6 = Dropout(0.5)(conv6)
    conv6 = Conv2D(256, (5, 5), activation='relu', padding='same', name='conv13')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', name='conv14')(conv6), conv3],
                      axis=3)
    conv7 = Conv2D(128, (5, 5), activation='relu', padding='same', name='conv15')(up7)
    conv7 = Dropout(0.5)(conv7)
    conv7 = Conv2D(128, (5, 5), activation='relu', padding='same', name='conv16')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', name='conv17')(conv7), conv2],
                      axis=3)
    conv8 = Conv2D(64, (5, 5), activation='relu', padding='same', name='conv18')(up8)
    conv8 = Dropout(0.5)(conv8)
    conv8 = Conv2D(64, (5, 5), activation='relu', padding='same', name='conv19')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', name='conv20')(conv8), conv1],
                      axis=3)
    conv9 = Conv2D(32, (5, 5), activation='relu', padding='same', name='conv21')(up9)
    conv9 = Dropout(0.5)(conv9)
    conv9 = Conv2D(32, (5, 5), activation='relu', padding='same', name='conv22')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid', name='conv23')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-5), loss=focal_loss, metrics=[dice_coef])

    return model


def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_cols, img_rows), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p


def focal_loss(y_true, y_pred):
    #defone the hyperparameters
    alpha = 0.25
    gamma = 2.0

    # filter out "ignore" anchors
    anchor_state = K.max(y_true, axis=0)  # -1 for ignore, 0 for background, 1 for object
    indices = tf.where(K.not_equal(anchor_state, -1))
    y_true = tf.gather_nd(y_true, indices)
    y_pred = tf.gather_nd(y_pred, indices)

    # compute the focal loss
    alpha_factor = K.ones_like(y_true) * alpha
    alpha_factor = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
    focal_weight = tf.where(K.equal(y_true, 1), 1 - y_pred, y_pred)
    focal_weight = alpha_factor * focal_weight ** gamma

    cls_loss = focal_weight * K.binary_crossentropy(y_true, y_pred)

    # compute the normalizer: the number of positive anchors
    normalizer = tf.where(K.equal(anchor_state, 1))
    normalizer = K.cast(K.shape(normalizer)[0], K.floatx())
    normalizer = K.maximum(1.0, normalizer)
    return K.sum(cls_loss) / normalizer


def train_and_predict():


    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)

    imgs_train, imgs_mask_train = load_train_data()

    #debug
    #monitor_image = imgs_train[100,:,:]
    #monitor_mask = imgs_mask_train[100,:,:]
    #plt.imshow(monitor_image, cmap='Greys_r')
    #plt.show()
    #plt.imshow(monitor_mask, cmap='Greys_r')
    #plt.show()

    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]

    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)
    model = get_unet()
    model_checkpoint = ModelCheckpoint('weights_fl.h5', monitor='val_loss', save_best_only=True)
    model_earlystopping = EarlyStopping(monitor='val_loss', patience=20)

    if more_training == 1:
        print('-' * 30)
        print('go on training...')
        print('-' * 30)
        model.load_weights('weights_fl.h5')#用于继续训练

    if layer_visualization == 1:
        print('-' * 30)
        print('See the intermediate outout...')
        print('-' * 30)
        specific_layer_input = model.input
        specific_layer_output = model.get_layer('conv10').output
        specific_layer_model = Model(inputs=[specific_layer_input], outputs=[specific_layer_output])
        specific_layer_model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    #交叉验证下的训练（伪的）
    training_length = imgs_train.shape[0]
    for k_count in range(0, k_fold):
        rand_order = np.random.randint(0, training_length, training_length)#生成随机索引号，近似为一种shuffle方式（缺点是很有可能遗漏一些数据没有被训练）
        for order_count in range(0, training_length):
            imgs_train[order_count] = imgs_train[[rand_order[order_count]]]
            imgs_mask_train[order_count] = imgs_mask_train[[rand_order[order_count]]]

        print('-'*30)
        print('Fitting model...')
        print('-'*30)
        model.fit(imgs_train, imgs_mask_train, batch_size=4, nb_epoch=299, verbose=1, shuffle=True, validation_split=0.2, callbacks=[model_checkpoint, model_earlystopping, TensorBoard(log_dir='visualization')])

    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    imgs_test, imgs_id_test = load_test_data()
    imgs_test = preprocess(imgs_test)

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights('weights_fl.h5')

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    imgs_mask_test = model.predict(imgs_test, verbose=1, batch_size=4)
    np.save('imgs_mask_test.npy', imgs_mask_test)

    # 给出测试集最终的正确率
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

    imgs_mask_test_manual = preprocess(imgs_mask_test_manual)
    imgs_mask_test_manual = imgs_mask_test_manual.astype('float32')
    imgs_mask_test_manual /= 255.  # scale masks to [0, 1]

    test_accuracy_vec = []
    test_len = imgs_mask_test.shape[0]
    for test_c in range(0, test_len):
        intersection = np.sum(imgs_mask_test_manual[test_c, :, :, 0] * imgs_mask_test[test_c, :, :, 0])
        test_accuracy_vec.append((2 * intersection) / (np.sum(imgs_mask_test_manual[test_c, :, :, 0]) + np.sum(imgs_mask_test[test_c, :, :, 0])))
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
        imsave(os.path.join(pred_dir, '%04d'% image_id + '_pred.jpg'), image)

if __name__ == '__main__':
    train_and_predict()
