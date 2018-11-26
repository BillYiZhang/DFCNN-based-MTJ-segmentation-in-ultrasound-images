import os
data_path = 'test/'
from skimage.io import imsave, imread

def rename_data():
    test_data_path = os.path.join(data_path)
    images = os.listdir(test_data_path)
    image_id = 1
    for image_name in images:
        img = imread(os.path.join(data_path, image_name), as_grey=True)
        imsave(os.path.join(data_path, '%01d'%image_id + '.jpg'), img)
        image_id = image_id + 1

if __name__ == '__main__':
    rename_data()