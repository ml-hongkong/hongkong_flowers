import glob
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

for my_class in glob.glob("./data/test/**"):
    save_to_dir = "./data/test_aug/" + os.path.basename(my_class)

    try:
        os.mkdir(save_to_dir)
    except:
        pass

    for filename in glob.glob(my_class + "/*.jpg"):
        img = load_img(filename)  # this is a PIL image
        x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
        x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
        datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')
        datagen.mean = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape((3, 1, 1))

        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=save_to_dir, save_format='jpg'):
            i += 1
            if i > 5:
                break
