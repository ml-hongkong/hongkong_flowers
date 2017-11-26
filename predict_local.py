import sys
import os
import os.path as osp
import shutil
import base64
import tensorflow as tf
import numpy as np
import keras

from tensorflow.contrib.session_bundle import exporter
from keras import backend as K
import numpy as np
from options import Options

trained_model = 'model_best_weights.h5'
model_dir = './checkpoints/2017-10-04_experiment_0/'

height = 244
width = 244
channels = 3

def preprocess_input(tf_image):
    r, g, b = tf.split(tf_image, num_or_size_splits=3, axis=3)

    # Zero-center by mean pixel
    b -= 103.939
    g -= 116.779
    r -= 123.68

    # 'RGB'->'BGR'
    data = tf.concat([b, g, r], 3)

    return data

def decode_and_resize(image_str_tensor):
  """Decodes jpeg string, resizes it and returns a uint8 tensor."""
  image = tf.image.decode_jpeg(image_str_tensor, channels=channels)
  # Note resize expects a batch_size, but tf_map supresses that index,
  # thus we have to expand then squeeze.  Resize returns float32 in the
  # range [0, uint8_max]
  image = tf.expand_dims(image, 0)
  image = tf.image.resize_bilinear(
      image, [height, width], align_corners=False)
  image = tf.squeeze(image, squeeze_dims=[0])
  image = tf.cast(image, dtype=tf.uint8)
  return image

if __name__ == '__main__':
    K.set_learning_phase(bool(0))

    classes = [i for i in os.listdir('./dataset/train/') if os.path.isdir('./dataset/train/' + i)]

    tf_model_path = osp.join(model_dir, 'export')

    image_str_tensor = tf.placeholder(tf.string, shape=[None])

    image = tf.map_fn(
        decode_and_resize, image_str_tensor, back_prop=False, dtype=tf.uint8)
    # convert_image_dtype, also scales [0, uint8_max] -> [0 ,1).

    image = tf.cast(image, dtype=tf.float32)
    # image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # setting up, options contains all our params
    options = Options(library=0,    # use keras
                      configs=2,    # use resnet50 model
                      transform=1)  # use transform for resnet50

    # load the weight file
    options.load = True
    options.load_file = osp.join(model_dir, trained_model)

    input_tensor = preprocess_input(image)

    # initialize model
    model = options.initializeModel(input_tensor=input_tensor)

    my_model = model.model

    keys_placeholder = tf.placeholder(tf.string, shape=[None])

    img = open(sys.argv[1], 'rb')  # this is a PIL image
    image_base64 = img.read()

    with K.get_session() as sess:
        input_tensor_arr, outputs = sess.run([input_tensor, my_model.output], feed_dict={
            image_str_tensor: [image_base64]
        })
        preds = outputs[0]
        pred = np.argmax(preds)
        print(preds[pred])
        print(pred, classes[pred])
