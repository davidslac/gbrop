from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import project_setup

import tensorflow as tf

def imgbatch_2_vgg16(imgs, channel_mean=8.46):
    imgs = tf.to_float(imgs)
    imgs = tf.tile(imgs, [1,1,1,3])
    resized = tf.image.resize_images(imgs, 224, 224, method=tf.image.ResizeMethod.BICUBIC)
    resized -= channel_mean
    return resized
