
import tensorflow as tf
import os
from tfs import *
from scipy.misc import imread
import matplotlib.pyplot as plt
import json

class ImageNet(object):

    def input_tensor(self, path, name):

        paths = []
        labels  = []
        class_folders = sorted(os.listdir(path))
        label = 0
        class_names = json.load(open('classes.json'))
        for c in class_folders:
            image_names = os.listdir(path + c)
            for i in image_names:
                paths.append(path + c + '/' + i)
                labels.append(label)
            label += 1

        print('processed paths')

        with tf.variable_scope(name):

            paths = tf.convert_to_tensor(paths, dtype=tf.string)
            labels = tf.convert_to_tensor(labels, dtype=tf.int32)

            q = tf.train.slice_input_producer([paths, labels], shuffle=True, capacity= 2*FLAGS.B)
            images = tf.image.decode_jpeg(tf.read_file(q[0]), channels=3)
            labels = q[1]
            images = tf.image.resize_images(images, [224, 224], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            images.set_shape([224, 224, 3])
            images = tf.cast(images, dtype = tf.float32)
            images, labels = tf.train.batch([images, labels], 
                                        batch_size = FLAGS.B, 
                                        capacity=2*FLAGS.B, 
                                        allow_smaller_final_batch= True,
                                        num_threads=FLAGS.threads)

            return { 'images' : images, 'labels' : tf.one_hot(labels, 1000) }

    def __init__(self, train_path, val_path, class_names):
        self.train  = self.input_tensor(train_path, "train_input")
        self.val    = self.input_tensor(val_path, "val_input")
        self.class_names =  json.load(open(class_names))