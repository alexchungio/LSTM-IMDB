#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : dataset_pipeline.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/9/8 上午11:07
# @ Software   : PyCharm
#-------------------------------------------------------


import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
from tensorflow.keras import preprocessing
from libs.configs import cfgs


data_dir = '/home/alex/Documents/dataset/flower_photos'



def dataset_batch(x, y, batch_size=32, epoch=None, is_training=False):
    """
    create dataset iterator
    :param data_dir:
    :param batch_size:
    :param epoch:
    :param class_name:
    :param img_shape:
    :param label_depth:
    :param convert_scale:
    :return:
    """

    # create dataset slice
    dataset = tf.data.Dataset.from_tensor_slices((x, y))

    # shuffle batch_size epoch
    if is_training:
        dataset = dataset.shuffle(buffer_size=batch_size * 4)

    # get batch
    dataset = dataset.batch(batch_size).repeat(epoch)
    # lets the dataset fetch batches in the background while the model is training.
    dataset = dataset.prefetch(buffer_size=batch_size * 10)

    # return iterator
    return dataset.make_one_shot_iterator()




if __name__ == "__main__":


    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=cfgs.FEATURE_SIZE)
    x_train = preprocessing.sequence.pad_sequences(x_train, cfgs.MAX_LENGTH)
    y_train = np.asarray(y_train).astype(np.float32)
    x_test = preprocessing.sequence.pad_sequences(x_test, cfgs.MAX_LENGTH)
    y_train = np.asarray(y_train).astype(np.float32)

    dataset_iterator = dataset_batch(x_train, y_train, batch_size=32)
    train_data_batch, train_label_batch = dataset_iterator.get_next()

    with tf.Session() as sess:
        for _ in range(10):
            # print(sess.run('vgg_16/conv1/conv1_1/biases:0'))
            train_data, train_label = sess.run([train_data_batch, train_label_batch])

            print(train_data)
