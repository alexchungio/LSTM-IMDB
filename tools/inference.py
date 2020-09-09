#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : inference.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/9/8 下午3:25
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras import preprocessing
import json

from libs.configs import cfgs

dataset_dir = '../data/aclImdb/train'

NUM_SAMPLES = 10

# get test samples
samples_text = []
samples_label = []


for index, type in enumerate(['neg', 'pos']):
    type_dir = os.path.join(dataset_dir, type)
    for file in os.listdir(type_dir)[:NUM_SAMPLES]:
        with open(os.path.join(type_dir, file)) as f:
            samples_text.append(f.read())
            samples_label.append(index)

def encoder_text(texts, word_index, num_words, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True, split=" "):
    sequences = []
    for text in texts:
        sequence = []
        if lower:
            text = text.lower()
        translate_dict = dict((c, split) for c in filters)
        translate_map = str.maketrans(translate_dict)
        text = text.translate(translate_map)
        for word in text.split(split):
            index = word_index.get(word)
            if index is not None and index < num_words:
                sequence.append(index)
            else:
                continue
        sequences.append(sequence)
    return sequences


def sequence_padding(sequences, max_length, padding='pre', truncating='pre', value=0.):

    num_samples = len(sequences)

    seq_padding = np.full(shape=(num_samples, max_length), fill_value=value, dtype=np.int32)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-max_length:]
        elif truncating == 'post':
            trunc = s[:max_length]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

        if padding == 'post':
            seq_padding[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            seq_padding[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return seq_padding

if __name__ == "__main__":
    # word_index = imdb.get_word_index()
    with open(cfgs.WORD_INDEX) as f:
        word_index = json.loads(f.read())

    word_sequence = encoder_text(samples_text, word_index=word_index, num_words=cfgs.FEATURE_SIZE)
    pad_sequence = sequence_padding(sequences=word_sequence, max_length=cfgs.MAX_LENGTH)

    # use train dataset to train and validation model
    x_dataset = pad_sequence
    y_dataset = samples_label
    # turn the samples into 2D tensor of shape (num_samples, max_length)
    y_dataset = np.asarray(y_dataset).astype(np.float32)
    # x_test = preprocessing.sequence.pad_sequences(x_test, cfgs.MAX_LENGTH)
    # y_train = np.asarray(y_train).astype(np.float32)

    num_val_samples = int(np.floor(len(x_dataset) * cfgs.SPLIT_RATIO))
    num_train_samples = len(x_dataset) - num_val_samples

    # shuffle dataset
    indices = np.arange(len(x_dataset))
    np.random.shuffle(indices)
    x_dataset = x_dataset[indices]
    y_dataset = y_dataset[indices]

    # split dataset
    # x_train, y_train = x_dataset[:num_train_samples], y_dataset[:num_train_samples]
    x_val, y_val = x_dataset[-10:], y_dataset[-10:]

    saver = tf.train.import_meta_graph(os.path.join(cfgs.TRAINED_CKPT, 'model_loss=0.075851.ckpt-7653.meta'))

    latest_ckpt = tf.train.latest_checkpoint(cfgs.TRAINED_CKPT)
    init_op = tf.group(tf.local_variables_initializer(),
                       tf.global_variables_initializer())

    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, latest_ckpt)
        for var in tf.model_variables():
            print(var.op.name)
            print(var.shape)

        graph = tf.get_default_graph()

        tensor_name_list = [tensor.name for tensor in graph.as_graph_def().node]
        input_data = graph.get_tensor_by_name('input_data:0')
        keep_prob = graph.get_tensor_by_name('keep_prob:0')
        predict = graph.get_tensor_by_name('predict:0')

        feed_dict = {
            input_data: x_val,
            keep_prob: 1.0
        }

        predict = sess.run(predict, feed_dict=feed_dict)
        print(predict)
        print(y_val)





