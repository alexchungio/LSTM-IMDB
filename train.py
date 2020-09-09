#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : train.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/9/7 下午2:34
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import numpy as np
import tensorflow.compat.v1 as tf
from tqdm import tqdm
import json
from tensorflow.keras.datasets import imdb
from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer

from libs.configs import cfgs
from data.dataset_pipeline import dataset_batch
from libs.nets.model import LSTM

imdb_dir = './data/aclImdb'
glove_dir = './data/glove.6B'
train_dir = os.path.join(imdb_dir, 'train')

def main(argv):

    # -------------------load dataset-------------------------------------------
    data_text = []
    data_label = []
    for index, label_type in enumerate(['neg', 'pos']):
        type_dir = os.path.join(train_dir, label_type)
        for filename in os.listdir(type_dir):
            if filename.split('.')[-1] == 'txt':
                with open(os.path.join(type_dir, filename)) as f:
                    data_text.append(f.read())
                data_label.append(index)

    tokenizer = Tokenizer(num_words=cfgs.FEATURE_SIZE)
    tokenizer.fit_on_texts(data_text)
    # save word index
    word_index = tokenizer.word_index
    with open(cfgs.WORD_INDEX, 'w') as f:
        f.write(json.dumps(word_index))
    print('Found uique token {0}'.format(len(word_index)))
    sequence = tokenizer.texts_to_sequences(data_text)
    # pad squeence
    # pad_sequence = preprocessing.sequence.pad_sequences(sequence, max_length)
    # max_index = max([max(seq) for seq in data])
    #
    # (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=cfgs.FEATURE_SIZE)
    # word_index = imdb.get_word_index()
    # word_index = dict(sorted(word_index.items(), key=lambda kv: (kv[1], kv[0]))) # sort word index
    # use train dataset to train and validation model
    x_dataset = sequence
    y_dataset = data_label
    # turn the samples into 2D tensor of shape (num_samples, max_length)
    x_dataset = preprocessing.sequence.pad_sequences(x_dataset, cfgs.MAX_LENGTH)
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
    x_train, y_train = x_dataset[:num_train_samples], y_dataset[:num_train_samples]
    x_val, y_val = x_dataset[num_train_samples:], y_dataset[num_train_samples:]

    # --------------------- construct model------------------------------------------
    model = LSTM(input_length=cfgs.MAX_LENGTH, feature_size=cfgs.FEATURE_SIZE, embedding_size= cfgs.EMBEDDING_SIZE,
                 num_layers=cfgs.NUM_LAYERS, num_units=cfgs.NUM_UNITS)

    saver = tf.train.Saver(max_to_keep=30)

    # get computer graph
    graph = tf.get_default_graph()

    write = tf.summary.FileWriter(logdir=cfgs.SUMMARY_PATH, graph=graph)

    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5  # maximun alloc gpu50% of MEM
    config.gpu_options.allow_growth = True

    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )
    # train and save model
    with tf.Session(config=config) as sess:
        sess.run(init_op)

        # get model variable of network
        model_variable = tf.model_variables()
        for var in model_variable:
            print(var.op.name, var.shape)

        #------------------load embedding pretrained weights---------------------
        # parse glove pretrained model
        if cfgs.EMBEDDING_TRANSFER:
            embedding_index = {}
            with open(os.path.join(glove_dir, 'glove.6B.100d.txt')) as f:
                for line in f:
                    value = line.split()
                    word = value[0]
                    coeff = value[1:]
                    embedding_index[word] = coeff

            embedding_matrix = np.zeros(shape=(cfgs.FEATURE_SIZE, cfgs.EMBEDDING_SIZE))
            for word, index in word_index.items():
                if index < cfgs.FEATURE_SIZE:
                    embedding_vector = embedding_index.get(word)
                    if embedding_vector is not None:
                        embedding_matrix[index] = embedding_vector

            embedding_variable = tf.global_variables(scope='embedding')
            tf.assign(embedding_variable[0], tf.convert_to_tensor(embedding_matrix, dtype=tf.float32))
            print('+++++++++++++++++++++Successful load glove embedding weights+++++++++++++++++++++++')
        # -----------------------train part------------------------------------------------
        # merges all summaries collected in the default graph
        summary_op = tf.summary.merge_all()

        train_step_per_epoch = num_train_samples // cfgs.BATCH_SIZE
        test_step_pre_epoch = num_val_samples // cfgs.BATCH_SIZE

        # generate batch
        train_dataset = dataset_batch(x_train, y_train, batch_size=cfgs.BATCH_SIZE, is_training=True)
        val_dataset = dataset_batch(x_val, y_val, batch_size=cfgs.BATCH_SIZE, is_training=False)
        train_data_batch, train_label_batch = train_dataset.get_next()
        val_data_batch, val_label_batch = val_dataset.get_next()
        # use k folder validation
        for epoch in range(cfgs.NUM_EPOCH):
            train_bar = tqdm(range(1, train_step_per_epoch+1))
            for step in train_bar:
                x_train, y_train = sess.run([train_data_batch, train_label_batch])
                y_train = y_train[:, np.newaxis]
                feed_dict = model.fill_feed_dict(x_train, y_train, keep_prob=cfgs.KEEP_PROB)
                summary, global_step, train_loss, train_acc, _ = sess.run([summary_op, model.global_step, model.loss, model.acc, model.train],
                                                                          feed_dict=feed_dict)
                if step % cfgs.SMRY_ITER == 0:
                    write.add_summary(summary=summary, global_step=global_step)
                    write.flush()

                train_bar.set_description("Epoch {0} : Step {1} => Train Loss: {2:.4f} | Train ACC: {3:.4f}".
                                          format(epoch, step, train_loss, train_acc))
            test_loss_list = []
            test_acc_list = []
            for step in range(test_step_pre_epoch):
                x_test, y_test = sess.run([val_data_batch, val_label_batch])
                y_test = y_test[:, np.newaxis]
                feed_dict = model.fill_feed_dict(x_test, y_test, keep_prob=1.0)

                test_loss, test_acc, _ = sess.run([model.loss, model.acc, model.train], feed_dict=feed_dict)
                test_loss_list.append(test_loss)
                test_acc_list.append(test_acc)
            test_loss = sum(test_loss_list) / len(test_loss_list)
            test_acc = sum(test_acc_list) / len(test_acc_list)
            print("Epoch {0} : Step {1} => Val Loss: {2:.4f} | Val ACC: {3:.4f} ".format(epoch, step,
                                                                                             test_loss, test_acc))
            ckpt_file = os.path.join(cfgs.TRAINED_CKPT, 'model_loss={0:4f}.ckpt'.format(test_loss))
            saver.save(sess=sess, save_path=ckpt_file, global_step=global_step)
    sess.close()
    print('model training has complete')



if __name__ == "__main__":

    tf.app.run()