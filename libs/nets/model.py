#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : model.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/9/7 下午2:45
# @ Software   : PyCharm
#-------------------------------------------------------

#  add  from tensorflow.python.ops.rnn import dynamic_rnn to tensorflow/nn/__init__.py

import tensorflow.compat.v1 as tf
from libs.configs import cfgs

class LSTM(object):
    def __init__(self, input_length=28, time_steps=28, feature_size=10000, embedding_size=64, num_layers=3, num_units=None,
                 num_outputs = 1):

        self.input_length = input_length  # sequence length, response to time_steps
        self.feature_size = feature_size
        self.embedded_size = embedding_size
        self.time_steps= time_steps
        self.num_layers = num_layers
        self.num_units = num_units
        self.num_outputs = num_outputs
        assert num_layers == len(num_units), "the number of units must equal to number layers"

        self.input_data = tf.placeholder(shape=(None, input_length), dtype=tf.int32, name="input_data")
        self.input_target = tf.placeholder(shape=(None, num_outputs), dtype=tf.int32, name="input_target")
        self.keep_prob = tf.placeholder(shape=(), dtype=tf.float32, name="keep_prob")

        self.global_step = tf.train.get_or_create_global_step()
        self.inference = self.forward()
        self.loss = self.losses()
        self.acc = self.accuracy()
        self.train = self.training()

    def forward(self):

        # embedding layer
        self.encode_outputs = tf.nn.embedding_lookup(tf.Variable(tf.random.uniform([self.feature_size, self.embedded_size], -1, 1), name="embedding"),
                                                       ids=self.input_data)

        # rnn cell
        cells = [tf.nn.rnn_cell.DropoutWrapper(self.get_lstm_cell(self.num_units[i]),
                                               input_keep_prob=self.keep_prob,
                                               output_keep_prob=self.keep_prob)
                 for i in range(self.num_layers)]

        lstm_cells = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
        # print(rnn_cells.state_size)  # (64, 128, 256)
        # initial_state = rnn_cells.zero_state(BATCH_SIZE, dtype=tf.float32)

        lstm_outputs, lstm_states = tf.nn.dynamic_rnn(cell=lstm_cells, inputs=self.encode_outputs, dtype=tf.float32, scope="lstm")


        self.logits = self.fully_connected(inputs=lstm_outputs[:, -1, :], output_size=self.num_outputs)
        self.predict = tf.sigmoid(self.logits, name="predict")

    def fill_feed_dict(self, input_data, input_target=None,  keep_prob=1.0):

        feed_dict = {
            self.input_data: input_data,
            self.input_target: input_target,
            self.keep_prob: keep_prob
        }
        return feed_dict

    def accuracy(self):

        predict = tf.cast(tf.greater(self.predict, 0.5), tf.int32)
        acc_mask = tf.equal(predict, self.input_target)
        acc = tf.reduce_mean(tf.cast(acc_mask, dtype=tf.float32))
        tf.summary.scalar("acc", acc)

        return acc

    def losses(self):
        with tf.variable_scope("loss"):
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,
                                                                    labels=tf.cast(self.input_target, tf.float32),
                                                                    name='entropy')
            loss = tf.reduce_mean(input_tensor=cross_entropy, name='entropy_mean')
            tf.summary.scalar("loss", loss)
            return loss


    def training(self):
            global_step_update = tf.assign_add(self.global_step, 1)
            with tf.control_dependencies([global_step_update]):
                return tf.train.AdamOptimizer(learning_rate=cfgs.LEARNING_RATE).minimize(self.loss)


    def get_lstm_cell(self, num_units=128, activation='tanh'):
        return tf.nn.rnn_cell.BasicLSTMCell(num_units=num_units, activation=activation)

    def fully_connected(self, inputs, output_size, scope="full_connected", is_activation=None):

        # get feature num
        shape = inputs.get_shape().as_list()
        # convolution layer
        if len(shape) == 4:
            input_size = shape[-1] * shape[-2] * shape[-3]
        # dense layers
        else:
            input_size = shape[1]

        with tf.variable_scope(name_or_scope=scope):
            flat_data = tf.reshape(tensor=inputs, shape=[-1, input_size], name='flatten')

            weights = tf.get_variable(shape=(input_size, output_size), initializer=tf.orthogonal_initializer(), name='W')

            biases = tf.get_variable('b', shape=(output_size), initializer=tf.zeros_initializer())

            if is_activation is not None:
                return tf.nn.relu_layer(x=input_size, weights=weights, biases=biases)
            else:
                return tf.nn.bias_add(value=tf.matmul(flat_data, weights), bias=biases)