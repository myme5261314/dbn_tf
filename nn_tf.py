#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Peng Liu <liupeng@imscv.com>
#
# Distributed under terms of the GNU GPL3 license.

"""
This file implement a class NN for Backward propragation Neural Network.
"""
import numpy as np
import math
import tensorflow as tf


class NN(object):

    """Docstring for NN. """

    def __init__(self, sizes, opts, X, Y):
        """TODO: to be defined1.

        :sizes: TODO
        :opts: TODO
        :X: TODO

        """
        self._sizes = sizes
        self._opts = opts
        self._X = X
        self._Y = Y
        self.w_list = []
        self.b_list = []
        input_size = X.shape[1]
        for size in self._sizes + [Y.shape[1]]:
            max_range = 4 * math.sqrt(6. / (input_size + size))
            self.w_list.append(
                np.random.uniform(
                    -max_range, max_range, [input_size, size]
                ).astype(np.float32))
            self.b_list.append(np.zeros([size], np.float32))
            input_size = size

    def load_from_dbn(self, dbn):
        """TODO: Docstring for load_from_dbn.

        :dbn: TODO
        :returns: TODO

        """
        assert len(dbn._sizes) == len(self._sizes)
        for i in range(len(self._sizes)):
            assert dbn._sizes[i] == self._sizes[i]
        for i in range(len(self._sizes)):
            self.w_list[i] = dbn.rbm_list[i].w
            self.b_list[i] = dbn.rbm_list[i].hb

    def train(self):
        """TODO: Docstring for train.
        :returns: TODO

        """
        _a = [None] * (len(self._sizes) + 2)
        _w = [None] * (len(self._sizes) + 1)
        _b = [None] * (len(self._sizes) + 1)
        _a[0] = tf.placeholder("float", [None, self._X.shape[1]])
        y = tf.placeholder("float", [None, self._Y.shape[1]])
        for i in range(len(self._sizes) + 1):
            _w[i] = tf.Variable(self.w_list[i])
            _b[i] = tf.Variable(self.b_list[i])
        for i in range(1, len(self._sizes) + 2):
            _a[i] = tf.nn.sigmoid(tf.matmul(_a[i - 1], _w[i - 1]) + _b[i - 1])
        cost = tf.reduce_mean(tf.square(_a[-1] - y))
        train_op = tf.train.MomentumOptimizer(
            self._opts._learning_rate, self._opts._momentum).minimize(cost)
        predict_op = tf.argmax(_a[-1], 1)
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for i in range(self._opts._epoches):
                for start, end in zip(
                    range(
                        0, len(self._X),
                        self._opts._batchsize),
                    range(
                        self._opts._batchsize, len(
                            self._X),
                        self._opts._batchsize)):
                    sess.run(train_op, feed_dict={
                        _a[0]: self._X[start:end], y: self._Y[start:end]})
                for i in range(len(self._sizes) + 1):
                    self.w_list[i] = sess.run(_w[i])
                    self.b_list[i] = sess.run(_b[i])
                print np.mean(np.argmax(self._Y, axis=1) ==
                              sess.run(predict_op, feed_dict={
                                  _a[0]: self._X, y: self._Y}))

    def predict(self, X):
        """TODO: Docstring for predict.

        :X: TODO
        :returns: TODO

        """
        _a = [None] * (len(self._sizes) + 2)
        _w = [None] * len(self.w_list)
        _b = [None] * len(self.b_list)
        _a[0] = tf.placeholder("float", [None, self._X.shape[1]])
        for i in range(len(self.w_list)):
            _w[i] = tf.constant(self.w_list[i])
            _b[i] = tf.constant(self.b_list[i])
        for i in range(1, len(self._sizes) + 2):
            _a[i] = tf.nn.sigmoid(tf.matmul(_a[i - 1], _w[i - 1]) + _b[i - 1])
        predict_op = tf.argmax(_a[-1], 1)
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            return sess.run(predict_op, feed_dict={_a[0]: X})
