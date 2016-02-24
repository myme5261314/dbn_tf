#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Peng Liu <liupeng@imscv.com>
#
# Distributed under terms of the GNU GPL3 license.

"""
This file implement class RBM for tensorflow library.
"""
import math
import tensorflow as tf
import numpy as np
import Image
from util import tile_raster_images


class RBM(object):

    """RBM class for tensorflow"""

    def __init__(self, name, input_size, output_size, opts):
        """Initialize a rbm object.

        :name: TODO
        :input_size: TODO
        :output_size: TODO

        """
        self._name = name
        self._input_size = input_size
        self._output_size = output_size
        self._opts = opts
        with tf.name_scope("rbm_" + name):
            self._w = tf.placeholder("float", [input_size, output_size])
            self._hb = tf.placeholder("float", [output_size])
            self._vb = tf.placeholder("float", [input_size])
            self.init_w = np.zeros([input_size, output_size], np.float32)
            self.init_hb = np.zeros([output_size], np.float32)
            self.init_vb = np.zeros([input_size], np.float32)
            self.w = np.zeros([input_size, output_size], np.float32)
            self.hb = np.zeros([output_size], np.float32)
            self.vb = np.zeros([input_size], np.float32)

    def reset_init_parameter(self, init_weights, init_hbias, init_vbias):
        """TODO: Docstring for reset_para.

        :init_weights: TODO
        :init_hbias: TODO
        :init_vbias: TODO
        :returns: TODO

        """
        self.init_w = init_weights
        self.init_hb = init_hbias
        self.init_vb = init_vbias

    def propup(self, visible):
        """TODO: Docstring for propup.

        :visible: TODO
        :returns: TODO

        """
        return tf.nn.sigmoid(tf.matmul(visible, self._w) + self._hb)

    def propdown(self, hidden):
        """TODO: Docstring for propdown.

        :hidden: TODO
        :returns: TODO

        """
        return tf.nn.sigmoid(
            tf.matmul(hidden, tf.transpose(self._w)) + self._vb)

    def sample_prob(self, probs):
        """TODO: Docstring for sample_prob.

        :probs: TODO
        :returns: TODO

        """
        return tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs))))

    def train(self, X):
        """TODO: Docstring for train.

        :X: TODO
        :returns: TODO

        """
        v0 = tf.placeholder("float", [None, self._input_size])
        h0 = self.sample_prob(self.propup(v0))
        v1 = self.sample_prob(self.propdown(h0))
        h1 = self.propup(v1)
        positive_grad = tf.matmul(tf.transpose(v0), h0)
        negative_grad = tf.matmul(tf.transpose(v1), h1)
        update_w = self._w * self._opts._momontum + self._opts._learning_rate *\
            (positive_grad - negative_grad) / tf.to_float(tf.shape(v0)[0])
        update_vb = self._vb * self._opts._momontum + \
            self._opts._learning_rate * tf.reduce_mean(v0 - v1, 0)
        update_hb = self._hb * self._opts._momontum + \
            self._opts._learning_rate * tf.reduce_mean(h0 - h1, 0)
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            old_w = self.init_w
            old_hb = self.init_hb
            old_vb = self.init_vb
            for i in range(self._opts._epoches):
                for start, end in zip(range(0, len(X), self._opts._batchsize),
                                      range(self._opts._batchsize,
                                            len(X), self._opts._batchsize)):
                    batch = X[start:end]
                    self.w = sess.run(update_w, feed_dict={
                        v0: batch, self._w: old_w, self._hb: old_hb,
                        self._vb: old_vb})
                    self.hb = sess.run(update_hb, feed_dict={
                        v0: batch, self._w: old_w, self._hb: old_hb,
                        self._vb: old_vb})
                    self.vb = sess.run(update_vb, feed_dict={
                        v0: batch, self._w: old_w, self._hb: old_hb,
                        self._vb: old_vb})
                    old_w = self.w
                    old_hb = self.hb
                    old_vb = self.vb
                image = Image.fromarray(
                    tile_raster_images(
                        X=self.w.T,
                        img_shape=(int(math.sqrt(self._input_size)),
                                   int(math.sqrt(self._input_size))),
                        tile_shape=(int(math.sqrt(self._output_size)),
                                    int(math.sqrt(self._output_size))),
                        tile_spacing=(1, 1)
                    )
                )
                image.save("%s_%d.png" % (self._name, i))

    def rbmup(self, X):
        """TODO: Docstring for rbmup.

        :X: TODO
        :returns: TODO

        """
        input_X = tf.placeholder("float", [None, self._input_size])
        out = tf.nn.sigmoid(tf.matmul(input_X, self._w) + self._hb)
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            return sess.run(out, feed_dict={
                input_X: X, self._w: self.w, self._hb: self.hb})
