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

    def propup(self, visible, w, hb):
        """TODO: Docstring for propup.

        :visible: TODO
        :returns: TODO

        """
        return tf.nn.sigmoid(tf.matmul(visible, w) + hb)

    def propdown(self, hidden, w, vb):
        """TODO: Docstring for propdown.

        :hidden: TODO
        :returns: TODO

        """
        return tf.nn.sigmoid(
            tf.matmul(hidden, tf.transpose(w)) + vb)

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
        _w = tf.placeholder("float", [self._input_size, self._output_size])
        _hb = tf.placeholder("float", [self._output_size])
        _vb = tf.placeholder("float", [self._input_size])
        _vw = tf.placeholder("float", [self._input_size, self._output_size])
        _vhb = tf.placeholder("float", [self._output_size])
        _vvb = tf.placeholder("float", [self._input_size])
        _current_vw = np.zeros(
            [self._input_size, self._output_size], np.float32)
        _current_vhb = np.zeros([self._output_size], np.float32)
        _current_vvb = np.zeros([self._input_size], np.float32)
        v0 = tf.placeholder("float", [None, self._input_size])
        h0 = self.sample_prob(self.propup(v0, _w, _hb))
        v1 = self.sample_prob(self.propdown(h0, _w, _vb))
        h1 = self.propup(v1, _w, _hb)
        positive_grad = tf.matmul(tf.transpose(v0), h0)
        negative_grad = tf.matmul(tf.transpose(v1), h1)
        update_vw = _vw * self._opts._momentum + self._opts._learning_rate *\
            (positive_grad - negative_grad) / tf.to_float(tf.shape(v0)[0])
        update_vvb = _vvb * self._opts._momentum + \
            self._opts._learning_rate * tf.reduce_mean(v0 - v1, 0)
        update_vhb = _vhb * self._opts._momentum + \
            self._opts._learning_rate * tf.reduce_mean(h0 - h1, 0)
        update_w = _w + _vw
        update_vb = _vb + _vvb
        update_hb = _hb + _vhb
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
                    _current_vw = sess.run(update_vw, feed_dict={
                        v0: batch, _w: old_w, _hb: old_hb, _vb: old_vb,
                        _vw: _current_vw})
                    _current_vhb = sess.run(update_vhb, feed_dict={
                        v0: batch, _w: old_w, _hb: old_hb, _vb: old_vb,
                        _vhb: _current_vhb})
                    _current_vvb = sess.run(update_vvb, feed_dict={
                        v0: batch, _w: old_w, _hb: old_hb, _vb: old_vb,
                        _vvb: _current_vvb})
                    old_w = sess.run(update_w, feed_dict={
                                     _w: old_w, _vw: _current_vw})
                    old_hb = sess.run(update_hb, feed_dict={
                        _hb: old_hb, _vhb: _current_vhb})
                    old_vb = sess.run(update_vb, feed_dict={
                        _vb: old_vb, _vvb: _current_vvb})
                image = Image.fromarray(
                    tile_raster_images(
                        X=old_w.T,
                        img_shape=(int(math.sqrt(self._input_size)),
                                   int(math.sqrt(self._input_size))),
                        tile_shape=(int(math.sqrt(self._output_size)),
                                    int(math.sqrt(self._output_size))),
                        tile_spacing=(1, 1)
                    )
                )
                image.save("%s_%d.png" % (self._name, i))
            self.w = old_w
            self.hb = old_hb
            self.vb = old_vb

    def rbmup(self, X):
        """TODO: Docstring for rbmup.

        :X: TODO
        :returns: TODO

        """
        input_X = tf.constant(X)
        _w = tf.constant(self.w)
        _hb = tf.constant(self.hb)
        out = tf.nn.sigmoid(tf.matmul(input_X, _w) + _hb)
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            return sess.run(out)
