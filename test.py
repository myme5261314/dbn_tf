#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Peng Liu <liupeng@imscv.com>
#
# Distributed under terms of the GNU GPL3 license.

"""
Test some function.
"""

import input_data
from rbm_tf import RBM
from opts import DLOption


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images,\
    mnist.test.labels

opts = DLOption(1, 1., 100, 0.9, 0., False, 0.)
rbm = RBM("rbm0", 784, 500, opts)
rbm.train(trX)
