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
from opts import DLOption
from dbn_tf import DBN
from nn_tf import NN
import numpy as np


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images,\
    mnist.test.labels

opts = DLOption(10, 1., 100, 0.0, 0., 0.)
dbn = DBN([400, 100], opts, trX)
dbn.train()
nn = NN([100], opts, trX, trY)
nn = NN([400, 100], opts, trX, trY)
nn.load_from_dbn(dbn)
nn.train()
print np.mean(np.argmax(teY, axis=1) == nn.predict(teX))
