#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Peng Liu <liupeng@imscv.com>
#
# Distributed under terms of the GNU GPL3 license.

"""
This file implement a class DBN.
"""

from rbm_tf import RBM


class DBN(object):

    """Docstring for DBN. """

    def __init__(self, sizes, opts, X):
        """TODO: to be defined1.

        :sizes: TODO
        :opts: TODO

        """
        self._sizes = sizes
        self._opts = opts
        self._X = X
        self.rbm_list = []
        input_size = X.shape[1]
        for i, size in enumerate(self._sizes):
            self.rbm_list.append(RBM("rbm%d" % i, input_size, size, self._opts))
            input_size = size

    def train(self):
        """TODO: Docstring for train.
        :returns: TODO

        """
        X = self._X
        for rbm in self.rbm_list:
            rbm.train(X)
            X = rbm.rbmup(X)
