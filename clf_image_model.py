#coding:utf-8

import os

import chainer
from chainer import optimizers
import chainer.functions as F
import chainer.links as L
import chainer.serializers as S

# This is also possible code
#from chainer import cuda, Function, gradient_check, report, training, utils, Variable
#from chainer import datasets, iterators, optimizers, serializers
#from chainer import Link, Chain, ChainList

import numpy as np

# sub class of Chain class
# Chain class can keep parameter, support CPU/GPU and saving/reading
class clf_image(chainer.Chain):
    def __init__(self):
        # Two-dimensional convolutional layer
        # INPUT -> (CONV -> POOL) * 2 -> FC(fully-connected layer)
        super(clf_image, self).__init__(

            ### convolutional layer ###
            # input_chanel=3, output_chanel=16, ksize=5, pad=2
            # Image Size is 50 * 50
            # (50 + 4(add top and bottom padding, each 2 size)) - 2[5/2] * (50 + 4((add right and left padding, each 2 size)) - 2[5/2] = 50 * 50
            conv1 = F.Convolution2D(3, 16, 5, pad=2),
            # input_chanel=16, output_chanel=32, ksize=5, pad=2
            # (27 + 4(padding)) - 2[5/2] * (27 + 4(padding)) - 2[5/2] = 27 * 27
            conv2 = F.Convolution2D(16, 32, 5, pad=2),

            ### affine transformation ###
            # 14 * 14 * 32 = 6272
            # x=6272, W=256, b=none
            # make sure of understading why 6272 and 256
            l3 = F.linear(6272, 256),
            # x=256, W=2, b=none
            l4 = F.linear(256, 2)
        )

    def clear(self):
        self.loss = None
        self.accuracy = None

    def forward(self, X_data, y_data, train=True):
        # Initialize
        self.clear()
        # Create Data
        X_data = chainer.Variable(np.asarray(X_data), valatile=not train)
        y_data = chainer.Variable(np.asarray(y_data), valatile=not train)

        ### max pooling ###
        # [((50 + 4) - 1) / 2] + 1 = 27, output map is 27*27
        h = F.max_pooling_2d(F.relu(self.conv1(X_data)), ksize=5, stride=2, pad=2)
        # [((27 + 4) - 1) / 2] + 1 = 16, output map is 16*16
        h = F.max_pooling_2d(F.relu(self.conv2(h)), ksize=5, stride=2, pad=2)
        h = F.dropout(F.relu(self.l3(h)), train=train)
        y = self.l4(h)
        return F.softmax_cross_entropy(y, y_data), F.accuracy(y, y_data)
