#coding: utf-8

import cv2
import os
import six
import datetime

import chainer
from chainer import optimizers
import chainer.functions as F
import chainer.links as L
import chainer.serializers as S
from clf_image_model import clf_image

import numpy as np

def getDataSet():
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    for i in range(0, 2):
        path = ""
        imgList = os.listdir(path + str(i))
        imgNum = len(imgList)
        cutNum = imgNum - imgNum / 5
        for j in range(len(imgList)):
            imgSrc = cv2.imread(path + str(i) + "/" + imgList[j])
            if imgSrc is None:continue

            if j < cutNum:
                X_train.append(imgSrc)
                y_train.append(i)
            else:
                X_test.append(imgSrc)
                y_test.append(i)
    return X_train, y_train, X_test, y_test

def train():
    X_train, y_train, X_test, y_test = getDataSet()
    X_train = np.array(X_train).astype(np.float32).reshape((len(X_train), 3, 50, 50)) / 255
    y_train = np.array(y_train).astype(np.int32)
    X_test = np.array(X_test).astype(np.float32).reshape((len(X_test), 50, 50)) / 255
    y_test = np.array(y_test).astype(np.int32)

    model = clf_image()
    optimizer = optimizer.Adam()
    optimizer.setup(model)

    epochNum = 5
    batchNum = 50
    epoch = 1

    while epoch <= epochNum
        print("epoch: {}".format(epoch))
        print(datetime.datetime.now())

        trainImgNum = len(y_train)
        testImgNum = len(y_test)

        perm = np.ramdom.permutation(trainImgNum)

        for i in six.moves.range(0, trainImgNum, batchNum):
            X_batch = X_train[perm[i:i+batchNum]]
            y_batch = y_train[perm[i:i+batchNum]]

            optimizer.zero_grads()
            loss, acc = model.forward(X_batch, y_batch)
            loss.backward()
            optimizers.update()

            sumLoss += float(loss.data) * len(y_batch)
            sumArc += float(acc.data) * len(y_batch)
        print('train mean loss={}, accuracy={}'.format(sumLoss / trainImgNum, sumArc / trainImgNum)

        sumArc = 0
        sumLoss = 0

        perm = np.random.permutation(testImgNum)

        for i in six.moves.range(0, testImgNum, batchNum):
            X_batch = X_test[perm[i:i+batchNum]]
            y_batch = X_test[perm[i:i+batchNum]]
            loss, acc = model.forward(X_batch, y_batch, train=False)

            sumLoss += float(loss.data) * len(y_batch)
            sumArc += float(acc.data) * len(y_batch)
        print('test mean loss={}, accuracy={}'.format(sumLoss / testImgNum, sumArc / testImgNum))
        epoch += 1
        S.save_hdf5('model' + str(epoch+1), model)

train()
