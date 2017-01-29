#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn import datasets
diabetes = datasets.load_diabetes()
print(diabetes.data.shape, diabetes.target.shape)

import matplotlib.pyplot as plt


class SingleNeuron(object):

  def __init__(self):
    self._w = 0
    self._b = 0
    self._x = 0

  def set_params(self, w, b):
    self._w = w
    self._b = b

  def forpass(self, x):
    self._x = x
    _y_hat = self._w * self._x + self._b
    return _y_hat

  def backprop(self, err):
    """에러를 입력받아 가중치와 바이어스의 변화율을 곱하고 평균을 낸 후 감쇠된 변경량을 저장합니다."""
    m = len(self._x)
    self._w_grad = 0.1 * np.sum(err * self._x) / m
    self._b_grad = 0.1 * np.sum(err * 1) / m

  def update_grad(self):
    self.set_params(self._w + self._w_grad, self._b + self._b_grad)


n1 = SingleNeuron()
n1.set_params(5, 1)
print(n1.forpass(3))

for i in range(30000):
  y_hat = n1.forpass(diabetes.data[:,2])
  error = diabetes.target - y_hat
  n1.backprop(error)
  n1.update_grad()

print('Final W', n1._w)
print('Final b', n1._b)


from sklearn import linear_model

sgd_regr = linear_model.SGDRegressor(n_iter=30000, penalty='none')
sgd_regr.fit(diabetes.data[:, 2].reshape(-1, 1), diabetes.target)
print('Coefficients: ', sgd_regr.coef_, sgd_regr.intercept_)