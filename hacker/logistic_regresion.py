#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn import datasets
diabetes = datasets.load_diabetes()

import matplotlib.pyplot as plt
from sklearn import metrics

#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(diabetes.data[:,2], diabetes.target, test_size=0.1, random_state=10)

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

  def update_grad(self):
    self.set_params(self._w + self._w_grad, self._b + self._b_grad)

  def backprop(self, err, lr=0.1):
    """에러를 입력받아 가중치와 바이어스의 변화율을 곱하고 평균을 낸 후 감쇠된 변경량을 저장합니다."""
    m = len(self._x)
    self._w_grad = lr * np.sum(err * self._x) / m
    self._b_grad = lr * np.sum(err * 1) / m

  def fit(self, X, y, n_iter=10, lr=0.1, cost_check=False):
    """정방향 계산을 하고 역방향으로 에러를 전파시키면서 모델을 최적화시킵니다."""
    cost = []
    for i in range(n_iter):
      y_hat = self.forpass(X)
      error = y - y_hat
      self.backprop(error, lr)
      self.update_grad()
      if cost_check:
        cost.append(np.sum(np.square(y - y_hat))/len(y))
    return cost

split_at = len(diabetes.data) - len(diabetes.data) / 10
(X_train, X_test) = (diabetes.data[:split_at,2], diabetes.data[split_at:,2])
(y_train, y_test) = (diabetes.target[:split_at], diabetes.target[split_at:]) 
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

n1 = SingleNeuron()
#n1.set_params(5, 1)
#n1.fit(X_train, y_train, 30000)
#print('Final W', n1._w)
#print('Final b', n1._b)
#
#y_hat = n1.forpass(X_test)
##print(np.sum(np.square(y_test - y_hat)/len(X_test)))
#print( metrics.mean_squared_error(y_test, y_hat))

costs = []
learning_rate = [1.999, 1.0, 0.1]
for lr in learning_rate:
  n1.set_params(5, 1)
  costs.append([])
  costs[-1] = n1.fit(X_train, y_train, 2000, lr, True)

print(len(costs))

for i, color in enumerate(['red', 'blue', 'black']):
    plt.plot(list(range(2000)), costs[i], color=color)
plt.ylim(3500, 7000)
plt.show()