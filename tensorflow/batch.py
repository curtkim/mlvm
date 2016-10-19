import tensorflow as tf
import numpy as np

xy = np.loadtxt('xor.txt', unpack=True)
x_data = np.transpose(xy[0:-1])             # shape: (4,2)
y_data = np.reshape(xy[-1], (4, 1))         # shape: (4,1)


X = tf.placeholder(tf.float32, name='x-input')

w1 = tf.Variable(tf.random_uniform([2, 5], -1.0, 1.0), name='weight1')
b1 = tf.Variable(tf.zeros([5]), name="Bias1")

Y = tf.matmul(X, w1) + b1

init = tf.initialize_all_variables()

with tf.Session() as sess:
  sess.run(init)
  print sess.run(Y, feed_dict={X: x_data}) # shape=(4,5)