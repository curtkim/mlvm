# http://www.datasciencecentral.com/profiles/blogs/a-primer-on-universal-function-approximation-with-deep-learning
import tensorflow as tf
import numpy as np

#xy = np.loadtxt('xor.txt', unpack=True)
#x_data = np.transpose(xy[0:-1])             # shape: (4,2)
#y_data = np.reshape(xy[-1], (4, 1))         # shape: (4,1)

x_data = np.zeros((20*20, 2))
y_data = np.zeros((20*20, 1))
for y in np.arange(-10, 10, 1):
  for x in np.arange(-10, 10, 1):
    x_data[(y+10)*20+(x+10)] = [y, x]
    y_data[(y+10)*20+(x+10)] = 2 * x**2 -3 * y**2 + 1


X = tf.placeholder(tf.float32, name='x-input')
Y = tf.placeholder(tf.float32, name='y-input')

w1 = tf.Variable(tf.random_uniform([2, 5], -1.0, 1.0), name='weight1')
w2 = tf.Variable(tf.random_uniform([5, 10], -1.0, 1.0), name='weight2')
w3 = tf.Variable(tf.random_uniform([10, 1], -1.0, 1.0), name='weight3')

b1 = tf.Variable(tf.zeros([5]), name="Bias1")
b2 = tf.Variable(tf.zeros([10]), name="Bias2")
b3 = tf.Variable(tf.zeros([1]), name="Bias3")

L2 = tf.matmul(X, w1) + b1
L3 = tf.matmul(L2, w2) + b2
hypothesis = tf.matmul(L3, w3) + b3



cost = tf.square(Y - hypothesis)
a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:
  sess.run(init)

  print sess.run(hypothesis, feed_dict={X: x_data})

  #for step in xrange(1):
  #  sess.run(train, feed_dict={X: x_data, Y: y_data})
  #  if step % 200 == 0:
  #    print step, sess.run(cost, feed_dict={X: x_data, Y: y_data})

  #print sess.run(hypothesis, feed_dict={X: [0,0]})
