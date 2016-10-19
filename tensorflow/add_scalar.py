#!/usr/bin/env python
import tensorflow as tf

a = tf.placeholder("float")
b = tf.placeholder("float")

y = tf.add(a, b)

with tf.Session() as sess:
  print("%f should equal 3.0" % sess.run(y, feed_dict={a: 1, b: 2}))
  print("%f should equal 6.0" % sess.run(y, feed_dict={a: 3, b: 3}))
