import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
# Demo1 hello tensorflow
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
# Demo2  plus
a = tf.constant(66)
b = tf.constant(88)
print(sess.run(a + b))

# import numpy as np
# w = np.array([[0.4], [1.2]])
# x = np.array([range(1, 6), range(5, 10)])
# print w
# print x
# print w * x


