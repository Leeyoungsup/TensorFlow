import numpy as np
import tensorflow as tf

a=tf.constant(1.0,name='a')
b=tf.constant(2.0,name='b')
c=tf.constant([[1.0,2.0],[3.0,4.0]])
print(a)
print(a+b)
print(c)
sess=tf.Session()
print(sess.run([a,b]))
print(sess.run(c))
print(sess.run([a+b]))
print(sess.run(c+1.0))
sess.close