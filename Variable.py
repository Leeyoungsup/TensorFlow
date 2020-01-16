
import numpy as np
import tensorflow as tf
W1=tf.Variable(tf.random_normal([1]))
B1=tf.Variable(tf.random_normal([1]))
W2=tf.Variable(tf.random_normal([1,2]))
B2=tf.Variable(tf.random_normal([1,2]))

#세션생성
sess=tf.Session()

#변수 노드값 초기화
sess.run(tf.global_variables_initializer())
for step in range(3):
    W1=W1-step
    B1=B1-step
    W2=W2-step
    B2=B2-step
    print('step=',step,', W1=', sess.run(W1),', B1=',sess.run(B1))
    print('step=',step,', W2=', sess.run(W2),', B1=',sess.run(B2))
sess.close