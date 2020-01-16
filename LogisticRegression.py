import numpy as np
import tensorflow as tf
load_data=np.loadtxt('./data-01.csv',delimiter=',')
x_data=load_data[:,0:-1]
t_data=load_data[:,[-1]]
W=tf.Variable(tf.random_normal([x_data.shape[1],t_data.shape[1]]))

x=tf.placeholder(tf.float32,[None,3])
t=tf.placeholder(tf.float32,[None,1])
learning_rate=1e-5
optimizer=tf.train.GradientDescentOptimizer(learning_rate)
#세션생성
sess=tf.Session()



y=tf.matmul(x,W)
loss=tf.reduce_mean(tf.square(y-t))
train=optimizer.minimize(loss)

#변수 노드값 초기화
sess.run(tf.global_variables_initializer())

for step in range(10001):
    loss_val,y_val,_=sess.run([loss,y,train],feed_dict={x:x_data,t:t_data})
    if step%500==0:
        print('step=',step,', loss=',loss_val)
print(sess.run(y,feed_dict={x:[[100,98,81]]}))
sess.close()