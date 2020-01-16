import numpy as np
import tensorflow as tf
load_data=np.loadtxt('./diabetes.csv',delimiter=',')
x_data=load_data[:,0:-1]
t_data=load_data[:,[-1]]
W=tf.Variable(tf.random_normal([x_data.shape[1],t_data.shape[1]]))
b=tf.Variable(tf.random_normal([1]))
x=tf.placeholder(tf.float32,[None,x_data.shape[1]])
t=tf.placeholder(tf.float32,[None,t_data.shape[1]])
learning_rate=1e-4
optimizer=tf.train.GradientDescentOptimizer(learning_rate)
#세션생성
sess=tf.Session()



z=tf.matmul(x,W)+b
y=tf.sigmoid(z)
loss=-tf.reduce_mean(t*tf.log(y)+(1-t)*tf.log(1-y))
train=optimizer.minimize(loss)
predicted=tf.cast(y>0.5,dtype=tf.float32)
accuracy=tf.reduce_mean(tf.cast(tf.equal(predicted,t),dtype=tf.float32))
#변수 노드값 초기화
sess.run(tf.global_variables_initializer())

for step in range(20001):
    loss_val,_=sess.run([loss,train],feed_dict={x:x_data,t:t_data})
    if step%500==0:
        print('step=',step,', loss=',loss_val)
y_val,predicted_val,accuracy_val=sess.run([y,predicted,accuracy],feed_dict={x:x_data,t:t_data})
print(y_val.shape, predicted_val.shape)
print('\n',accuracy_val)
sess.close()