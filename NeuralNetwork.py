
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('MNIST_data/',one_hot=True)
input_nodes=784
hidden_nodes=100
output_nodes=10
W2=tf.Variable(tf.random_normal([input_nodes,hidden_nodes]))
b2=tf.Variable(tf.random_normal([hidden_nodes]))
W3=tf.Variable(tf.random_normal([hidden_nodes,output_nodes]))
b3=tf.Variable(tf.random_normal([output_nodes]))
x=tf.placeholder(tf.float32,[None,input_nodes])
t=tf.placeholder(tf.float32,[None,output_nodes])
learning_rate = 0.1
epochs = 100 
batch_size = 100 
optimizer=tf.train.GradientDescentOptimizer(learning_rate)
#세션생성
sess=tf.Session()

z2=tf.matmul(x,W2)+b2
a2=tf.nn.relu(z2)
z3=logits=tf.matmul(a2,W3)+b3
y=tf.nn.softmax(z3)
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=z3,labels=t))
train=optimizer.minimize(loss)
predicted_val=tf.equal(tf.argmax(y,1),tf.argmax(t,1))
accuracy=tf.reduce_mean(tf.cast(predicted_val,dtype=tf.float32))
#변수 노드값 초기화
sess.run(tf.global_variables_initializer())
for i in range(epochs):
    total_batch=int(mnist.train.num_examples/batch_size)
    for step in range(total_batch):
        batch_x_data,batch_t_data=mnist.train.next_batch(batch_size)
        
        loss_val,_=sess.run([loss,train],feed_dict={x:batch_x_data,t:batch_t_data})
        if step%100==0:
            print('step=',step,', loss=',loss_val)
    test_x_data=mnist.test.images
    test_t_data=mnist.test.labels
    accuracy_val=sess.run([accuracy],feed_dict={x:test_x_data,t:test_t_data})
    print('\n',accuracy_val)
sess.close()