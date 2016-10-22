'''
predict a MNIST digit by comparing it to the most recent n_mem digits.
Each digit is only encoded once. Synthetic gradients are used to update
based on the predicted effect of future comparisons.

A non-parametric algorithm with O(1) time complexity, and O(k*n) space complexity,
where k is the embedding dimensionality, and n is n_mem.

non-minibatch
'''
import tensorflow as tf
import numpy as np
from ops import *
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt
import sys
import time
cur_time = time.clock()
plt.ion()

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

in_dim = 784
hid_dim = 256
z_dim = 64
out_dim = 10
n_mem = 1280
eps = 1e-10
knn = 128

def make_encoder(inp,scope,tied=False):
    with tf.variable_scope('classifier'):
        with tf.variable_scope(scope,reuse=tied):
            hid1 = linear(inp,hid_dim,'hid1',tf.nn.relu)
            hid2 = linear(hid1,z_dim,'hid2')
    return hid2
x_ = tf.placeholder(tf.float32,shape=[1,in_dim])
#memories
mem_x_ = tf.placeholder(tf.float32,shape=[n_mem,in_dim])
mem_y_ = tf.placeholder(tf.float32,shape=[n_mem,out_dim])

'''
encode->decode x_hat for z and recon_loss
'''
#encoder
z = make_encoder(x_,'encoder')
mem_z_ = make_encoder(mem_x_,'encoder',True)
classifier_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='classifier')

#cos_sim comparison
mem_inv_mag = tf.rsqrt(tf.clip_by_value(tf.reduce_sum(tf.square(mem_z_),1,keep_dims=True),eps,float("inf")))
cos_sim = tf.transpose(tf.matmul(mem_z_,tf.transpose(tf.stop_gradient(z)))*mem_inv_mag)
cos_sim,k_inds = tf.nn.top_k(cos_sim,k=knn,sorted=False)
k_inds = k_inds[0]
weighting = tf.nn.softmax(cos_sim) #a [1,n_mem] shaped tensor
label_prob = tf.squeeze(tf.matmul(weighting,tf.gather(mem_y_,k_inds)))
y_ = tf.placeholder(tf.float32,shape=[1,out_dim])
#supervised loss
acc = tf.to_float(tf.nn.in_top_k(tf.expand_dims(label_prob,0),tf.arg_max(y_,1),1))[0]
class_loss = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(label_prob,eps,1)))
optim = tf.train.AdamOptimizer(1e-4)
train_step = optim.minimize(class_loss,var_list=classifier_vars)

sess = tf.Session()
sess.run(tf.initialize_all_variables())
refresh = int(1e3)
cumacc = 0.0
cumloss = 0
cur_loss = 0
from collections import deque
M = {'z': deque(np.zeros((n_mem,in_dim))),'label': deque(np.zeros((n_mem,out_dim)))}
for i in range(n_mem):
    cur_input,cur_output = mnist.train.next_batch(1)
    M['z'].popleft()
    M['z'].append(cur_input[0])
    M['label'].popleft()
    M['label'].append(np.copy(cur_output)[0])
acc_hist = []
loss_hist = []
rho = .999
for i in range(int(1e5)):
    cur_input,cur_output = mnist.train.next_batch(1)
    _,cur_loss,cur_acc,cur_z = sess.run([train_step,class_loss,acc,z],feed_dict={x_:cur_input,y_:cur_output,mem_x_:M['z'],mem_y_:M['label']})
    M['z'].popleft()
    M['z'].append(cur_input[0])
    M['label'].popleft()
    M['label'].append(np.copy(cur_output)[0])
    cumloss*=rho
    cumacc*=rho
    cumloss+=cur_loss*(1-rho)
    cumacc+=cur_acc*(1-rho)
    if (i+1) % refresh == 0: 
        acc_hist.append(cumacc)
        loss_hist.append(cumloss)
        plt.clf()
        #plt.ylim((0,1000))
        time_list = list(range(len(acc_hist)))
        plt.plot(time_list,np.asarray(acc_hist),time_list,np.asarray(loss_hist)/max(np.asarray(loss_hist)))
        plt.pause(.01)
        print(i+1,cumloss,'acc: ',cumacc,'time: ',time.clock()-cur_time)
        cur_time = time.clock()
        #cumloss[:] = 0
        #cumacc = 0.0
