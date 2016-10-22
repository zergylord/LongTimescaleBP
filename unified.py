'''
predict a MNIST digit by comparing it to the most recent n_mem digits.
Each digit is only encoded once. Synthetic gradients are used to update
based on the predicted effect of future comparisons.

A non-parametric algorithm with O(1) time complexity, and O(k*n) space complexity,
where k is the embedding dimensionality, and n is n_mem.

non-minibatch
'''
import sys
if len(sys.argv) < 3:
    print('condition:{0,1,2} and trial # please')
    sys.exit()
import tensorflow as tf
import numpy as np
from ops import *
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt
import time
cur_time = time.clock()
plt.ion()

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

in_dim = 784
hid_dim = 256
z_dim = 64
out_dim = 10
n_mem = 128
eps = 1e-10
knn = 128
helper_lr = 1e-3
condition = int(sys.argv[1]) #0 use fake, 1 recon, 2 syn
if condition == 0:
    print('use fake recon')
    lr = 1e-4
elif condition == 1:
    print('use recon for credit')
    lr = 1e-3
elif condition == 2:
    print('use synthetic gradients')
    lr = 1e-4

def make_encoder(inp,scope,tied=False):
    with tf.variable_scope('classifier'):
        with tf.variable_scope(scope,reuse=tied):
            hid1 = linear(inp,hid_dim,'hid1',tf.nn.relu)
            hid2 = linear(hid1,z_dim,'hid2')
    return hid2
def make_decoder(inp,scope,tied=False):
    with tf.variable_scope(scope,reuse=tied):
        hid = linear(inp,hid_dim,'hid3',tf.nn.relu)
        recon = linear(hid,in_dim,'recon',tf.nn.sigmoid)
    return recon
def make_syn_net(inputs,tie_weights=False):
    syn_hid1 = linear(tf.concat(1,inputs),hid_dim,'syn_hid1',tf.nn.relu,tied=tie_weights)
    syn_out = linear(syn_hid1,z_dim,'syn_out',bias_value=0.0,init=tf.constant_initializer(),tied=tie_weights)
    return syn_out

x_ = tf.placeholder(tf.float32,shape=[None,in_dim])
#memories
mem_z_ = tf.placeholder(tf.float32,shape=[n_mem,z_dim])
mem_y_ = tf.placeholder(tf.float32,shape=[n_mem,out_dim])

'''
encode->decode x_hat for z and recon_loss
'''
#encoder
z = make_encoder(x_,'encoder')
#decoder
recon = make_decoder(z,'helper')

helper_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='helper')
recon_loss = tf.reduce_mean(-tf.reduce_sum(x_*tf.log(tf.clip_by_value(recon,1e-10,1))+(1-x_)*tf.log(tf.clip_by_value(1-recon,1e-10,1)),1))
'''
decode->encode mem_z_ for passing class_loss gradient
'''
classifier_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='classifier')

#cos_sim comparison
if condition == 0:
    mem_recon = make_decoder(mem_z_,'helper',True)
    fake_z = make_encoder(mem_recon,'encoder',True)
    mem_inv_mag = tf.rsqrt(tf.clip_by_value(tf.reduce_sum(tf.square(fake_z),1,keep_dims=True),eps,float("inf")))
    cos_sim = tf.transpose(tf.matmul(fake_z,tf.transpose(tf.stop_gradient(z)))*mem_inv_mag)
elif condition == 1 or condition == 2:
    mem_inv_mag = tf.rsqrt(tf.clip_by_value(tf.reduce_sum(tf.square(mem_z_),1,keep_dims=True),eps,float("inf")))
    cos_sim = tf.transpose(tf.matmul(mem_z_,tf.transpose(tf.stop_gradient(z)))*mem_inv_mag)
cos_sim,k_inds = tf.nn.top_k(cos_sim,k=knn,sorted=False)
k_inds = k_inds[0]
weighting = tf.nn.softmax(cos_sim) #a [1,n_mem] shaped tensor
label_prob = tf.matmul(weighting,tf.gather(mem_y_,k_inds))

y_ = tf.placeholder(tf.float32,shape=[None,out_dim])
#supervised loss
acc = tf.to_float(tf.nn.in_top_k(label_prob,tf.arg_max(y_,1),1))[0]
valid_acc = tf.reduce_mean(tf.to_float(tf.nn.in_top_k(label_prob,tf.arg_max(y_,1),1)))
class_loss = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(tf.squeeze(label_prob),eps,1)))

'''
crazy stuff
'''
optim = tf.train.AdamOptimizer(lr)
helper_optim = tf.train.AdamOptimizer(helper_lr)
if condition == 0: #use fake
    syn_loss = tf.constant(0) #dummy op for consistency
    train_step = optim.minimize(class_loss,var_list=classifier_vars)
    helper_train_step = helper_optim.minimize(recon_loss)#,var_list=helper_vars)
elif condition == 1: #recon
    mem_recon = make_decoder(tf.gather(mem_z_,k_inds),'helper',True)
    fake_z = make_encoder(mem_recon,'encoder',True)
    mem_z_grad = tf.gradients(class_loss,mem_z_)[0]
    syn_loss = tf.reduce_mean(tf.batch_matmul(tf.expand_dims(fake_z,1),
        tf.expand_dims(tf.stop_gradient(tf.gather(mem_z_grad,k_inds)),2)))
    train_step = optim.minimize(syn_loss,var_list=classifier_vars)
    helper_train_step = helper_optim.minimize(recon_loss)#,var_list=helper_vars)
elif condition == 2: #syn net
    syn_out = tf.stop_gradient(make_syn_net([z,y_]))
    syn_loss = tf.squeeze(tf.matmul(z,tf.transpose(syn_out)))
    syn_out_copy = make_syn_net([mem_z_,mem_y_],True)
    mem_z_grad = tf.gradients(class_loss,mem_z_)[0]
    grad_loss = tf.reduce_mean(tf.reduce_sum(tf.square(syn_out_copy
        -tf.stop_gradient(mem_z_grad)),1))
    train_step = optim.minimize(grad_loss + syn_loss)
    helper_train_step = tf.constant(0) #dummy op for consistency



sess = tf.Session()
sess.run(tf.initialize_all_variables())
refresh = int(1e3)
cumacc = 0.0
cumloss = np.zeros((3,))
cur_los = np.zeros((3,))
from collections import deque
M = {'z': deque(np.zeros((n_mem,z_dim))),'label': deque(np.zeros((n_mem,out_dim)))}
for i in range(n_mem):
    cur_input,cur_output = mnist.train.next_batch(1)
    M['z'].popleft()
    M['z'].append(sess.run(z,feed_dict={x_:cur_input})[0])
    M['label'].popleft()
    M['label'].append(np.copy(cur_output)[0])
acc_hist = []
loss_hist = []
rho = .999
valid_images = mnist.validation.images
valid_labels = mnist.validation.labels
num_steps = int(1e4)
valid_acc_hist = np.zeros((num_steps/refresh))
for i in range(num_steps+1):
    cur_input,cur_output = mnist.train.next_batch(1)
    _,_,*cur_loss,cur_acc,cur_z,image = sess.run([train_step,helper_train_step,class_loss,syn_loss,recon_loss,acc,z,recon],
            feed_dict={x_:cur_input,y_:cur_output,mem_z_:M['z'],mem_y_:M['label']})
    M['z'].popleft()
    M['z'].append(cur_z[0])
    M['label'].popleft()
    M['label'].append(np.copy(cur_output)[0])
    cumloss*=rho
    cumacc*=rho
    cumloss+=np.asarray(cur_loss)*(1-rho)
    cumacc+=cur_acc*(1-rho)
    if (i+1) % refresh == 0: 
        acc_hist.append(cumacc)
        loss_hist.append(cumloss[0])
        plt.clf()
        #plt.ylim((0,1000))
        time_list = list(range(len(acc_hist)))
        plt.plot(time_list,np.asarray(acc_hist),time_list,np.asarray(loss_hist)/max(np.asarray(loss_hist)))
        #plt.imshow(np.reshape(image,[28,28]))
        plt.pause(.01)
        v_acc = sess.run(valid_acc,feed_dict={x_:valid_images,y_:valid_labels,mem_z_:M['z'],mem_y_:M['label']})
        valid_acc_hist[(i+1)/refresh - 1] = v_acc
        print(i+1,*(cumloss),'acc: ',cumacc,'valid acc: ',v_acc,'time: ',time.clock()-cur_time)
        cur_time = time.clock()
        #cumloss[:] = 0
        #cumacc = 0.0
np.save('res_'+sys.argv[1]+'_'+sys.argv[2],valid_acc_hist)
