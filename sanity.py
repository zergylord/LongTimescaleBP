mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
in_dim = 28*28
hid_dim = 256
out_dim = 10
inp = tf.placeholder(tf.float32,shape[None,in_dim])
target = tf.placeholder(tf.float32,shape[None,out_dim])
hid1 = linear(inp,hid_dim,'hid1',tf.nn.relu)
hid2 = linear(hid1,z_dim,'hid2')
hid3 = linear(hid2,hid_dim,'hid3',tf.nn.relu)
recon = linear(hid3,in_dim,'recon',tf.nn.sigmoid)
class_loss = -tf.reduce_sum(target*tf.log(tf.clip_by_value(label_prob,eps,1)))
optim = tf.train.AdamOptimizer(1e-3)
train_step = optim.minimize(
for i in range(int(1e5)):
    cur_input,cur_output = mnist.train.next_batch(32)

