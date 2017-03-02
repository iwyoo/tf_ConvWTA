import tensorflow as tf
tf.set_random_seed(2017)

from model import ConvWTA

import os
ckpt_dir = "ckpt/"
if not os.path.isdir(ckpt_dir): 
  os.makedirs(ckpt_dir)
ckpt_path = "ckpt/model.ckpt"

epochs = 100
batch_size = 200
learning_rate = 1e-3
shape = [batch_size, 28, 28, 1]

# Basic tensorflow setting
sess = tf.Session()
ae = ConvWTA(sess)
x = tf.placeholder(tf.float32, shape)
loss = ae.loss(x)

optim = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optim.minimize(loss, var_list=ae.t_vars)

sess.run(tf.global_variables_initializer())

# Data read & train
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("mnist/", one_hot=True)

import time
start_time = time.time()
for epoch in range(epochs):
  total_batch = int(mnist.train.num_examples / batch_size)
  avg_loss = 0
  for i in range(total_batch):
    batch_x, _ = mnist.train.next_batch(batch_size)

    batch_x = batch_x.reshape(shape)
    l, _ =  sess.run([loss, train], {x:batch_x})
    avg_loss += l / total_batch
    print l
    
  print("Epoch : {:04d}, Loss : {:.9f}".format(epoch+1, avg_loss))
print("Training time : {}".format(time.time() - start_time))

ae.save(ckpt_path)
