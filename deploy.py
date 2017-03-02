import tensorflow as tf
import numpy as np
from PIL import Image
import os

from model import ConvWTA

dict_dir = "dict"
if not os.path.isdir(dict_dir): 
  os.makedirs(dict_dir)
recon_dir = "recon"
if not os.path.isdir(recon_dir): 
  os.makedirs(recon_dir)

sess = tf.Session()
ae = ConvWTA(sess, num_features=60)
ae.restore("ckpt/model.ckpt")

# Data read & train
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("mnist/", one_hot=True)

# Save deconv kernels as images.
f = ae.features()
for idx in range(f.shape[-1]):
  Image.fromarray(f[:,:,0,idx]).save("{}/{:03d}.tif".format(dict_dir, idx+1))

# Save recon images
x = tf.placeholder(tf.float32, [1, 28, 28, 1])
y = ae.reconstruct(x)

for i in range(20):
  image = mnist.test.images[i, :]
  image = image.reshape([1, 28, 28, 1])
  result = sess.run(y, {x:image})
  Image.fromarray(result[0,:,:,0]).save("{}/{:03d}.tif".format(recon_dir, i+1))
