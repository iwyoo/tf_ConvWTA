import tensorflow as tf
import numpy as np
from PIL import Image
import os

from model import ConvWTA

dict_dir = "dict"
if not os.path.isdir(dict_dir): 
  os.makedirs(dict_dir)

sess = tf.Session()
ae = ConvWTA(sess)
ae.restore("ckpt/model.ckpt")

# Save deconv kernels as images.
f = ae.features()
for idx in range(f.shape[-1]):
  Image.fromarray(f[:,:,0,idx]).save("{}/{:03d}.tif".format(dict_dir, idx+1))
