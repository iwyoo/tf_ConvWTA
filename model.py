import tensorflow as tf

"""
Convolutioanl Winner-Take-All Autoencoder TensorFlow implementation.
Usage :
  ae = ConvWTA(sess)

  # 1. to train an Autoencoder
  loss = ae.loss()
  train = optimizer.minimize(loss)
  sess.run(train)

  # 2. to get the sparse codes
  h = ae.encoder(x)
  sess.run(h, feed_dict={...})

  # 3. to get the reconstructed results
  y = ae.reconstruct(x)
  sess.run(y, feed_dict={...})

  # 4. to get the learned features
  f = ae.features() # np.float32 array with shape [11, 11, 1, 16]
    # 4-1. to train a different number of features
    ae = ConvWTA(sess, num_features=32)

  # 5. to save & restore the variables
  ae.save(save_path)
  ae.restore(save_path)

Reference: 
  [1] https://arxiv.org/pdf/1409.2752.pdf
"""

class ConvWTA(object):
  """
    Args :
      sess : TensorFlow session.
      x : Input tensor.
  """
  def __init__(self, sess, num_features=16,  name="ConvWTA"):
    self.sess = sess
    self.name = name
    self.size = [1, 128, 128, num_features]  # ref [1]

    self._set_variables()
    self.t_vars = tf.get_collection(
      tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
    self.sess.run(tf.variables_initializer(self.t_vars))
    self.saver = tf.train.Saver(self.t_vars)
      
  def encoder(self, x):
    with tf.variable_scope(self.name) as vs:
      h = self._conv(x, self.size[1], 5, 5, 1, 1, "conv_1")
      h = self._conv(h, self.size[2], 5, 5, 1, 1, "conv_2")
      h = self._conv(h, self.size[3], 5, 5, 1, 1, "conv_3")
    return h

  def _decoder(self, h):
    shape = tf.shape(h)
    out_shape = tf.pack([shape[0], shape[1], shape[2], 1])
    with tf.variable_scope(self.name) as vs:
      y = self._deconv(h, out_shape, self.size[0], 
                       11, 11, 1, 1, "deconv", end=True)
    return y

  def loss(self, x, lifetime_sparsity=0.20):
    h = self.encoder(x)
    h, winner = self._spatial_sparsity(h)
    h = self._lifetime_sparsity(h, winner, lifetime_sparsity)
    y = self._decoder(h)

    return tf.reduce_sum(tf.square(y - x))

  def reconstruct(self, x):
    h = self.encoder(x)
    h, _ = self._spatial_sparsity(h)
    y = self._decoder(h)
    return y
    
  def _set_variables(self):
    with tf.variable_scope(self.name) as vs:
      self._conv_var(self.size[0], self.size[1],  5,  5, "conv_1")
      self._conv_var(self.size[1], self.size[2],  5,  5, "conv_2")
      self._conv_var(self.size[2], self.size[3],  5,  5, "conv_3")
      self.f, _ = self._deconv_var(
        self.size[-1], self.size[0], 11, 11, "deconv")
    
  def _conv_var(self, in_dim, out_dim, k_h, k_w, name, stddev=0.1):
    with tf.variable_scope(name) as vs:
      k = tf.get_variable('filter',
          [k_h, k_w, in_dim, out_dim],
          initializer=tf.truncated_normal_initializer(stddev=stddev))
      b = tf.get_variable('biases', [out_dim],
        initializer=tf.constant_initializer(0.0001))
    return k, b

  def _deconv_var(self, in_dim, out_dim, k_h, k_w, name, stddev=0.1):
    with tf.variable_scope(name) as vs:
      k = tf.get_variable('filter',
          [k_h, k_w, out_dim, in_dim],
          initializer=tf.truncated_normal_initializer(stddev=stddev))
      b = tf.get_variable('biases', [out_dim],
        initializer=tf.constant_initializer(0.0001))
    return k, b

  def _conv(self, x, out_dim, 
            k_h, k_w, s_h, s_w, name, end=False):
    with tf.variable_scope(name, reuse=True) as vs:
      k = tf.get_variable('filter')
      b = tf.get_variable('biases')
      conv = tf.nn.conv2d(x, k, [1, s_h, s_w, 1], "SAME") + b
    return conv if end else tf.nn.relu(conv)

  def _deconv(self, x, out_shape, out_dim,
            k_h, k_w, s_h, s_w, name, end=False):
    with tf.variable_scope(name, reuse=True) as vs:
      k = tf.get_variable('filter')
      b = tf.get_variable('biases')
      deconv = tf.nn.conv2d_transpose(
        x, k, out_shape, [1, s_h, s_w, 1], "SAME") + b
    return deconv if end else tf.nn.relu(deconv)

  def _spatial_sparsity(self, h):
    shape = tf.shape(h)
    n = shape[0]
    c = shape[3]

    h_t = tf.transpose(h, [0, 3, 1, 2]) # n, c, h, w
    h_r = tf.reshape(h_t, tf.pack([n, c, -1])) # n, c, h*w

    th, _ = tf.nn.top_k(h_r, 1) # n, c, 1
    th_r = tf.reshape(th, tf.pack([n, 1, 1, c])) # n, 1, 1, c
    drop = tf.select(h < th_r, 
      tf.zeros(shape, tf.float32), tf.ones(shape, tf.float32))

    # spatially dropped & winner
    return h*drop, tf.reshape(th, tf.pack([n, c])) # n, c
    
  def _lifetime_sparsity(self, h, winner, rate):
    shape = tf.shape(winner)
    n = shape[0]
    c = shape[1]
    k = tf.cast(rate * tf.cast(n, tf.float32), tf.int32)

    winner = tf.transpose(winner) # c, n
    th_k, _ = tf.nn.top_k(winner, k) # c, k

    shape_t = tf.pack([c, n])
    drop = tf.select(winner < th_k[:,k-1:k], # c, n
      tf.zeros(shape_t, tf.float32), tf.ones(shape_t, tf.float32))
    drop = tf.transpose(drop) # n, c
    return h * tf.reshape(drop, tf.pack([n, 1, 1, c]))

  def features(self): 
    return self.sess.run(self.f)
    
  def save(self, ckpt_path):
    self.saver.save(self.sess, ckpt_path)

  def restore(self, ckpt_path):
    self.saver.restore(self.sess, ckpt_path)
