import os
import tensorflow as tf
import numpy as np
# Import os to set the environment variable CUDA_VISIBLE_DEVICES
import GPUtil

# Get the first available GPU
DEVICE_ID_LIST = GPUtil.getFirstAvailable()
DEVICE_ID = DEVICE_ID_LIST[0] # grab first element from list

# Set CUDA_VISIBLE_DEVICES to mask out all other GPUs than the first available device id
os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)

# Since all other GPUs are masked out, the first available GPU will now be identified as GPU:0
device = '/gpu:0'
print('Device ID (unmasked): ' + str(DEVICE_ID))
print('Device ID (masked): ' + str(0))

# Creates a graph.
with tf.device(device):
  a = tf.placeholder(tf.float32, shape=(4096, 4096))
  b = tf.placeholder(tf.float32, shape=(4096, 4096))
  c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
# Runs the op.
x = np.random.rand(4096, 4096)
for i in range(100000):
  sess.run(c, feed_dict={a: x, b: x})
