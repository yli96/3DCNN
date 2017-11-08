from __future__ import absolute_import  
from __future__ import print_function 
import numpy as np
import nibabel as nib
import tensorflow as tf 
from keras.preprocessing.image import ImageDataGenerator  
from keras.models import Sequential  
from keras.layers.core import Dense, Dropout, Activation, Flatten  
from keras.layers.advanced_activations import PReLU  
from keras.layers.convolutional import Convolution2D, MaxPooling2D,Conv3D,MaxPooling3D
from keras.optimizers import SGD, Adadelta, Adagrad  
from keras.utils import np_utils, generic_utils
import os
from six.moves import range  
import tensorflow as tf
#import GPUtil
# Get the first available GPU
'''
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
'''
#os.chdir('/Users/anthony/tx/689Project')
#sess = tf.InteractiveSession()

c = 0
def load_mri_images(filename):
    global c
    
    temp = nib.load(filename)
    data=temp.get_data()
    tmp = c
    c = tmp + 1
    
    print ('Loaded image set %d of 65.' %c)
    
    return data
def imgwise_3d_scaling(data):
     #loop over patients
    for i in range(len(data)):
        max_val_3d = np.amax(data[i][0])

        data[i][0] = data[i][0].astype('float32')
        data[i][0] /= max_val_3d

    print ('Executed imagewise 3d scaling.')
    
    return data

train_label=np.zeros((45,3))
train_data = np.array([], np.float32)
train_data = np.append(train_data, np.asarray(load_mri_images('AD (1).nii'), dtype='float32'))
train_label[0][0]=1
for i in range(2,16):
    train_cur = load_mri_images('AD (%d).nii' %i)
    train_data = np.append(train_data, np.asarray(train_cur, dtype='float32'))
    train_label[i-1][0]=1
for i in range(1,16):
    train_cur = load_mri_images('MCI (%d).nii' %i)
    train_data = np.append(train_data, np.asarray(train_cur, dtype='float32'))
    train_label[14+i][1]=1
for i in range(1,16):
    train_cur = load_mri_images('Normal (%d).nii' %i)
    train_data = np.append(train_data, np.asarray(train_cur, dtype='float32'))
    train_label[29+i][2]=1
train_data = train_data.reshape(45,1,256,256,166)
'''
val_data=np.array([], np.float32)
val_label=np.zeros((7,3))
val_data = np.append(val_data, np.asarray(load_mri_images('AD6.nii'), dtype='float32'))
val_label[0]=1
for i in range(7,9):
    valid_cur = load_mri_images('AD(%d).nii' %i)
    val_data = np.append(val_data, np.asarray(valid_cur, dtype='float32'))
    val_label[i-6][0]=1
for i in range(9,11):
    valid_cur = load_mri_images('MCI(%d).nii' %i)
    val_data = np.append(val_data, np.asarray(valid_cur, dtype='float32'))
    val_label[i-6][1]=1
for i in range(7,9):
    valid_cur = load_mri_images('Normal(%d).nii' %i)
    val_data = np.append(val_data, np.asarray(valid_cur, dtype='float32'))
    val_label[i-2][2]=1
val_data = val_data.reshape(7,1,256,256,166)
'''
test_data=np.array([], np.float32)
test_data = np.append(test_data, np.asarray(load_mri_images('AD (16).nii'), dtype='float32'))
for i in range(17,20):
    test_cur = load_mri_images('AD (%d).nii' %i)
    test_data = np.append(test_data, np.asarray(test_cur, dtype='float32'))
for i in range(16,20):
    test_cur = load_mri_images('MCI (%d).nii' %i)
    test_data = np.append(test_data, np.asarray(test_cur, dtype='float32'))
for i in range(16,28):
    test_cur = load_mri_images('Normal (%d).nii' %i)
    test_data = np.append(test_data, np.asarray(test_cur, dtype='float32'))
test_data = test_data.reshape(20,1,256,256,166)


    
# normalize the images on a per-image basis
imgwise_scaling = True
if imgwise_scaling:
    for n in range(len(train_data)):
        train_data[n,:,:,:] = train_data[n,:,:,:] - np.mean(train_data[n,:,:,:].flatten())
        train_data[n,:,:,:] = train_data[n,:,:,:] / np.std(train_data[n,:,:,:].flatten())
#    for n in range(len(val_data)):
#        val_data[n,:,:,:] = val_data[n,:,:,:] - np.mean(val_data[n,:,:,:].flatten())
#        val_data[n,:,:,:] = val_data[n,:,:,:] / np.std(val_data[n,:,:,:].flatten())
    for n in range(len(test_data)):
        test_data[n,:,:,:] = test_data[n,:,:,:] - np.mean(test_data[n,:,:,:].flatten())
        test_data[n,:,:,:] = test_data[n,:,:,:] / np.std(test_data[n,:,:,:].flatten())
   
width = 256
height = 256
depth = 166
nLabel = 3


model = Sequential()
model.add(Conv3D(32, kernel_size=(3, 3, 3), input_shape=(
    train_data.shape[1:]), border_mode='same'))
model.add(Activation('relu'))
#model.add(Conv3D(32, kernel_size=(3, 3, 3), border_mode='same'))
#model.add(Activation('softmax'))
model.add(MaxPooling3D(pool_size=(2, 2, 2), border_mode='same'))
model.add(Dropout(0.25))

model.add(Conv3D(64, kernel_size=(3, 3, 3), border_mode='same'))
model.add(Activation('relu'))
#model.add(Conv3D(64, kernel_size=(3, 3, 3), border_mode='same'))
#model.add(Activation('softmax'))
model.add(MaxPooling3D(pool_size=(2, 2, 2), border_mode='same'))
model.add(Dropout(0.25))

model.add(Conv3D(64, kernel_size=(3, 3, 3), border_mode='same'))
model.add(Activation('relu'))
#model.add(Conv3D(64, kernel_size=(3, 3, 3), border_mode='same'))
#model.add(Activation('softmax'))
model.add(MaxPooling3D(pool_size=(2, 2, 2), border_mode='same'))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(512, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(nLabel, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='Adam', metrics=['accuracy'])
model.summary()
model.fit(train_data, train_label, epochs=10, batch_size=40)


classes = model.predict(test_data, batch_size=20)
print (classes)





'''


# Placeholders (MNIST image:28x28pixels=784, label=10)
x = tf.placeholder(tf.float32, shape=[None, width,height,depth]) # [None, 28*28]
y_ = tf.placeholder(tf.float32, shape=[None, nLabel])  # [None, 10]

## Weight Initialization
# Create lots of weights and biases & Initialize with a small positive number as we will use ReLU
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

## Convolution and Pooling
# Convolution here: stride=1, zero-padded -> output size = input size
def conv3d(x, W):
  return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME') # conv2d, [1, 1, 1, 1]

# Pooling: max pooling over 2x2 blocks
def max_pool_2x2(x):  # tf.nn.max_pool. ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1]
  return tf.nn.max_pool3d(x, ksize=[1, 4, 4, 4, 1], strides=[1, 4, 4, 4, 1], padding='SAME')


## First Convolutional Layer
# Conv then Max-pooling. 1st layer will have 32 features for each 5x5 patch. (1 feature -> 32 features)
W_conv1 = weight_variable([5, 5, 5, 1, 32])  # shape of weight tensor = [5,5,1,32]
b_conv1 = bias_variable([32])  # bias vector for each output channel. = [32]

# Reshape 'x' to a 4D tensor (2nd dim=image width, 3rd dim=image height, 4th dim=nColorChannel)
x_image = tf.reshape(x, [-1,width,height,depth,1]) # [-1,28,28,1]
print(x_image.get_shape) # (?, 256, 256, 40, 1)  # -> output image: 28x28 x1

# x_image * weight tensor + bias -> apply ReLU -> apply max-pool
h_conv1 = tf.nn.relu(conv3d(x_image, W_conv1) + b_conv1)  # conv2d, ReLU(x_image * weight + bias)
print(h_conv1.get_shape) # (?, 256, 256, 40, 32)  # -> output image: 28x28 x32
h_pool1 = max_pool_2x2(h_conv1)  # apply max-pool 
print(h_pool1.get_shape) # (?, 128, 128, 20, 32)  # -> output image: 14x14 x32


## Second Convolutional Layer
# Conv then Max-pooling. 2nd layer will have 64 features for each 5x5 patch. (32 features -> 64 features)
W_conv2 = weight_variable([5, 5, 5, 32, 64]) # [5, 5, 32, 64]
b_conv2 = bias_variable([64]) # [64]

h_conv2 = tf.nn.relu(conv3d(h_pool1, W_conv2) + b_conv2)  # conv2d, .ReLU(x_image * weight + bias)
print(h_conv2.get_shape) # (?, 128, 128, 20, 64)  # -> output image: 14x14 x64
h_pool2 = max_pool_2x2(h_conv2)  # apply max-pool 
print(h_pool2.get_shape) # (?, 64, 64, 10, 64)    # -> output image: 7x7 x64


## Densely Connected Layer (or fully-connected layer)
# fully-connected layer with 1024 neurons to process on the entire image
W_fc1 = weight_variable([16*16*11*64, 1024])  # [7*7*64, 1024]
b_fc1 = bias_variable([1024]) # [1024]]

h_pool2_flat = tf.reshape(h_pool2, [-1, 16*16*11*64])  # -> output image: [-1, 7*7*64] = 3136
print(h_pool2_flat.get_shape)  # (?, 2621440)
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # ReLU(h_pool2_flat x weight + bias)
print(h_fc1.get_shape) # (?, 1024)  # -> output: 1024

## Dropout (to reduce overfitting; useful when training very large neural network)
# We will turn on dropout during training & turn off during testing
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
print(h_fc1_drop.get_shape)  # -> output: 1024

## Readout Layer
W_fc2 = weight_variable([1024, nLabel]) # [1024, 10]
b_fc2 = bias_variable([nLabel]) # [10]

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
print(y_conv.get_shape)  # -> output: 10

## Train and Evaluate the Model
# set up for optimization (optimizer:ADAM)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)  # 1e-4
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())
# Include keep_prob in feed_dict to control dropout rate.
for i in range(100):
    # Logging every 100th iteration in the training process.
    if i%5 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:train_data, y_: train_label, keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x:train_data, y_: train_label, keep_prob: 0.5})
    
print("test accuracy %g"%accuracy.eval(feed_dict={x: test_data, y_: test_label, keep_prob: 1.0}))


'''
'''
model = Sequential()
# 256 256 166
# 160*100*22
model.add(Conv3D(
    10,
    kernel_dim1=9, # depth
    kernel_dim2=9, # rows
    kernel_dim3=9, # cols
    input_shape=(3,256,256,166),
    activation='relu'
))# now 152*92*14

model.add(MaxPooling3D(
    pool_size=(2,2)
))# now 76*46*14

model.add(Conv3D(
    30,
    kernel_dim1=7, # depth
    kernel_dim2=9, # rows
    kernel_dim3=9, # cols
    activation='relu'
))# now 68*38*8

model.add(MaxPooling3D(
    pool_size=(2,2)
))# now 34*19*8

model.add(Conv3D(
    50,
    kernel_dim1=5, # depth
    kernel_dim2=9, # rows
    kernel_dim3=8, # cols
    activation='relu'
))# now 26*12*4

model.add(MaxPooling3D(
    pool_size=(2,2)
))# now 13*6*4

model.add(Conv3D(
    150,
    kernel_dim1=3, # depth
    kernel_dim2=4, # rows
    kernel_dim3=3, # cols
    activation='relu'
))# now 10*4*2

model.add(MaxPooling3D(
    pool_size=(2,2)
))# now 5*2*2

model.add(Dropout(0.25))
model.add(Flatten())

model.add(Dense(1500,activation='relu'))

model.add(Dense(750,activation='relu'))

model.add(Dense(num,activation='softmax')) #classification

# Compile
model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
'''
