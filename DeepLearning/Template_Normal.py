# -*- coding: utf-8 -*-
"""
Created on Wed May 10 16:35:10 2017

@author: jsysley
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Hyperparameters
lr = 0.5                  # learning rate
training_iters = 1000     # train step upper bound
batch_size = 100            
n_inputs = 784              # MNIST data input (img shape: 28*28)
n_hidden_units = 128        # neurons in hidden layer
n_classes = 10              # MNIST classes (0-9 digits)

# Definition
def Add_Layer(inputs, in_size, out_size, activation_function=None,keep_prob=1):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]),dtype=tf.float32)
    biases = tf.Variable(tf.zeros([out_size]) + 0.1,dtype=tf.float32)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs
    
def Accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result
    
# Load data
#x_data = 
#y_data = 

# Define placeholder for inputs to network
keep_prob = tf.placeholder(tf.float32)
xs = tf.placeholder(tf.float32, [None, 784])  # change
ys = tf.placeholder(tf.float32, [None, 10])   # change

# add output layer
#l1 = Add_Layer(xs, 784, 500, activation_function = tf.nn.softmax)
prediction = Add_Layer(xs, 784, 10, activation_function = tf.nn.softmax)

# Loss function
loss_function = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))

# Give the optimizer and the learning rate
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss_function)

# Initialize
init = tf.global_variables_initializer()
#Run
with tf.Session() as sess:
    sess.run(init)
    for i in range(training_iters):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # here to determine the keeping probability
        _,loss = sess.run([train_step,loss_function],feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 1})
        if i % 50 == 0:
            # record loss
            print("Step ",repr(i),"th loss:",loss)
    saver.save(sess,path)
    print(Accuracy(mnist.test.images, mnist.test.labels))
    

