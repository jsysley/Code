# -*- coding: utf-8 -*-
"""
Created on Wed May 10 23:01:46 2017

@author: jsysley
"""
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
tf.set_random_seed(1)   # set random seed

# Load data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# hyperparameters
lr = 0.001                  # learning rate
iterations = 100     # train step upper bound
batch_size = 128            
n_inputs = 28               # MNIST data input (img shape: 28*28)
n_steps = 28                # time steps
n_hidden_units = 128        # neurons in hidden layer
n_classes = 10              # MNIST classes (0-9 digits)
n = 55000                   # Samples size
display_epoch = 1           # display the result

# x y placeholder
xs = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
ys = tf.placeholder(tf.float32, [None, n_classes])

# Initialize weights biases 
weights = {
    # shape (28, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # shape (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    # shape (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # shape (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}

# Define RNN structure
def RNN(X, weights, biases):
    # 原始的 X 是 3 维数据, 我们需要把它变成 2 维数据才能使用 weights 的矩阵乘法
    # X ==> (128 batches * 28 steps, 28 inputs)
    X = tf.reshape(X, [-1, n_inputs])

    # X_in = W*X + b
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (128 batches, 28 steps, 128 hidden) 换回3维
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # 使用 basic LSTM Cell.
    with tf.variable_scope("cell") as fscope:
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
        init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32) # 初始化全零 state
    #如果 inputs 为 (batches, steps, inputs) ==> time_major=False;
    #如果 inputs 为 (steps, batches, inputs) ==> time_major=True;
    try:
        with tf.variable_scope(fscope,reuse = True):   
            outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)
    except ValueError:
        with tf.variable_scope(fscope):   
            outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)
    #最后是 output_layer 和 return 的值
    results = tf.matmul(final_state[1], weights['out']) + biases['out']
    return results

# Cost
pred = RNN(xs, weights, biases)
loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=ys))
train_op = tf.train.AdamOptimizer(lr).minimize(loss_function)

# Train
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(ys, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 替换成下面的写法:
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    size = int(n / batch_size)
    all_loss = []
    for epoch in range(iterations):
        avg_loss = 0
        for i in range(size - 1):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
            if epoch == 1:
                feed = {xs: batch_xs,ys: batch_ys}
            else:
                feed = {xs: batch_xs,ys: batch_ys}
            _,loss = sess.run([train_op,loss_function], feed_dict=feed)
            avg_loss += loss / size
        all_loss.append(avg_loss)
        if epoch % display_epoch == 0:
            print("RNN-Epoch:", '%04d' % (epoch+1), "cost= ","%.9f" % avg_loss)
    print("Optimization Finished!")
    print("Accuracy: ",sess.run(accuracy, feed_dict={xs: batch_xs,ys: batch_ys,}))
    plt.figure()
    plt.plot(list(range(len(all_loss))), all_loss, color = 'b')
    plt.show()
    #print(sess.run(accuracy, feed_dict={x: batch_xs,y: batch_ys,}))