# -*- coding: utf-8 -*-
"""
Created on Tue May 16 18:52:25 2017

@author: jsysley
"""
import time
import tensorflow as tf
import matplotlib.pyplot as plt
class CNNDecay:
    def __init__(self,lr,iterations,batch_size,n_inputs,n_hiddens,
                 n_classes,n,display_epoch = 1,keep_prob = 1):
        # Hyperparameters
        self.lr = lr
        self.keep_prob = keep_prob
        self.iterations = iterations
        self.batch_size = batch_size
        self.n_inputs = n_inputs
        self.n_hiddens = n_hiddens
        self.n_classes = n_classes
        self.n = n
        self.display_epoch = display_epoch
        self.build()
    
    def conv2d(self,x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self,x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

    def variable_with_weight_loss(self,shape, stddev, w1):
	    var = tf.Variable(tf.truncated_normal(shape, stddev = stddev))
	    if w1 is not None:
	        weight_loss = tf.multiply(tf.nn.l2_loss(var), w1, name = 'weight_loss')
	        tf.add_to_collection('losses', weight_loss)
	    return var

    def create_wb(self):
        # input and output
        self.xs = tf.placeholder("float", [None, self.n_inputs])
        self.ys = tf.placeholder("float", [None, self.n_classes])
        self.dropout = tf.placeholder("float")
        
        self.weights = {
            'conv1': self.variable_with_weight_loss(shape = [5, 5, 1, 32], stddev = 5e-2, w1 = 0.0),
            'conv2': self.variable_with_weight_loss(shape = [5, 5, 32, 64], stddev = 5e-2, w1 = 0.0),
            'fc1' : self.variable_with_weight_loss(shape = [7*7*64,256], stddev = 5e-2, w1 = 0.0),
            'out': self.variable_with_weight_loss(shape = [256,self.n_classes], stddev = 5e-2, w1 = 0.0)
            }
            
        self.biases = {
            'conv_b1': tf.Variable(tf.random_normal([32])),
            'conv_b2': tf.Variable(tf.random_normal([64])),
            'fc1_b': tf.Variable(tf.random_normal([256])),
            'out_b': tf.Variable(tf.random_normal([self.n_classes]))
            }
            
    def structure(self):
        # reshape
        x_ = tf.reshape(self.xs, [-1,28,28,1])
        # conv1
        h_conv1 = tf.nn.relu(self.conv2d(x_, self.weights['conv1']) + self.biases['conv_b1'])
        h_pool1 = self.max_pool_2x2(h_conv1)
        h_pool1 = tf.nn.lrn(h_pool1, 4, bias = 1.0, alpha = 0.001 / 9.0, beta = 0.75)
        # conv2
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, self.weights['conv2']) + self.biases['conv_b2'])
        h_pool2 = self.max_pool_2x2(h_conv2)
        h_pool2 = tf.nn.lrn(h_pool2, 4, bias = 1.0, alpha = 0.001 / 9.0, beta = 0.75)
        # fc1
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, self.weights['fc1']) + self.biases['fc1_b'])
        # dropout
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
        # fc2
        self.pred = tf.matmul(h_fc1_drop, self.weights['out']) + self.biases['out_b']
            
    def compute_loss(self):
        self.loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.ys))
        tf.add_to_collection('losses', self.loss_function)
        self.total_loss = tf.add_n(tf.get_collection('losses'), name = 'total_loss')
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.total_loss)

    def build(self):
        # build paras
        self.create_wb()
        # build net
        self.structure()
        # build loss
        self.compute_loss()
        
    def train(self,xdata,ydata):
        init = tf.global_variables_initializer()
        
        self.sess = tf.Session()
        with self.sess.as_default() as sess:
            sess.run(init)
            # Training cycle
            all_loss = []
            total_batch = int(self.n / self.batch_size)
            for epoch in range(self.iterations):
                avg_loss = 0.
                # Loop over all batches
                for i in range(total_batch):
                    batch_x = xdata[(i * self.batch_size):((i + 1) * self.batch_size)]
                    batch_y = ydata[(i * self.batch_size):((i + 1) * self.batch_size)]
                    # Run optimization op (backprop) and cost op (to get loss value)
                    _,loss = sess.run([self.optimizer, self.loss_function],
                                      feed_dict={self.xs: batch_x,self.ys: batch_y,self.dropout:self.keep_prob})
                    # Compute average loss
                    avg_loss += loss / total_batch
                all_loss.append(avg_loss)
                # Display logs per epoch step
                if epoch % self.display_epoch == 0:
                    print(time.strftime('%Y-%m-%d,%H:%M:%S',time.localtime()),"CNNDecay-Epoch:", '%04d' % (epoch+1), "loss= ","%.9f" % avg_loss)
            print("Optimization Finished!")
            plt.figure()
            plt.plot(list(range(len(all_loss))), all_loss, color = 'b')
            plt.show()
            
    def test(self,xdata,ydata):
        with self.sess.as_default() as sess:
            # Test model
            correct_prediction = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.ys, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print("Accuracy:", accuracy.eval({self.xs: xdata, self.ys: ydata,self.dropout:1.0}))
        
    def predict(self,tdata):
        with self.sess.as_default() as sess:
            prob = sess.run(self.pred,feed_dict = {self.xs:tdata})
        return prob
    
    def save(self,save_path):
        saver = tf.train.Saver()
        with self.sess.as_default() as sess:
            print("Save to path: " + saver.save(sess,save_path))
        
    def load(self,load_path):
        saver = tf.train.Saver()
        self.sess = tf.Session()
        with self.sess.as_default() as sess:
            saver.restore(sess, load_path) 
            print("Load from path: " + load_path)
#==============================================================================
#==============================================================================
from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
cnnd = CNNDecay(lr = 1e-4,keep_prob = 1,iterations = 1,batch_size = 100,
          n_inputs = 784,n_hiddens = 128,n_classes = 10,n = 55000,display_epoch = 1)
cnnd.train(mnist.train.images,mnist.train.labels)
cnnd.test(mnist.test.images,mnist.test.labels)
cnnd.predict(mnist.test.images)
cnnd.save(r'F:/code/python3/DeepLearning/template/record/model.ckpt')

cnnd1 = CNNDecay(lr = 1e-4,keep_prob = 1,iterations = 1,batch_size = 100,
          n_inputs = 784,n_hiddens = 128,n_classes = 10,n = 55000,display_epoch = 1)
cnnd1.load(r'F:/code/python3/DeepLearning/template/record/model.ckpt')
cnnd1.test(mnist.test.images,mnist.test.labels)

