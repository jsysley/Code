# -*- coding: utf-8 -*-
"""
Created on Tue May 16 18:52:25 2017

@author: jsysley
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
class DenosingNet:
    def __init__(self,lr,iterations,batch_size,n_inputs,n_hiddens,
                 n,scale = 0.1,display_epoch = 1,keep_prob = 1):
        # Hyperparameters
        self.lr = lr
        self.keep_prob = keep_prob
        self.iterations = iterations
        self.batch_size = batch_size
        self.n_inputs = n_inputs
        self.n_hiddens = n_hiddens
        self.n = n
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        self.display_epoch = display_epoch
        self.build()

    #实现标准的均匀分布
    def xavier_init(self, f_in, f_out, constant = 1):
        low = - constant * np.sqrt(6.0 / (f_in + f_out))
        high = constant * np.sqrt(6.0 / (f_in + f_out))
        return tf.random_uniform((f_in, f_out),
            minval = low, maxval = high, dtype = tf.float32)
    
    def create_wb(self):
        # input and output
        self.xs = tf.placeholder("float", [None, self.n_inputs])
        
        self.weights = {
            'encoder_h1': tf.Variable(self.xavier_init(self.n_inputs, self.n_hiddens[0])),
            'encoder_h2': tf.Variable(tf.random_normal([self.n_hiddens[0], self.n_inputs]))
            }
        self.biases = {
            'encoder_h1': tf.Variable(tf.random_normal([self.n_hiddens[0]])),
            'encoder_h2': tf.Variable(tf.random_normal([self.n_inputs]))
            }
            
    def structure(self):
        h1 = self.xs + self.scale * tf.random_normal([self.n_inputs])
        h2 = tf.add(tf.matmul(h1, self.weights['encoder_h1']),self.biases['encoder_h1'])
        self.hidden = tf.nn.softplus(h2)
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['encoder_h2']), self.biases['encoder_h2'])
    
    def compute_loss(self):
        self.loss_function = tf.reduce_mean(tf.pow(self.reconstruction - self.xs, 2))
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss_function)

    def build(self):
        # build paras
        self.create_wb()
        # build net
        self.structure()
        # build loss
        self.compute_loss()
        
    def train(self,xdata):
        init = tf.global_variables_initializer()
        # Launch the graph
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
                    # Run optimization op (backprop) and cost op (to get loss value)
                    _,loss = sess.run([self.optimizer, self.loss_function],
                                      feed_dict={self.xs: batch_x,self.scale: self.training_scale})
                    # Compute average loss
                    avg_loss += loss / total_batch
                all_loss.append(avg_loss)
                # Display logs per epoch step
                if epoch % self.display_epoch == 0:
                    print("Denosing-Epoch:", '%04d' % (epoch+1), "loss= ","%.9f" % avg_loss)
            print("Optimization Finished!")
            plt.figure()
            plt.plot(list(range(len(all_loss))), all_loss, color = 'b')
            plt.show()
            
    def test(self,xdata,scale):
        with self.sess.as_default() as sess:
            # Test model
            squre_error = tf.reduce_mean(tf.pow(self.xs - self.reconstruction, 2))
            print("Squre_Error:", squre_error.eval({self.xs: xdata,self.scale:scale}))
        
    def predict(self,tdata,scale):
        with self.sess.as_default() as sess:
            prob = sess.run(self.reconstruction,feed_dict = {self.xs:tdata,self.scale:scale})
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
            
    def get(self,name,xdata,scale):
        with self.sess.as_default() as sess:
            exec('self.res = sess.run(self.' + name + ',feed_dict = {self.xs:xdata,self.scale: scale})')
            return self.res

#==============================================================================
#==============================================================================
from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
de = DenosingNet(lr = 0.01,keep_prob = 1,iterations = 20,batch_size = 256,
          n_hiddens=[256,128],n_inputs = 784,scale = 0.1,n = 55000,display_epoch = 1)
de.train(mnist.train.images)
de.test(mnist.test.images,scale = 0.1)
de.predict(mnist.train.images,scale = 0.1)
de.get('hidden',mnist.train.images,scale = 0.1)
de.save(r'F:/code/python3/DeepLearning/template/record/denosingNet.ckpt')

de1 = DenosingNet(lr = 0.01,keep_prob = 1,iterations = 20,batch_size = 256,
          n_hiddens=[256,128],n_inputs = 784,scale = 0.1,n = 55000,display_epoch = 1)
de1.load(r'F:/code/python3/DeepLearning/template/record/denosingNet.ckpt')
de1.test(mnist.test.images,scale = 0.1)

