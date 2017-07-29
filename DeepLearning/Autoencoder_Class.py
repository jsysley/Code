# -*- coding: utf-8 -*-
"""
Created on Tue May 16 18:52:25 2017

@author: jsysley
"""
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
class Autoencoder:
    def __init__(self,lr,iterations,batch_size,n_inputs,n_hiddens,
                 n_outputs,n,display_epoch = 1,keep_prob = 1):
        # Hyperparameters
        self.lr = lr
        self.keep_prob = keep_prob
        self.iterations = iterations
        self.batch_size = batch_size
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hiddens = n_hiddens
        self.n = n
        self.display_epoch = display_epoch
        self.build()
    
    # 一种参数初始化方法，Xavier初始化器会根据某一层网络的输入，输出节点数量自动调整最适合的分布。
    def xavier_init(self,fan_in, fan_out, constant = 1):
        low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
        high = constant * np.sqrt(6.0 / (fan_in + fan_out))
        return tf.random_uniform((fan_in, fan_out),
                                 minval = low, maxval = high,
                                 dtype = tf.float32)
    
    # 标准化
    def standard_scale(self,xtrain,xtest):
        preprocessor = StandardScaler().fit(xtrain)
        xstrain = preprocessor.transform(xtrain)
        xstest = preprocessor.transform(xtest)
        return xstrain, xstest
        
    def create_wb(self):
        # input and output
        self.xs = tf.placeholder("float", [None, self.n_inputs])
        self.ys = tf.placeholder("float", [None, self.n_outputs])
        
        self.weights = {
            'encoder_h1': tf.Variable(tf.random_normal([self.n_inputs, self.n_hiddens[0]])),
            'encoder_h2': tf.Variable(tf.random_normal([self.n_hiddens[0], self.n_hiddens[1]])),
            'decoder_h1': tf.Variable(tf.random_normal([self.n_hiddens[1], self.n_hiddens[0]])),
            'decoder_h2': tf.Variable(tf.random_normal([self.n_hiddens[0], self.n_outputs])),
            }
        self.biases = {
            'encoder_b1': tf.Variable(tf.random_normal([self.n_hiddens[0]])),
            'encoder_b2': tf.Variable(tf.random_normal([self.n_hiddens[1]])),
            'decoder_b1': tf.Variable(tf.random_normal([self.n_hiddens[0]])),
            'decoder_b2': tf.Variable(tf.random_normal([self.n_outputs])),
            }
            
    def structure(self):
        # Encoder Hidden layer with sigmoid activation #1
        self.layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(self.xs, self.weights['encoder_h1']),self.biases['encoder_b1']))
        
        # Decoder Hidden layer with sigmoid activation #2
        self.layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(self.layer_1, self.weights['encoder_h2']),self.biases['encoder_b2']))
        
        # Encoder Hidden layer with sigmoid activation #1
        self.layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(self.layer_2, self.weights['decoder_h1']),self.biases['decoder_b1']))
        
        # Decoder Hidden layer with sigmoid activation #2
        self.pred = tf.nn.sigmoid(tf.add(tf.matmul(self.layer_3, self.weights['decoder_h2']),self.biases['decoder_b2']))
    
    def compute_loss(self):
        self.loss_function = tf.reduce_mean(tf.pow(self.ys - self.pred, 2))
        self.optimizer = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss_function)

    def build(self):
        # build paras
        self.create_wb()
        # build net
        self.structure()
        # build loss
        self.compute_loss()
        
    def train(self,xdata,ydata):
        init = tf.global_variables_initializer()
        # Launch the graph
        self.sess = tf.Session()
        with tf.device("gpu:%d" % 4):
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
                        #batch_y = ydata[(i * self.batch_size):((i + 1) * self.batch_size)]
                        batch_y = batch_x
                        # Run optimization op (backprop) and cost op (to get loss value)
                        _,loss = sess.run([self.optimizer, self.loss_function],
                                          feed_dict={self.xs: batch_x,self.ys: batch_y})
                        # Compute average loss
                        avg_loss += loss / total_batch
                    all_loss.append(avg_loss)
                    # Display logs per epoch step
                    if epoch % self.display_epoch == 0:
                        print(time.strftime('%Y-%m-%d,%H:%M:%S',time.localtime()),"AutoEncoder-Epoch:", '%04d' % (epoch+1), "cost=","%.9f" % avg_loss)
            print("Optimization Finished!")
            plt.figure()
            plt.plot(list(range(len(all_loss))), all_loss, color = 'b')
            plt.show()
            
    def test(self,xdata,ydata):
        with self.sess.as_default() as sess:
            # Test model
            squre_error = tf.reduce_mean(tf.pow(self.ys - self.pred, 2))
            print("Squre_Error:", squre_error.eval({self.xs: xdata, self.ys: ydata}))
        
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
            
    def get(self,name,xdata):
        with self.sess.as_default() as sess:
            exec('self.res = sess.run(self.' + name + ',feed_dict = {self.xs:xdata})')
            return self.res
#==============================================================================
#==============================================================================
from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
auto = Autoencoder(lr = 0.01,keep_prob = 1,iterations = 20,batch_size = 20,
          n_hiddens=[256,128],n_inputs = 784,n_outputs = 784,n = 55000,display_epoch = 1)
#x_tr,x_te = auto.standard_scale(mnist.train.images,mnist.test.images)
auto.train(mnist.train.images,mnist.train.images)
auto.test(mnist.test.images,mnist.test.images)
auto.get('layer_2',mnist.train.images)
auto.save(r'F:/code/python3/DeepLearning/template/record/auto.ckpt')
auto1 = Autoencoder(lr = 0.01,keep_prob = 1,iterations = 20,batch_size = 20,
          n_hiddens=[256,128],n_inputs = 784,n_outputs = 784,n = 55000,display_epoch = 1)
auto1.load(r'F:/code/python3/DeepLearning/template/record/auto.ckpt')
auto1.get('layer_2',mnist.train.images)
auto1.test(mnist.test.images,mnist.test.images)

