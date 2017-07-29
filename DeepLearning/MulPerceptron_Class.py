# -*- coding: utf-8 -*-
"""
Created on Tue May 16 18:52:25 2017

@author: jsysley
"""
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
class MulPerceptron:
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
        self.ys = tf.placeholder("float", [None, self.n_classes])
        self.dropout = tf.placeholder("float")
        
        self.weights = {
            'h1': tf.Variable(self.xavier_init(self.n_inputs, self.n_hiddens[0])),
            'h2': tf.Variable(self.xavier_init(self.n_hiddens[0], self.n_hiddens[1])),
            'h3': tf.Variable(self.xavier_init(self.n_hiddens[1], self.n_hiddens[2])),
            'out': tf.Variable(self.xavier_init(self.n_hiddens[2], self.n_classes)),
            }
        self.biases = {
            'h1': tf.Variable(tf.random_normal([self.n_hiddens[0]])),
            'h2': tf.Variable(tf.random_normal([self.n_hiddens[1]])),
            'h3': tf.Variable(tf.random_normal([self.n_hiddens[2]])),
            'out': tf.Variable(tf.random_normal([self.n_classes])),
            }
            
    def structure(self):
        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(self.xs, self.weights['h1']), self.biases['h1'])
        layer_1 = tf.nn.relu(layer_1)
        # Hidden layer with RELU activation
        layer_2 = tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['h2'])
        layer_2 = tf.nn.relu(layer_2)
        # Hidden layer with RELU activation
        layer_3 = tf.add(tf.matmul(layer_2, self.weights['h3']), self.biases['h3'])
        layer_3 = tf.nn.relu(layer_3)
        # dropout
        layer_3 = tf.nn.dropout(layer_3, self.dropout)
        # Output layer with linear activation
        self.pred = tf.matmul(layer_3, self.weights['out']) + self.biases['out']
    
    def compute_loss(self):
        self.loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.ys))
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss_function)

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
                    print(time.strftime('%m-%d,%H:%M:%S',time.localtime()),"MulPerceptron-Epoch:", '%04d' % (epoch+1), "loss= ","%.9f" % avg_loss)
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
            
    def get(self,name,xdata):
        with self.sess.as_default() as sess:
            exec('self.res = sess.run(self.' + name + ',feed_dict = {self.xs:xdata,self.dropout:1.0})')
            return self.res
#==============================================================================
#==============================================================================
from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
per = MulPerceptron(lr = 0.01,keep_prob = 1,iterations = 20,batch_size = 100,
          n_hiddens=[256,128,64],n_classes = 10,n_inputs = 784,n = 55000,display_epoch = 1)
per.train(mnist.train.images,mnist.train.labels)
per.test(mnist.test.images,mnist.test.labels)
per.save(r'F:/code/python3/DeepLearning/template/record')
per1 = MulPerceptron(lr = 0.01,keep_prob = 1,iterations = 20,batch_size = 256,
          n_hiddens=[256,128],n_inputs = 784,n = 55000,display_epoch = 1)
per1.load(r'F:/code/python3/DeepLearning/template/record')
per1.test(mnist.test.images,mnist.test.labels)

