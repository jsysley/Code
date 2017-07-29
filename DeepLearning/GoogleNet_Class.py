# -*- coding: utf-8 -*-
"""
Created on Sun May 21 12:27:01 2017

@author: jsysley
"""

import time
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
import math
class GoogleNet:
    def __init__(self,lr,iterations,batch_size,n_inputs,
                 n_classes,n,display_epoch = 1,keep_prob = 1):
        # Hyperparameters
        self.lr = lr
        self.keep_prob = keep_prob
        self.iterations = iterations
        self.batch_size = batch_size
        self.n_inputs = n_inputs
        self.n_classes = n_classes
        self.n = n
        self.display_epoch = display_epoch
        self.build()
    
    def inception_unit(self,inputdata, weights, biases):
        # A3 inception 3a
        inception_in = inputdata

        # Conv 1x1+S1
        inception_1x1_S1 = tf.nn.conv2d(inception_in, weights['inception_1x1_S1'], strides=[1,1,1,1], padding='SAME')
        inception_1x1_S1 = tf.nn.bias_add(inception_1x1_S1, biases['inception_1x1_S1'])
        inception_1x1_S1 = tf.nn.relu(inception_1x1_S1)
        # Conv 3x3+S1
        inception_3x3_S1_reduce = tf.nn.conv2d(inception_in, weights['inception_3x3_S1_reduce'], strides=[1,1,1,1], padding='SAME')
        inception_3x3_S1_reduce = tf.nn.bias_add(inception_3x3_S1_reduce, biases['inception_3x3_S1_reduce'])
        inception_3x3_S1_reduce = tf.nn.relu(inception_3x3_S1_reduce)
        inception_3x3_S1 = tf.nn.conv2d(inception_3x3_S1_reduce, weights['inception_3x3_S1'], strides=[1,1,1,1], padding='SAME')
        inception_3x3_S1 = tf.nn.bias_add(inception_3x3_S1, biases['inception_3x3_S1'])
        inception_3x3_S1 = tf.nn.relu(inception_3x3_S1)
        # Conv 5x5+S1
        inception_5x5_S1_reduce = tf.nn.conv2d(inception_in, weights['inception_5x5_S1_reduce'], strides=[1,1,1,1], padding='SAME')
        inception_5x5_S1_reduce = tf.nn.bias_add(inception_5x5_S1_reduce, biases['inception_5x5_S1_reduce'])
        inception_5x5_S1_reduce = tf.nn.relu(inception_5x5_S1_reduce)
        inception_5x5_S1 = tf.nn.conv2d(inception_5x5_S1_reduce, weights['inception_5x5_S1'], strides=[1,1,1,1], padding='SAME')
        inception_5x5_S1 = tf.nn.bias_add(inception_5x5_S1, biases['inception_5x5_S1'])
        inception_5x5_S1 = tf.nn.relu(inception_5x5_S1)
        # MaxPool
        inception_MaxPool = tf.nn.max_pool(inception_in, ksize=[1,3,3,1], strides=[1,1,1,1], padding='SAME')
        inception_MaxPool = tf.nn.conv2d(inception_MaxPool, weights['inception_MaxPool'], strides=[1,1,1,1], padding='SAME')
        inception_MaxPool = tf.nn.bias_add(inception_MaxPool, biases['inception_MaxPool'])
        inception_MaxPool = tf.nn.relu(inception_MaxPool)
        # Concat
        #tf.concat(concat_dim, values, name='concat')
        #concat_dim 是 tensor 连接的方向（维度）， values 是要连接的 tensor 链表， name 是操作名。 cancat_dim 维度可以不一样，其他维度的尺寸必须一样。
        inception_out = tf.concat(axis=3, values=[inception_1x1_S1, inception_3x3_S1, inception_5x5_S1, inception_MaxPool])
        return inception_out

    def create_wb(self):
        # input and output
        self.xs = tf.placeholder("float", [None, self.n_inputs])
        self.ys = tf.placeholder("float", [None, self.n_classes])
        
        self.weights = {
            'conv1_7x7_S2': tf.Variable(tf.random_normal([7,7,4,64])),
            'conv2_1x1_S1': tf.Variable(tf.random_normal([1,1,64,64])),
            'conv2_3x3_S1': tf.Variable(tf.random_normal([3,3,64,192])),
            'FC2': tf.Variable(tf.random_normal([7*7*1024, self.n_classes]))
            }

        self.biases = {
            'conv1_7x7_S2': tf.Variable(tf.random_normal([64])),
            'conv2_1x1_S1': tf.Variable(tf.random_normal([64])),
            'conv2_3x3_S1': tf.Variable(tf.random_normal([192])),
            'FC2': tf.Variable(tf.random_normal([self.n_classes]))
            }

        self.pooling = {
            'pool1_3x3_S2': [1,3,3,1],
            'pool2_3x3_S2': [1,3,3,1],
            'pool3_3x3_S2': [1,3,3,1],
            'pool4_3x3_S2': [1,3,3,1]
            }

        self.conv_W_3a = {
            'inception_1x1_S1': tf.Variable(tf.random_normal([1,1,192,64])),
            'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([1,1,192,96])),
            'inception_3x3_S1': tf.Variable(tf.random_normal([1,1,96,128])),
            'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([1,1,192,16])),
            'inception_5x5_S1': tf.Variable(tf.random_normal([5,5,16,32])),
            'inception_MaxPool': tf.Variable(tf.random_normal([1,1,192,32]))
            }

        self.conv_B_3a = {
            'inception_1x1_S1': tf.Variable(tf.random_normal([64])),
            'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([96])),
            'inception_3x3_S1': tf.Variable(tf.random_normal([128])),
            'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([16])),
            'inception_5x5_S1': tf.Variable(tf.random_normal([32])),
            'inception_MaxPool': tf.Variable(tf.random_normal([32]))
            }

        self.conv_W_3b = {
            'inception_1x1_S1': tf.Variable(tf.random_normal([1,1,256,128])),
            'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([1,1,256,128])),
            'inception_3x3_S1': tf.Variable(tf.random_normal([1,1,128,192])),
            'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([1,1,256,32])),
            'inception_5x5_S1': tf.Variable(tf.random_normal([5,5,32,96])),
            'inception_MaxPool': tf.Variable(tf.random_normal([1,1,256,64]))
            }

        self.conv_B_3b = {
            'inception_1x1_S1': tf.Variable(tf.random_normal([128])),
            'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([128])),
            'inception_3x3_S1': tf.Variable(tf.random_normal([192])),
            'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([32])),
            'inception_5x5_S1': tf.Variable(tf.random_normal([96])),
            'inception_MaxPool': tf.Variable(tf.random_normal([64]))
            }

        self.conv_W_4a = {
            'inception_1x1_S1': tf.Variable(tf.random_normal([1,1,480,192])),
            'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([1,1,480,96])),
            'inception_3x3_S1': tf.Variable(tf.random_normal([1,1,96,208])),
            'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([1,1,480,16])),
            'inception_5x5_S1': tf.Variable(tf.random_normal([5,5,16,48])),
            'inception_MaxPool': tf.Variable(tf.random_normal([1,1,480,64]))
            }

        self.conv_B_4a = {
            'inception_1x1_S1': tf.Variable(tf.random_normal([192])),
            'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([96])),
            'inception_3x3_S1': tf.Variable(tf.random_normal([208])),
            'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([16])),
            'inception_5x5_S1': tf.Variable(tf.random_normal([48])),
            'inception_MaxPool': tf.Variable(tf.random_normal([64]))
            }

        self.conv_W_4b = {
            'inception_1x1_S1': tf.Variable(tf.random_normal([1,1,512,160])),
            'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([1,1,512,112])),
            'inception_3x3_S1': tf.Variable(tf.random_normal([1,1,112,224])),
            'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([1,1,512,24])),
            'inception_5x5_S1': tf.Variable(tf.random_normal([5,5,24,64])),
            'inception_MaxPool': tf.Variable(tf.random_normal([1,1,512,64]))
            }

        self.conv_B_4b = {
            'inception_1x1_S1': tf.Variable(tf.random_normal([160])),
            'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([112])),
            'inception_3x3_S1': tf.Variable(tf.random_normal([224])),
            'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([24])),
            'inception_5x5_S1': tf.Variable(tf.random_normal([64])),
            'inception_MaxPool': tf.Variable(tf.random_normal([64]))
            }

        self.conv_W_4c = {
            'inception_1x1_S1': tf.Variable(tf.random_normal([1,1,512,128])),
            'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([1,1,512,128])),
            'inception_3x3_S1': tf.Variable(tf.random_normal([1,1,128,256])),
            'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([1,1,512,24])),
            'inception_5x5_S1': tf.Variable(tf.random_normal([5,5,24,64])),
            'inception_MaxPool': tf.Variable(tf.random_normal([1,1,512,64]))
            }

        self.conv_B_4c = {
            'inception_1x1_S1': tf.Variable(tf.random_normal([128])),
            'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([128])),
            'inception_3x3_S1': tf.Variable(tf.random_normal([256])),
            'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([24])),
            'inception_5x5_S1': tf.Variable(tf.random_normal([64])),
            'inception_MaxPool': tf.Variable(tf.random_normal([64]))
            }

        self.conv_W_4d = {
            'inception_1x1_S1': tf.Variable(tf.random_normal([1,1,512,112])),
            'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([1,1,512,144])),
            'inception_3x3_S1': tf.Variable(tf.random_normal([1,1,144,288])),
            'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([1,1,512,32])),
            'inception_5x5_S1': tf.Variable(tf.random_normal([5,5,32,64])),
            'inception_MaxPool': tf.Variable(tf.random_normal([1,1,512,64]))
            }

        self.conv_B_4d = {
            'inception_1x1_S1': tf.Variable(tf.random_normal([112])),
            'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([144])),
            'inception_3x3_S1': tf.Variable(tf.random_normal([288])),
            'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([32])),
            'inception_5x5_S1': tf.Variable(tf.random_normal([64])),
            'inception_MaxPool': tf.Variable(tf.random_normal([64]))
            }

        self.conv_W_4e = {
            'inception_1x1_S1': tf.Variable(tf.random_normal([1,1,528,256])),
            'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([1,1,528,160])),
            'inception_3x3_S1': tf.Variable(tf.random_normal([1,1,160,320])),
            'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([1,1,528,32])),
            'inception_5x5_S1': tf.Variable(tf.random_normal([5,5,32,128])),
            'inception_MaxPool': tf.Variable(tf.random_normal([1,1,528,128]))
            }

        self.conv_B_4e = {
            'inception_1x1_S1': tf.Variable(tf.random_normal([256])),
            'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([160])),
            'inception_3x3_S1': tf.Variable(tf.random_normal([320])),
            'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([32])),
            'inception_5x5_S1': tf.Variable(tf.random_normal([128])),
            'inception_MaxPool': tf.Variable(tf.random_normal([128]))
            }

        self.conv_W_5a = {
            'inception_1x1_S1': tf.Variable(tf.random_normal([1,1,832,256])),
            'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([1,1,832,160])),
            'inception_3x3_S1': tf.Variable(tf.random_normal([1,1,160,320])),
            'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([1,1,832,32])),
            'inception_5x5_S1': tf.Variable(tf.random_normal([5,5,32,128])),
            'inception_MaxPool': tf.Variable(tf.random_normal([1,1,832,128]))
            }

        self.conv_B_5a = {
            'inception_1x1_S1': tf.Variable(tf.random_normal([256])),
            'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([160])),
            'inception_3x3_S1': tf.Variable(tf.random_normal([320])),
            'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([32])),
            'inception_5x5_S1': tf.Variable(tf.random_normal([128])),
            'inception_MaxPool': tf.Variable(tf.random_normal([128]))
            }

        self.conv_W_5b = {
            'inception_1x1_S1': tf.Variable(tf.random_normal([1,1,832,384])),
            'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([1,1,832,192])),
            'inception_3x3_S1': tf.Variable(tf.random_normal([1,1,192,384])),
            'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([1,1,832,48])),
            'inception_5x5_S1': tf.Variable(tf.random_normal([5,5,48,128])),
            'inception_MaxPool': tf.Variable(tf.random_normal([1,1,832,128]))
            }

        self.conv_B_5b = {
            'inception_1x1_S1': tf.Variable(tf.random_normal([384])),
            'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([192])),
            'inception_3x3_S1': tf.Variable(tf.random_normal([384])),
            'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([48])),
            'inception_5x5_S1': tf.Variable(tf.random_normal([128])),
            'inception_MaxPool': tf.Variable(tf.random_normal([128]))
            }
                    
    def structure(self):
        # A0 输入数据
        x = tf.reshape(self.xs,[-1,224,224,4])  # 调整输入数据维度格式

        # A1  Conv 7x7_S2
        x = tf.nn.conv2d(x, self.weights['conv1_7x7_S2'], strides = [1,2,2,1], padding = 'SAME')
        # 卷积层 卷积核 7*7 扫描步长 2*2 
        x = tf.nn.bias_add(x, self.biases['conv1_7x7_S2'])
        #print (x.get_shape().as_list())
        # 偏置向量
        x = tf.nn.relu(x)
        # 激活函数
        x = tf.nn.max_pool(x, ksize = self.pooling['pool1_3x3_S2'], strides = [1,2,2,1], padding = 'SAME')
        # 池化取最大值
        x = tf.nn.local_response_normalization(x, depth_radius = 5/2.0, bias = 2.0, alpha = 1e-4, beta = 0.75)
        # 局部响应归一化

        # A2
        x = tf.nn.conv2d(x, self.weights['conv2_1x1_S1'], strides = [1,1,1,1], padding = 'SAME')
        x = tf.nn.bias_add(x, self.biases['conv2_1x1_S1'])
        x = tf.nn.conv2d(x, self.weights['conv2_3x3_S1'], strides = [1,1,1,1], padding = 'SAME')
        x = tf.nn.bias_add(x, self.biases['conv2_3x3_S1'])
        x = tf.nn.local_response_normalization(x, depth_radius = 5/2.0, bias = 2.0, alpha = 1e-4, beta = 0.75)
        x = tf.nn.max_pool(x, ksize = self.pooling['pool2_3x3_S2'], strides = [1,2,2,1], padding = 'SAME')

        # inception 3
        inception_3a = self.inception_unit(inputdata = x, weights = self.conv_W_3a, biases = self.conv_B_3a)
        inception_3b = self.inception_unit(inception_3a, weights = self.conv_W_3b, biases = self.conv_B_3b)

        # 池化层
        x = inception_3b
        x = tf.nn.max_pool(x, ksize = self.pooling['pool3_3x3_S2'], strides = [1,2,2,1], padding = 'SAME' )

        # inception 4
        inception_4a = self.inception_unit(inputdata = x, weights = self.conv_W_4a, biases = self.conv_B_4a)
        # 引出第一条分支
        #softmax0 = inception_4a
        inception_4b = self.inception_unit(inception_4a, weights = self.conv_W_4b, biases = self.conv_B_4b)    
        inception_4c = self.inception_unit(inception_4b, weights = self.conv_W_4c, biases = self.conv_B_4c)
        inception_4d = self.inception_unit(inception_4c, weights = self.conv_W_4d, biases = self.conv_B_4d)
        # 引出第二条分支
        #softmax1 = inception_4d
        inception_4e = self.inception_unit(inception_4d, weights = self.conv_W_4e, biases = self.conv_B_4e)

        # 池化
        x = inception_4e
        x = tf.nn.max_pool(x, ksize = self.pooling['pool4_3x3_S2'], strides = [1,2,2,1], padding = 'SAME' )

        # inception 5
        inception_5a = self.inception_unit(x, weights = self.conv_W_5a, biases = self.conv_B_5a)
        inception_5b = self.inception_unit(inception_5a, weights = self.conv_W_5b, biases = self.conv_B_5b)
        softmax2 = inception_5b

        # 后连接
        softmax2 = tf.nn.avg_pool(softmax2, ksize = [1,7,7,1], strides = [1,1,1,1], padding = 'SAME')
        softmax2 = tf.nn.dropout(softmax2, keep_prob = 0.4)
        softmax2 = tf.reshape(softmax2, [-1,self.weights['FC2'].get_shape().as_list()[0]])
        softmax2 = tf.nn.bias_add(tf.matmul(softmax2,self.weights['FC2']),self.biases['FC2'])
        #print(softmax2.get_shape().as_list())
        #return softmax2 
        self.pred = softmax2

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
                                      feed_dict={self.xs: batch_x,self.ys: batch_y})
                    # Compute average loss
                    avg_loss += loss / total_batch
                all_loss.append(avg_loss)
                # Display logs per epoch step
                if epoch % self.display_epoch == 0:
                    print(time.strftime('%Y-%m-%d,%H:%M:%S',time.localtime()),"CNN-Epoch:", '%04d' % (epoch+1), "cost= ","%.9f" % avg_loss)
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
            print("Accuracy:", accuracy.eval({self.xs: xdata, self.ys: ydata}))
        
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
#from tensorflow.examples.tutorials.mnist import input_data
## number 1 to 10 data
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
goo = GoogleNet(lr = 1e-3,keep_prob = 0.8,iterations = 1,batch_size = 64,
          n_inputs = 784,n_classes = 10,n = 55000,display_epoch = 1)
goo.train(mnist.train.images,mnist.train.labels)
