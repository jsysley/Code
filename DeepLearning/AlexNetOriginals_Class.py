# -*- coding: utf-8 -*-
"""
Created on Tue May 16 18:52:25 2017

@author: jsysley
"""
import time
import tensorflow as tf
import matplotlib.pyplot as plt
class AlexNetOriginal:
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
    # convolution
    def conv2d(self,name,x, W,b,k = 1):
        return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, W, strides = [1, k, k, 1], padding = 'SAME'),b), name = name)
    # pooling
    def max_pool(self,name,x,k,s,pad = 'SAME'):
        return tf.nn.max_pool(x, ksize = [1, k, k, 1], strides = [1, s, s, 1], padding = pad, name = name)

    def create_wb(self):
        # input and output
        self.xs = tf.placeholder("float", [None, self.n_inputs])
        self.ys = tf.placeholder("float", [None, self.n_classes])
        
        self.weights = {
            'wc1': tf.Variable(tf.truncated_normal([11, 11, 3, 96], dtype = tf.float32, stddev = 1e-1), trainable = True),
            'wc2': tf.Variable(tf.truncated_normal([5, 5, 96, 256], dtype = tf.float32, stddev = 1e-1), trainable = True),
            'wc3': tf.Variable(tf.truncated_normal([3, 3, 256, 384], dtype = tf.float32, stddev = 1e-1), trainable = True),
            'wc4': tf.Variable(tf.truncated_normal([3, 3, 384, 384], dtype = tf.float32, stddev = 1e-1), trainable = True),
            'wc5': tf.Variable(tf.truncated_normal([3, 3, 384, 256], dtype = tf.float32, stddev = 1e-1), trainable = True),
            'wd1': tf.Variable(tf.random_normal([6*6*256, 4096])),
            'wd2': tf.Variable(tf.random_normal([4096, 4096])),
            'out': tf.Variable(tf.random_normal([4096, self.n_classes]))
            }
        self.biases = {
            'bc1': tf.Variable(tf.constant(0.0, shape = [96], dtype = tf.float32), trainable = True),
            'bc2': tf.Variable(tf.constant(0.0, shape = [256], dtype = tf.float32), trainable = True),
            'bc3': tf.Variable(tf.constant(0.0, shape = [384], dtype = tf.float32), trainable = True),
            'bc4': tf.Variable(tf.constant(0.0, shape = [384], dtype = tf.float32), trainable = True),
            'bc5': tf.Variable(tf.constant(0.0, shape = [256], dtype = tf.float32), trainable = True),
            'bd1': tf.Variable(tf.random_normal([4096])),
            'bd2': tf.Variable(tf.random_normal([4096])),
            'out': tf.Variable(tf.random_normal([self.n_classes]))
            }
    # lrn
    def lrn(self,name, x, lsize = 4):
        return tf.nn.lrn(x, lsize, bias = 1.0, alpha = 0.001 / 9.0, beta = 0.75, name = name)
        
    # 显示网络每一层结构的函数,展示每一个卷积层或池化层输出tensor尺寸
    def print_activation(self,t):
        print(t.op.name," ",t.get_shape().as_list())
        
    def structure(self):
        # 向量转为矩阵
        self.x_ = tf.reshape(self.xs, shape=[-1, 227, 227, 3])
        ### 第一层
        # 卷积层
        self.conv1 = self.conv2d('conv1', self.x_, self.weights['wc1'], self.biases['bc1'],k = 4)
        #self.print_activation(self.conv1)
        # lrn
        self.lrn1 = self.lrn('lrn1', self.conv1, lsize = 4)
        # 下采样层
        self.pool1 = self.max_pool('pool1', self.lrn1, k = 3, s =2, pad = "VALID")
        #self.print_activation(self.pool1)
        
        ###第二层
        # 卷积
        self.conv2 = self.conv2d('conv2', self.pool1, self.weights['wc2'], self.biases['bc2'])
        #self.print_activation(self.conv2)
        # lrn
        self.lrn2 = self.lrn('lrn2', self.conv2, lsize = 4)
        # 下采样
        self.pool2 = self.max_pool('pool2', self.lrn2, k = 3, s = 2, pad = "VALID")
        #self.print_activation(self.conv1)
        
        ###第三层
        # 卷积
        self.conv3 = self.conv2d('conv3', self.pool2, self.weights['wc3'], self.biases['bc3'])
        #self.print_activation(self.conv3)
        
        ###第四层
        # 卷积
        self.conv4 = self.conv2d('conv4', self.conv3, self.weights['wc4'], self.biases['bc4'])
        #self.print_activation(self.conv4)
        
        ###第五层
        # 卷积
        self.conv5 = self.conv2d('conv5', self.conv4, self.weights['wc5'], self.biases['bc5'])
        #self.print_activation(self.conv5)
        
        # 下采样
        self.pool3 = self.max_pool('pool3', self.conv5, k = 3, s = 2, pad = "VALID")
        #self.print_activation(self.pool3)
        
        # 全连接层，先把特征图转为向量
        self.dense1 = tf.reshape(self.pool3, [-1, self.weights['wd1'].get_shape().as_list()[0]]) 
        self.dense1 = tf.nn.relu(tf.matmul(self.dense1, self.weights['wd1']) + self.biases['bd1'], name = 'fc1') 
        # 全连接层
        self.dense2 = tf.nn.relu(tf.matmul(self.dense1, self.weights['wd2']) + self.biases['bd2'], name = 'fc2') # Relu activation

        # 网络输出层
        self.pred = tf.matmul(self.dense2, self.weights['out']) + self.biases['out']
        #self.print_activation(self.pred)
        
    def compute_loss(self):
        self.loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.pred, labels = self.ys))
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
            for epoch in range(self.iterations):
                avg_loss = 0.
                total_batch = int(self.n / self.batch_size)
                # Loop over all batches
                for i in range(total_batch):
                    batch_x = xdata[(i * self.batch_size):((i + 1) * self.batch_size)]
                    batch_y = ydata[(i * self.batch_size):((i + 1) * self.batch_size)]
                    # Run optimization op (backprop) and cost op (to get loss value)
                    _,loss = sess.run([self.optimizer, self.loss_function],
                                      feed_dict = {self.xs: batch_x,self.ys: batch_y})
                    # Compute average loss
                    avg_loss += loss / total_batch
                all_loss.append(avg_loss)
                # Display logs per epoch step
                if epoch % self.display_epoch == 0:
                    print(time.strftime('%Y-%m-%d,%H:%M:%S',time.localtime()),"AlexNet-Epoch:", '%04d' % (epoch+1), "loss= ","%.9f" % avg_loss)
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
sess = tf.Session()
images = tf.Variable(tf.random_normal([32*3, #batch_size,
                                    227*227*3], #image_size,
                                    dtype = tf.float32,
                                    stddev = 1e-1))
init = tf.global_variables_initializer()
sess.run(init)
images = sess.run(images)
from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
images_labels = mnist.train.labels[:32*3]
al = AlexNetOriginal(lr = 1e-3,keep_prob = 0.8,iterations = 10,batch_size = 32,
          n_inputs = 227*227*3,n_classes = 10,n = 32*3,display_epoch = 1)
al.train(images,images_labels)
al.test(mnist.test.images,mnist.test.labels)
al.predict(images)
al.save(r'/home/max/Documents/LJQ/Code/rnn.model')

al1 = AlexNetOriginal(lr = 1e-3,keep_prob = 0.8,iterations = 1,batch_size = 32,
          n_inputs = 227*227*3,n_classes = 10,n = 32*3,display_epoch = 1)
al1.load('/home/max/Documents/LJQ/Code/rnn.model')
al1.test(mnist.test.images,mnist.test.labels)