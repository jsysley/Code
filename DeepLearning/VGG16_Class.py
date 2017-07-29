# -*- coding: utf-8 -*-
"""
Created on Sat May 20 14:05:12 2017

@author: jsysley
"""

import time
import tensorflow as tf
import matplotlib.pyplot as plt
class VGG16:
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
        self.dropout = tf.placeholder(tf.float32)
        self.build()
    
    # conv
    # input_op 输入；name 层名；kh 卷积核高，kw 卷积核宽；n_out 卷积核数量；dh 步长的高;dw 步长的宽；p 参数列表
    def conv_op(self,input_op, name, kh, kw, n_out, dh, dw, p):
        n_in = input_op.get_shape()[-1].value # 获得通道数
        with tf.name_scope(name) as scope:
            # weights
            kernel = tf.get_variable(scope + "w", shape = [kh, kw, n_in, n_out], dtype = tf.float32, 
                                     initializer = tf.contrib.layers.xavier_initializer_conv2d())#Xavier初始化
            # 卷积核
            conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding = 'SAME')
            # 偏置项
            bias_init_val = tf.constant(0.0, shape = [n_out], dtype = tf.float32)
            biases = tf.Variable(bias_init_val, trainable = True, name = 'b')
            # 求和
            z = tf.nn.bias_add(conv, biases)
            # 激励函数
            activation = tf.nn.relu(z, name = scope)
            p += [kernel,biases]
            return activation
            
    # fc
    # input_op 输入；name 层名；n_out 卷积核数量；p 参数列表
    def fc_op(self,input_op, name, n_out, p):
        n_in = input_op.get_shape()[-1].value
        with tf.name_scope(name) as scope:
            # weights
            kernel = tf.get_variable(scope + "w", shape = [n_in, n_out], dtype = tf.float32, 
                                     initializer = tf.contrib.layers.xavier_initializer())
            # bias
            biases = tf.Variable(tf.constant(0.1, shape = [n_out], dtype = tf.float32), name = 'b')#避免死亡节点
            # 激励
            activation = tf.nn.relu_layer(input_op, kernel, biases, name = scope)
            p += [kernel, biases]
            return activation
            
    # pooling
    # input_op 输入；name 层名；kh 卷积核高，kw 卷积核宽；dh 步长的高;dw 步长的宽；p 参数列表
    def mpool_op(self,input_op, name, kh, kw, dh, dw):
        return tf.nn.max_pool(input_op, ksize = [1, kh, kw, 1], strides = [1, dh, dw, 1], 
                              padding = 'SAME', name = name)
                              
    def structure(self):
        self.p = []
        # input and output
        self.xs = tf.placeholder("float", [None, self.n_inputs])
        self.ys = tf.placeholder("float", [None, self.n_classes])
        # reshape
        x_ = tf.reshape(self.xs, [-1,224,224,3])
        
        # 第一段卷积层
        conv1_1 = self.conv_op(x_, name = "conv1_1", kh = 3, kw = 3, n_out = 64, dh = 1, dw = 1, p = self.p)
        conv1_2 = self.conv_op(conv1_1, name = "conv1_2", kh = 3, kw = 3, n_out = 64, dh = 1, dw = 1, p = self.p)
        pool1 = self.mpool_op(conv1_2, name = "pool1", kh = 2, kw = 2, dw = 2, dh = 2)
    
        #第二段卷积层
        conv2_1 = self.conv_op(pool1, name = "conv2_1", kh = 3, kw = 3, n_out = 128, dh = 1, dw = 1, p = self.p)
        conv2_2 = self.conv_op(conv2_1, name = "conv2_2", kh = 3, kw = 3, n_out = 128, dh = 1, dw = 1, p = self.p)
        pool2 = self.mpool_op(conv2_2, name = "pool2", kh = 2, kw = 2, dh = 2, dw = 2)
    
        #第三段卷积层
        conv3_1 = self.conv_op(pool2, name = "conv3_1", kh = 3, kw = 3, n_out = 256, dh = 1, dw =1, p = self.p)
        conv3_2 = self.conv_op(conv3_1, name = "conv3_2", kh = 3, kw = 3, n_out = 256, dh = 1, dw = 1, p = self.p)
        conv3_3 = self.conv_op(conv3_2, name = "conv3_3", kh = 3, kw = 3, n_out = 256, dh = 1, dw = 1, p = self.p)
        pool3 = self.mpool_op(conv3_3, name = "pool3", kh = 2, kw = 2, dh = 2, dw = 2)
    
        #第四段卷积层
        conv4_1 = self.conv_op(pool3, name = "conv4_1", kh = 3, kw = 3, n_out = 512, dh = 1, dw = 1, p = self.p)
        conv4_2 = self.conv_op(conv4_1, name = "conv4_2", kh = 3, kw = 3, n_out = 512, dh = 1, dw = 1, p = self.p)
        conv4_3 = self.conv_op(conv4_2, name = "conv4_3", kh = 3, kw = 3, n_out = 512, dh = 1, dw = 1, p = self.p)
        pool4 = self.mpool_op(conv4_3, name = "pool4", kh = 2, kw = 2, dh = 2, dw = 2)
    
        #第五段卷积层
        conv5_1 = self.conv_op(pool4, name = "conv5_1", kh = 3, kw = 3, n_out = 512, dh = 1, dw = 1, p = self.p)
        conv5_2 = self.conv_op(conv5_1, name = "conv5_2", kh = 3, kw = 3, n_out = 512, dh = 1, dw = 1, p = self.p)
        conv5_3 = self.conv_op(conv5_2, name = "conv5_3", kh = 3, kw = 3, n_out = 512, dh = 1, dw = 1, p = self.p)
        pool5 = self.mpool_op(conv5_3, name = "pool5", kh = 2, kw = 2, dw = 2, dh = 2)
        
        #将第五段卷积层输出结果抽成一维向量
        shp = pool5.get_shape()
        flattended_shape = shp[1].value * shp[2].value * shp[3].value
        resh1 = tf.reshape(pool5, [-1, flattended_shape], name = "resh1")

        #三个全连接层
        fc6 = self.fc_op(resh1, name = "fc6", n_out = 4096, p = self.p)
        fc6_drop = tf.nn.dropout(fc6, self.dropout, name = "fc6_drop")
    
        fc7 = self.fc_op(fc6_drop, name = "fc7", n_out = 4096, p = self.p)
        fc7_drop = tf.nn.dropout(fc7, self.dropout, name = "fc7_drop")
    
        self.pred = self.fc_op(fc7_drop, name = "fc8", n_out = self.n_classes, p = self.p)
        self.softmax = tf.nn.softmax(self.pred) # 使用SoftMax分类器输出概率最大的类别
        self.pred_label = tf.argmax(self.softmax, 1)
            
    def compute_loss(self):
        self.loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.ys))
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss_function)

    def build(self):
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
                    print(time.strftime('%Y-%m-%d,%H:%M:%S',time.localtime()),"VGG-Epoch:", '%04d' % (epoch+1), "cost= ","%.9f" % avg_loss)
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
            prob = sess.run(self.pred,feed_dict = {self.xs:tdata,self.dropout:1.0})
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
                                    224*224*3], #image_size,
                                    dtype = tf.float32,
                                    stddev = 1e-1))
init = tf.global_variables_initializer()
sess.run(init)
images = sess.run(images)
from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
images_labels = mnist.train.labels[:32*3]

vgg = VGG16(lr = 1e-4,keep_prob = 1,iterations = 1,batch_size = 32,
          n_inputs = 224*224*3,n_classes = 10,n = 32*3,display_epoch = 1)
vgg.train(images,images_labels)
vgg.predict(images)
vgg.test(images,images_labels)
vgg.save(r'F:/code/python3/DeepLearning/template/record/vgg.ckpt')

vgg1 = VGG16(lr = 1e-4,keep_prob = 1,iterations = 1,batch_size = 100,
          n_inputs = 784,n_classes = 10,n = 55000,display_epoch = 1)
vgg1.load(r'F:/code/python3/DeepLearning/template/record/vgg.ckpt')
vgg1.test(mnist.test.images,mnist.test.labels)

