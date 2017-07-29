# -*- coding: utf-8 -*-
"""
Created on Sun May 21 12:27:01 2017

@author: jsysley
"""

import time
import tensorflow as tf
from datetime import datetime
import math
class GoogleV3:
    def __init__(self,batch_size,height,width,depth):
        # Hyperparameters
        #self.lr = lr
        #self.iterations = iterations
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.depth = depth
        #self.n_classes = n_classes
        #self.n = n
        #self.display_epoch = display_epoch
        #self.build()

    #定义函数inception_v3_arg_scope,用来生成网络中常用到的函数的默认参数
    def incepton_v3_arg_scope(self,weight_decay = 0.00004, stddev = 0.1, batch_norm_var_collection = 'moving_vars'):
        batch_norm_params = {
                'decay': 0.9997,
                'epsilon': 0.001,
                'updates_collections': tf.GraphKeys.UPDATE_OPS,
                'variables_collections': {
                        'beta': None,
                        'gamma': None,
                        'moving_mean': [batch_norm_var_collection],
                        'moving_variance': [batch_norm_var_collection],
                        }
                }
                
        # 给函数的参数自动赋予某些默认值
        # 对tf.contrib.slim.conv2d，tf.contrib.slim.fully_connected两个函数的参数自动赋值，将weights_regularizer的默认值设为tf.contrib.slim.l2_regularizer(weight_decay)
        with tf.contrib.slim.arg_scope([tf.contrib.slim.conv2d, tf.contrib.slim.fully_connected], weights_regularizer = tf.contrib.slim.l2_regularizer(weight_decay)):
            with tf.contrib.slim.arg_scope([tf.contrib.slim.conv2d],
                                weights_initializer = tf.truncated_normal_initializer(stddev = stddev),
                                activation_fn = tf.nn.relu,
                                normalizer_fn = tf.contrib.slim.batch_norm,
                                normalizer_params = batch_norm_params) as sc:
                return sc
    
    # 定义inception_v3_base,生成Inception V3网络的卷积部分
    # inputs为输入图片的tensor，scope为包含了函数默认参数的环境
    def inception_v3_base(self, inputs, scope = None):
        slim = tf.contrib.slim
        end_points = {}#用来保存某些关键点
        
        with tf.variable_scope(scope, 'InceptionV3', [inputs]):
            with tf.contrib.slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                    stride = 1, padding = 'VALID'):
                # 第一个参数为tensor，第二个参数为输出通道数，第三个参数时卷积核尺寸
                net = slim.conv2d(inputs, 32, [3, 3], stride = 2, scope = 'Conv2d_1a_3x3')
                net = slim.conv2d(net, 32, [3, 3], scope = 'Conv2d_2a_3x3')
                net = slim.conv2d(net, 64, [3, 3], padding = 'SAME', scope = 'Conv2d_2b_3x3')
                net = slim.max_pool2d(net, [3, 3], stride = 2, scope = 'MaxPool_3a_3x3')
                net = slim.conv2d(net, 80, [1, 1], scope = 'Conv2d_3b_1x1')
                net = slim.conv2d(net, 192, [3, 3], scope = 'Conv2d_4a_3x3')
                net = slim.max_pool2d(net, [3, 3], stride = 2, scope = 'MaxPool_5a_3x3')       
        
        #接下来就是三个连续的Inception模块组，这个按照实际网络写参数就可以了
        #第1个Inception模块组的第一个module
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride = 1, padding = 'SAME'):
            with tf.variable_scope('Mixed_5b'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 64, [1, 1], scope = 'Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 48, [1, 1], scope = 'Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope = 'Conv2d_0b_5x5')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 64, [1, 1], scope = 'Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope = 'Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope = 'Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope = 'AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 32, [1, 1], scope = 'Conv2d_0b_1x1')
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
    
            #第二个module      
            with tf.variable_scope('Mixed_5c'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 64, [1, 1], scope = 'Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 48, [1, 1], scope = 'Conv2d_0b_1x1')
                    branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope = 'Conv2d_0c_5x5')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 64, [1, 1], scope = 'Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope = 'Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope = 'Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope = 'AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope = 'Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2,branch_3], 3)
            
            #第三个module
            with tf.variable_scope('Mixed_5d'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 64, [1, 1], scope = 'Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 48, [1, 1], scope = 'Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope = 'Conv2d_0b_5x5')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 64, [1, 1], scope = 'Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope = 'Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope = 'Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope = 'AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope = 'Conv2d_0b_1x1')
    
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)#把每个分支的输出以第三个维度合并，即通道数合并

            #第2个Inception的第一个module
            with tf.variable_scope('Mixed_6a'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 384, [3, 3], stride = 2, padding = 'VALID', scope = 'Conv2d_1a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 64, [1, 1], scope = 'Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 96, [3, 3], scope = 'Conv2d_0b_3x3')
                    branch_1 = slim.conv2d(branch_1, 96, [3, 3], stride = 2, padding = 'VALID', scope = 'Conv2d_1a_1x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(net, [3, 3], stride = 2, padding = 'VALID', scope = 'Conv2d_1a_3x3')
                net = tf.concat([branch_0, branch_1, branch_2], 3)
            
            #第二个module
            with tf.variable_scope('Mixed_6b'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope = 'Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 128, [1, 1], scope = 'Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 128, [1, 7], scope = 'Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope = 'Conv2d_0c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 128, [1, 1], scope = 'Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 128, [7, 1], scope = 'Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, 128, [1, 7], scope = 'Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, 128, [7, 1], scope = 'Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope = 'Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.max_pool2d(net, [3, 3], scope = 'AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope = 'Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
                
            #第三个module
            with tf.variable_scope('Mixed_6c'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope = 'Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 160, [1, 1], scope = 'Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope = 'Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope = 'Conv2d_0c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 160, [1, 1], scope = 'Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope = 'Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope = 'Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope = 'Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope = 'Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.max_pool2d(net, [3, 3], scope = 'AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope = 'Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)    
                
            #第四个module
            with tf.variable_scope('Mixed_6d'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope = 'Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 160, [1, 1], scope = 'Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope = 'Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope = 'Conv2d_0c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 160, [1, 1], scope = 'Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope = 'Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope = 'Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope = 'Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope = 'Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.max_pool2d(net, [3, 3], scope = 'AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope = 'Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
            
            
            #第五个module
            with tf.variable_scope('Mixed_6e'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope = 'Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 160, [1, 1], scope = 'Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope = 'Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope = 'Conv2d_0c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 160, [1, 1], scope = 'Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope = 'Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope = 'Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope = 'Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope = 'Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.max_pool2d(net, [3, 3], scope = 'AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope = 'Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
    
            end_points['Mixed_6e'] = net#把Mixed_6e作为保存的输出，作为辅助分类节点
    
            #第3个Inception的第一个module
            with tf.variable_scope('Mixed_7a'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope = 'Conv2d_0a_1x1')
                    branch_0 = slim.conv2d(branch_0, 320, [3, 3], stride = 2, padding = 'VALID', scope = 'Conv2d_0b_3x3')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 192, [1, 1], scope = 'Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 192, [1, 7], scope = 'Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope = 'Conv2d_0c_7x1')
                    branch_1 = slim.conv2d(branch_1, 192, [3, 3], stride = 2, padding = 'VALID', scope = 'Conv2d_0e_3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(net, [3, 3], stride = 2, padding = 'VALID', scope = 'MaxPool_1a_3x3')
                net = tf.concat([branch_0, branch_1, branch_2], 3)
                
            #第二个module
            with tf.variable_scope('Mixed_7b'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 320, [1, 1], scope = 'Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 384, [1, 1], scope = 'Conv2d_0a_1x1')
                    branch_1 = tf.concat([
                            slim.conv2d(branch_1, 384, [1, 3], scope = 'Conv2d_0b_1x3'),
                            slim.conv2d(branch_1, 384, [3, 1], scope = 'Conv2d_0c_3x1')], 3)
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 448, [1, 1], scope = 'Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 384, [3, 3], scope = 'Conv2d_0a_3x3')
                    branch_2 = tf.concat([
                            slim.conv2d(branch_2, 384, [1, 3], scope = 'Conv2d_0c_1x3'),
                            slim.conv2d(branch_2, 384, [3, 1], scope = 'Conv2d_0e_3x1')], 3)
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.max_pool2d(net, [3, 3], scope = 'MaxPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope = 'Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
            
            #第三个module
            with tf.variable_scope('Mixed_7c'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 320, [1, 1], scope = 'Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 384, [1, 1], scope = 'Conv2d_0a_1x1')
                    branch_1 = tf.concat([
                            slim.conv2d(branch_1, 384, [1, 3], scope = 'Conv2d_0b_1x3'),
                            slim.conv2d(branch_1, 384, [3, 1], scope = 'Conv2d_0c_3x1')], 3)
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 448, [1, 1], scope = 'Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 384, [3, 3], scope = 'Conv2d_0a_3x3')
                    branch_2 = tf.concat([
                            slim.conv2d(branch_2, 384, [1, 3], scope = 'Conv2d_0c_1x3'),
                            slim.conv2d(branch_2, 384, [3, 1], scope = 'Conv2d_0e_3x1')], 3)
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.max_pool2d(net, [3, 3], scope = 'MaxPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope = 'Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
    
            return net, end_points#返回计算出的结果和作为辅助分类的节点的结果
    
    
    # 定义Inception V3网络的全局平均池化，Softmax和Auxiliary Logits
    # num_classes:最后需要分类的数量；
    # is_training:是否是训练过程，对batch normalization和dropout有影响（训练时才启用）
    # dropout_keep_prob:即训练时Dropout所需保留节点的比例
    # prediction_fn是最后用来分类的函数，默认使用slim.softmax
    # spatial_squeeze参数标志是否对输出进行squeeze操作（即去除维数为1的维度，如5x3x1转为5x3）
    # reuse表示是否会对网络和Variable进行重复使用
    # scope包含了函数默认参数的环境
    def inception_v3(self,inputs, num_classes = 1000, is_training = True, dropout_keep_prob = 0.8,
                    prediction_fn = tf.contrib.slim.softmax, spatial_squeeze = True, reuse = None, scope = 'InceptionV3'):
        slim = tf.contrib.slim
        trunc_normal = lambda stddev: tf.truncated_normal_initializer(0, 0, stddev)
        
        with tf.variable_scope(scope, 'InceptionV3', [inputs, num_classes], reuse = reuse) as scope:
            with slim.arg_scope([slim.batch_norm, slim.dropout], is_training = is_training):
                net, end_points = self.inception_v3_base(inputs, scope = scope)
                
                # Auxiliary Logits作为辅助分类的节点
                with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride = 1, padding = 'SAME'):
                    aux_logits = end_points['Mixed_6e']
                    with tf.variable_scope('AuxLogits'):
                        aux_logits = slim.avg_pool2d(aux_logits, [5, 5], stride = 3, padding = 'VALID', scope = 'AvgPool_1a_5x5')
                        aux_logits = slim.conv2d(aux_logits, 128, [1, 1], scope = 'Conv2d_1b_1x1')
                        aux_logits = slim.conv2d(aux_logits, 768, [5, 5], weights_initializer = trunc_normal(0.01),
                                                padding = 'VALID', scope = 'Conv2d_2a_5x5')
                        aux_logits = slim.conv2d(aux_logits, num_classes, [1, 1], activation_fn = None,
                                                normalizer_fn = None, weights_initializer = trunc_normal(0.001), scope = 'Conv2d_2b_1x1')
                        if spatial_squeeze:
                            aux_logits = tf.squeeze(aux_logits, [1, 2], name = 'SpatialSqueeze')
    
                        end_points['AuxLogits'] = aux_logits
                
                # 正常得分分类预测逻辑
                with tf.variable_scope('Logits'):
                    net = slim.avg_pool2d(net, [8, 8], padding = 'VALID', scope = 'AvgPool_1a_8x8')
                    net = slim.dropout(net, keep_prob = dropout_keep_prob, scope = 'Dropout_1b')
                    end_points['PreLogits'] = net
    
                    logits = slim.conv2d(net, num_classes, [1, 1], activation_fn = None, normalizer_fn = None, scope = 'Conv2d_1c_1x1')
    
                    if spatial_squeeze:
                        logits = tf.squeeze(logits, [1, 2], name = 'SpatialSqueeze')
                    end_points['Predictions'] = prediction_fn(logits, scope = 'Predictions')
            return logits, end_points

    def time_tensorflow_run(self,session, target, info_string,num_batches):
        num_steps_burn_in = 10
        total_duration = 0.0
        total_duration_squared = 0.0

        for i in range(num_batches + num_steps_burn_in):
            start_time = time.time()
            _ = session.run(target)
            duration = time.time() - start_time
            if i >= num_steps_burn_in:
                if not i % 10:
                    print('%s: step %d, duration = %.3f' %(datetime.now(), i - num_steps_burn_in, duration))
                total_duration += duration
                total_duration_squared += duration * duration

        mn = total_duration / num_batches
        vr = total_duration_squared / num_batches - mn * mn
        sd = math.sqrt(vr)
        print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %(datetime.now(), info_string, num_batches, mn, sd))
    
    def Try(self):
        inputs = tf.random_uniform((self.batch_size, self.height, self.width, 3))
        with tf.contrib.slim.arg_scope(self.incepton_v3_arg_scope()):
            logits, end_points = self.inception_v3(inputs, is_training = False)
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        self.time_tensorflow_run(sess, logits, "Forward",num_batches = 100)
    
            
    
            
#==============================================================================
#==============================================================================
googlev3 = GoogleV3(batch_size = 32,height = 299,width = 299,depth = 3)
googlev3.Try()
