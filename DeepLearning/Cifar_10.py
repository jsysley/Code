# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 15:09:38 2017

@author: jsysley
"""

import os
os.chdir(u'F:/code/python3/官方文档/models-master/models-master/tutorials/image/cifar10')
import cifar10,cifar10_input
import tensorflow as tf
import numpy as np
import time

max_steps = 3000 #迭代次数
batch_size = 128 # batch size
# 下载数据集路径
data_dir = '/tmp/cifar10_data/cifar-10-batches-bin'

# 使用wl控制L2loss的大小，tf.nn.l2.loss函数计算weight的L2 loss，
# 再使用tf.multiply让L2 loss乘以wl，得到最后的weight loss。
# 使用tf.add_to_collection把weight loss统一存到一个名为"losses"的collection上
def variable_with_weight_loss(shape, stddev, wl):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var

# 使用cifar10类下载数据集，并解压、展开到其默认位置
cifar10.maybe_download_and_extract()

# 产生训练数据，返回的是封装好的tensor，每次执行都会产生一个batch_size的数量样本
images_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir,batch_size=batch_size)
images_test, labels_test = cifar10_input.inputs(eval_data=True,
                                                data_dir=data_dir,
                                                batch_size=batch_size)     
# 创建输入数据的placeholder 
image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
label_holder = tf.placeholder(tf.int32, [batch_size])

# 第一层
# 注意这里最大池化的尺寸和步长不一致，这样可以增加数据的丰富性
# tf.nn.lrn:即LRN对结果进行处理，LRN模仿了生物系统的“侧抑制”机制，对局部神经元的活动
# 创建了竞争环境，使得其中响应比较大的值变得相对更大，并抑制其他反馈较小的神经元，增强模型泛化能力
weight1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, wl=0.0)
kernel1 = tf.nn.conv2d(image_holder, weight1, [1, 1, 1, 1], padding='SAME')
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],padding='SAME')
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

# 第二层 
# bias初始化为0.1而不是0
# 调换了最大池化层和LRN层的顺序，先LRN层处理再进行最大池化层
weight2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, wl=0.0)
kernel2 = tf.nn.conv2d(norm1, weight2, [1, 1, 1, 1], padding='SAME')
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],padding='SAME')

# 第三层-全连接层
# get_shape函数获取数据扁平化后的长度
reshape = tf.reshape(pool2, [batch_size, -1])
dim = reshape.get_shape()[1].value
weight3 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, wl=0.004)
bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)

# 第四层-全连接层
weight4 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, wl=0.004)
bias4 = tf.Variable(tf.constant(0.1, shape=[192]))                                      
local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4)

# 第五层-输出层
weight5 = variable_with_weight_loss(shape=[192, 10], stddev=1/192.0, wl=0.0)
bias5 = tf.Variable(tf.constant(0.0, shape=[10]))
logits = tf.add(tf.matmul(local4, weight5), bias5)


def loss(logits, labels):
#      """Add L2Loss to all the trainable variables.
#      Add summary for "Loss" and "Loss/avg".
#      Args:
#        logits: Logits from inference().
#        labels: Labels from distorted_inputs or inputs(). 1-D tensor
#                of shape [batch_size]
#      Returns:
#        Loss tensor of type float.
#      """
#      # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

loss = loss(logits, label_holder)
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss) #0.72

# 使用tf.nn.in_top_k函数求出输出结果中top k的准确率，默认使用top1，即分数最高的那一类概率
top_k_op = tf.nn.in_top_k(logits, label_holder, 1)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# 启动图片数据增强的线程队列，一共使用16个线程来进行加速
tf.train.start_queue_runners()

# 训练
for step in range(max_steps):
    start_time = time.time()
    image_batch,label_batch = sess.run([images_train,labels_train])
    _, loss_value = sess.run([train_op, loss],feed_dict={image_holder: image_batch, 
                                                         label_holder:label_batch})
    duration = time.time() - start_time

    if step % 10 == 0:
        examples_per_sec = batch_size / duration
        sec_per_batch = float(duration)
    
        format_str = ('step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
        print(format_str % (step, loss_value, examples_per_sec, sec_per_batch))
        
### 测试集
num_examples = 10000
import math
num_iter = int(math.ceil(num_examples / batch_size))
true_count = 0  
total_sample_count = num_iter * batch_size
step = 0
while step < num_iter:
    image_batch,label_batch = sess.run([images_test,labels_test])
    predictions = sess.run([top_k_op],feed_dict={image_holder: image_batch,
                                                 label_holder:label_batch})
    true_count += np.sum(predictions)
    step += 1

precision = true_count / total_sample_count
print('precision @ 1 = %.3f' % precision)