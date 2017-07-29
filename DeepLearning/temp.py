# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


per = MulPerceptron(lr = 0.01,keep_prob = 1,iterations = 20,batch_size = 100,
          n_hiddens=[256,128,64],n_classes = 10,n_inputs = 784,n = 55000,display_epoch = 1)


path = r'/home/max/Documents/LJQ/Code/record/model.ckpt'

import tensorflow as tf
w = {'v1':tf.Variable(tf.random_normal([1,2]), name="v1"),
     'v2':tf.Variable(tf.random_normal([2,3]), name="v2")
        }
#v1 = tf.Variable(tf.random_normal([1,2]), name="v1")
#v2 = tf.Variable(tf.random_normal([2,3]), name="v2")
init_op = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init_op)
    print (sess.run(w['v1']))
    print (sess.run(w['v2']))
    saver_path = saver.save(sess, path)
    print ("Model saved in file: ", saver_path)
    
    
import tensorflow as tf
w = {'v1':tf.Variable(tf.random_normal([1,2]), name="v1"),
     'v2':tf.Variable(tf.random_normal([2,3]), name="v2")
        }
#v1 = tf.Variable(tf.random_normal([1,2]), name="v1")
#v2 = tf.Variable(tf.random_normal([2,3]), name="v2")
#v3 = tf.Variable(tf.random_normal([2,2]), name="v3")

saver = tf.train.Saver({'v1':w['v1']})
s = tf.Session()
with s.as_default() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, path)
    print (sess.run(w['v1']))
    #print sess.run(w['v2'])
    #print sess.run(v3)
    print ("Model restored")

