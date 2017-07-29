# -*- coding: utf-8 -*-
"""
Created on Wed May 17 16:21:26 2017

@author: jsysley
"""
import tensorflow as tf
import matplotlib.pyplot as plt
class BiRNN:
    def __init__(self,lr,iterations,batch_size,n_inputs,n_hiddens,n_steps,
                 n_classes,n,display_epoch = 1,keep_prob = 1):
        # Hyperparameters
        self.lr = lr
        self.keep_prob = keep_prob
        self.iterations = iterations
        self.batch_size = batch_size
        self.n_inputs = n_inputs
        self.n_hiddens = n_hiddens
        self.n_steps = n_steps
        self.n_classes = n_classes
        self.n = n
        self.display_epoch = display_epoch
        self.build()
    
    def create_wb(self):
        # input and output
        self.xs = tf.placeholder("float", [None, self.n_steps, self.n_inputs])
        self.ys = tf.placeholder("float", [None, self.n_classes])
        
        self.weights = {
            'in': 
                tf.Variable(tf.random_normal([self.n_inputs, self.n_hiddens],mean = 0.,stddev =  1.)),
            # Hidden layer weights => 2*n_hidden because of forward + backward cells
            'out': tf.Variable(tf.random_normal([2 * self.n_hiddens, self.n_classes]))
            }
            
        self.biases = {
            'in': tf.Variable(tf.random_normal([self.n_hiddens])),
            'out': tf.Variable(tf.random_normal([self.n_classes]))
            }
            
    def structure(self):
        # intput
        l_in_x = tf.reshape(self.xs, [-1, self.n_inputs])
        # ?x10
        l_in_y = tf.matmul(l_in_x, self.weights['in']) + self.biases['in']
        # reshape l_in_y ==> (batch, n_steps, cell_size)
        l_in_y = tf.reshape(l_in_y, [-1, self.n_steps, self.n_hiddens])
        
        # Prepare data shape to match `bidirectional_rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_hiddens)
        
        # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_hiddens)
        x_ = tf.unstack(self.xs, self.n_steps, 1)

        # Define lstm cells with tensorflow
        # Forward direction cell
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hiddens, forget_bias=1.0)
        
        # Backward direction cell
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hiddens, forget_bias=1.0)

        # Get lstm cell output
        try:
            #outputs: number of n_steps tensor ,each of then is 2*n_hidden
            outputs, no1,no2 = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x_,
                                                  dtype = tf.float32)
        except Exception: # Old TensorFlow version only returns outputs not states
            outputs = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x_,
                                            dtype = tf.float32)

        # Linear activation, using rnn inner loop last output
        self.pred = tf.matmul(outputs[-1], self.weights['out']) + self.biases['out']
        

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
                    batch_x = batch_x.reshape([self.batch_size, self.n_steps,self. n_inputs])
                    batch_y = ydata[(i * self.batch_size):((i + 1) * self.batch_size)]
                    # Run optimization op (backprop) and cost op (to get loss value)
                    _,loss = sess.run([self.optimizer, self.loss_function],
                                      feed_dict={self.xs: batch_x,self.ys: batch_y})
                    # Compute average loss
                    avg_loss += loss / total_batch
                all_loss.append(avg_loss)
                # Display logs per epoch step
                if epoch % self.display_epoch == 0:
                    print("RNN-Epoch:", '%04d' % (epoch+1), "cost= ","%.9f" % avg_loss)
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
            xdata = xdata.reshape([1, self.n_steps,self. n_inputs])
            print("Accuracy:", accuracy.eval({self.xs: xdata, self.ys: ydata}))
        
    def predict(self,tdata):
        with self.sess as sess:
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
birnn = BiRNN(lr = 1e-3,keep_prob = 1,iterations = 1,batch_size = 128,n_steps = 28,
          n_inputs = 28,n_hiddens = 128,n_classes = 10,n = 55000,display_epoch = 1)
birnn.train(mnist.train.images,mnist.train.labels)
birnn.save(r'F:/code/python3/DeepLearning/template/record/model.ckpt')
birnn.test(mnist.test.images,mnist.test.labels)
birnn.save(r'F:/code/python3/DeepLearning/template/record/model.ckpt')

rnn1 = BiRNN(lr = 1e-3,keep_prob = 1,iterations = 1,batch_size = 128,n_steps = 28,
          n_inputs = 28,n_hiddens = 128,n_classes = 10,n = 55000,display_epoch = 1)
rnn1.load(r'F:/code/python3/DeepLearning/template/record/model.ckpt')
rnn1.test(mnist.test.images,mnist.test.labels)
