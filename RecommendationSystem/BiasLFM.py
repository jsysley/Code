# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 14:50:29 2017

@author: Ivan
"""

import random
import pickle
import math
import os
#import decimal
#decimal.getcontext().prec = 100
# from types import MethodType

class BiasLFM(object):
    
    def __init__(self, K, alpha = 0.1, lmbd_user = 1, lmbd_item = 1, iterations = 500):
        '''
        param K : The number of latent factor
        param alpha : Learning rate
        param lmbda_user : Regulization of Pu
        param lmbda_item : Regulization of Qr
        param iterations : Maximum iterations 
        '''
        self.trainset = dict()
        self.testset = dict()
        self.alluser = set()
        self.allitem = set()

        self.K = K
        # R = P Q T
        self.P = dict()
        self.Q = dict()
        self.alpha = alpha
        self.lmbd_user = lmbd_user
        self.lmbd_item = lmbd_item
        self.iterations = iterations
        self.bu = dict() # 一个用户的平均评分
        self.bi = dict() # 第一个物品的平均评分
        self.mu = 0.0 # 全局平均评分
        random.seed(1)
        
    @staticmethod
    def loadfile(filename):
        # load a file, return a generator
        fp = open(filename, 'r')
        for i, line in enumerate(fp):
            yield line.strip('\r\n')
            if i % 100000 == 0:
                print 'loading %s(%s)' % (filename, i)
        fp.close()
        print 'load %s successfully' % filename

    def generate_dataset(self, filename, pivot = 0.7):
        '''
        load rating data and split it to training set and test set
        param filename : file path
        param pivot : the ratio of training set and test set
        '''
        
        trainset_len = 0
        testset_len = 0

        for line in self.loadfile(filename):
            user, movie, rating, _ = line.split('::')
            self.alluser.add(user)
            self.allitem.add(movie)
            # split the data by pivot
            if user in self.alluser and movie in self.allitem:
                if random.random() < pivot:
                    self.trainset.setdefault(user, {})
                    self.trainset[user][movie] = int(rating)
                    trainset_len += 1
                else:
                    self.testset.setdefault(user, {})
                    self.testset[user][movie] = int(rating)
                    testset_len += 1
            else:
                self.trainset.setdefault(user, {})
                self.trainset[user][movie] = int(rating)
                trainset_len += 1

        print 'split training set and test set successfully'
        print 'train set = %s' % trainset_len
        print 'test set = %s' % testset_len
    
    def initialize(self):
        '''
        Random initialize matrix P and Q
        rating_data format : dict(user : dict(movie:rating))
        matrix format : dict(user : list(scores))
        '''
        
        cnt = 0
        for user, watched in self.trainset.items():
            # initialize matrix P
            self.P[user] = [random.random() / math.sqrt(self.K) for x in xrange(self.K)]
            self.bu[user] = 0 # 初始化每个用户的平均评分
            cnt += len(watched) # 记录以所有的评分的数量
            # initialize matrix Q
            for item, rating in watched.items():
                self.mu += rating # 求所有评分的总和
                if item not in self.Q:
                    self.Q[item] = [random.random() / math.sqrt(self.K) for x in xrange(self.K)]
                self.bi[item] = 0 # 初始化每个物品的平均评分
        self.mu /= cnt
                
    def predict(self, user, item):
        # precict the score of user for item
        res = sum(self.P[user][k] * self.Q[item][k] for k in xrange(self.K))
        res += self.bu[user] + self.bi[item] + self.mu
        return res
    
    def train(self, online = False, path = None):
        # stochastic gradient descent
        if not online:
            self.initialize()
            print '>>> initialize matrix P & Q successfully'
        else:
            self.load(path)
            print '>>> load matrix P & Q successfully'
        for step in xrange(self.iterations):
            loss = 0
            length = len(self.trainset)
            for user, watched in self.trainset.items():
                for item, rui in watched.items():
#                    print step,'a',rui
                    hat_rui = self.predict(user, item)
                    err_ui = rui - hat_rui
                    loss += err_ui ** 2 / (1.0 * length)
#                    print step,'a',err_ui
                    self.bu[user] += self.alpha * (err_ui - self.lmbd_user * self.bu[user])
                    self.bi[item] += self.alpha * (err_ui - self.lmbd_item * self.bi[item])
                    
                    for k in xrange(self.K):
                        self.P[user][k] += self.alpha * (err_ui * self.Q[item][k] - self.lmbd_user * self.P[user][k])
                        self.Q[item][k] += self.alpha * (err_ui * self.P[user][k] - self.lmbd_item * self.Q[item][k])

            print 'iteraions: %04d, loss: %.4f' % (step, loss)
            #self.alpha *= 0.9 # 每次迭代次数缩减

    def save(self, path):
        allVar = [self.P, self.Q, self.bu, self.bi, self.mu]
        with open(path, 'w') as f:
             pickle.dump(allVar, f)
        print 'save the key variables successfully'

    def load(self, path):
        with open(path, 'r') as f:
            self.P, self.Q, self.bu, self.bi, self.mu = pickle.load(f)
        print 'load the key variables successfully'
    
    def evaluation(self):
        loss = 0.0
        length = len(self.testset)
        
        for i, user in enumerate(self.trainset):
            print '\r>>> check for %dth record' % i,
            test_items = self.testset.get(user,{})
            for item ,rating in test_items.items():
                rec_item = self.predict(user,item)
                loss += (rec_item - rating) ** 2 / (1.0 * length)
        print '\n>>> test loss %4f' % loss

if __name__ == '__main__':
    path = r'D:\file\RS\MovieLens-RecSys-python2'
    ratingfile = os.path.join(path,'ml-1m/ratings.dat')
    biaslfm = BiasLFM(K = 5)
    biaslfm.generate_dataset(ratingfile)
    biaslfm.train()
#    temp = {'A':{'a':1.0,'b':5.0},'B':{'b':2.0,'c':1.0},'C':{'c':3.0,'d':2.0}}
#    biaslfm.trainset = temp
#    biaslfm.train()
#    for item in ['a','b','c','d']:
#        print item,biaslfm.predict('A',item)