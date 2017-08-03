# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 15:16:04 2017

@author: Ivan
"""

import random
import pickle
import math
import os

#import decimal
#decimal.getcontext().prec = 100
# from types import MethodType

class SVDPP(object):
    
    def __init__(self, K, alpha = 0.1, lmbd_user = 1, lmbd_item = 1, iterations = 500):
        '''
        param K : The number of latent factor
        param alpha : Learning rate
        param lmbda_user : Regulization of Pu
        param lmbda_item : Regulization of Qr
        param iterations : Maximum iterations 
        '''
        self.trainset = {}
        self.testset = {}
        
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
        self.Y = dict()
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
            # split the data by pivot
            if random.random() < pivot:
                self.trainset.setdefault(user, {})
                self.trainset[user][movie] = int(rating)
                trainset_len += 1
            else:
                self.testset.setdefault(user, {})
                self.testset[user][movie] = int(rating)
                testset_len += 1

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
                if item not in self.Y:
                    self.Y[item] = [random.random() / math.sqrt(self.K) for x in xrange(self.K)]
                self.bi[item] = 0 # 初始化每个物品的平均评分
        self.mu /= cnt
                
    def predict(self, user, item, watched):
        # precict the score of user for item
        z = [0.0 for k in xrange(self.K)]
        for ri, _ in watched.items():
            for k in xrange(self.K):
                z[k] += self.Y[ri][k]
        return sum((self.P[user][k] + z[k] / math.sqrt(1.0 * len(watched))) * self.Q[item][k] for k in xrange(self.K)) + self.bu[user] + self.bi[item] + self.mu
        
    
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
            for user, watched in self.trainset.items():
                z = [0.0 for k in xrange(self.K)]
                for item, _ in watched.items():
                    for k in xrange(self.K):
                        z[k] += self.Y[item][k]
                ru = 1.0 / math.sqrt(1.0 * len(watched))
                s = [0.0 for k in xrange(self.K)]
                for item, rui in watched.items():
#                    print step,'a',rui
                    hat_rui = self.predict(user, item, watched)
                    err_ui = rui - hat_rui
                    loss += abs(err_ui)
#                    print step,'a',err_ui
                    self.bu[user] += self.alpha * (err_ui - self.lmbd_user * self.bu[user])
                    self.bi[item] += self.alpha * (err_ui - self.lmbd_item * self.bi[item])
                    for k in xrange(self.K):
                        s[k] += self.Q[item][k] * err_ui # * ru 
                        self.P[user][k] += self.alpha * (err_ui * self.Q[item][k] - self.lmbd_user * self.P[user][k])
                        self.Q[item][k] += self.alpha * (err_ui * (self.P[user][k] + z[k] * ru) - self.lmbd_item * self.Q[item][k])
                for item, _ in watched.items():
                    for k in xrange(self.K):
                        self.Y[item][k] += self.alpha * (s[k] * ru - self.lmbd_item * self.Y[item][k])
            print 'iteraions: %04d, loss: %.4f' % (step, loss)
#            self.alpha *= 0.9 # 每次迭代次数缩减
    
    def save(self, path):
        allVar = [self.P, self.Q, self.bu, self.bi, self.mu, self.Y]
        with open(path, 'w') as f:
             pickle.dump(allVar, f)
        print 'save the key variables successfully'

    def load(self, path):
        with open(path, 'r') as f:
            self.P, self.Q, self.bu, self.bi, self.mu, self.Y = pickle.load(f)
        print 'load the key variables successfully'
        
if __name__ == '__main__':
    path = r'D:\file\RS\MovieLens-RecSys-python2'
    ratingfile = os.path.join(path,'ml-1m/ratings.dat')
    svdpp = SVDPP(K = 5)
    svdpp.generate_dataset(ratingfile)
    svdpp.train()
#    temp = {'A':{'a':1.0,'b':5.0},'B':{'b':2.0,'c':1.0},'C':{'c':3.0,'d':2.0}}
#    svdpp.trainset = temp
#    svdpp.train()
#    for item in ['a','b','c','d']:
#        print item,svdpp.predict('A',item, svdpp.trainset['A'])