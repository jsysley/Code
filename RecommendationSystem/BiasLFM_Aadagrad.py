# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 14:50:29 2017

@author: jsysley
"""
path = 'D:/file/SuperMarketResearch/project'
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import math
import os
os.chdir(os.path.join(path,'Code'))
from Processing import *

class BiasLFM(Base):
    
    def __init__(self, K, alpha = 0.1, lmbd_user = 1, lmbd_item = 1, iterations = 500):
        '''
        param K : The number of latent factor
        param alpha : Learning rate
        param lmbda_user : Regulization of Pu
        param lmbda_item : Regulization of Qr
        param iterations : Maximum iterations 
        '''
        Base.__init__(self)
        
        self.K = K
        # R = P Q T
        self.P = dict()
        self.Q = dict()
        self.alpha = alpha
        self.lmbd_user = lmbd_user
        self.lmbd_item = lmbd_item
        self.iterations = iterations
        self.bu = dict() 
        self.bi = dict() 
        self.mu = 0.0 
        random.seed(1)
    
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
            self.bu[user] = 0 
            cnt += len(watched) 
            # initialize matrix Q
            for item, rating in watched.items():
                self.mu += rating 
                if item not in self.Q:
                    self.Q[item] = [random.random() / math.sqrt(self.K) for x in xrange(self.K)]
                self.bi[item] = 0 
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
            print 'initialize matrix P & Q successfully'
        else:
            self.load(path)
            
        all_loss = []
        length = len(self.trainset)
        menbu = dict()
        menbi = dict()
        menp = dict()
        menq = dict()
        dbu = dict()
        dbi = dict()
        dp = dict()
        dq = dict()
        for step in xrange(self.iterations):
            loss = 0
            
            for user, watched in self.trainset.items():
                for item, rui in watched.items():
#                    print step,'a',rui
                    hat_rui = self.predict(user, item)
                    err_ui = rui - hat_rui
                    loss += err_ui ** 2 / (1.0 * length)
#                    print step,'a',err_ui
                    # Adagrad
                    dbu[user] = (err_ui - self.lmbd_user * self.bu[user])
                    dbi[item] = (err_ui - self.lmbd_item * self.bi[item])
                    menbu[user] = menbu.get(user,0) + dbu[user] ** 2
                    menbi[item] = menbi.get(item,0) + dbi[item] ** 2

                    self.bu[user] += self.alpha * dbu[user] / np.sqrt(menbu[user] + 1e-8)
                    self.bi[item] += self.alpha * dbi[item] / np.sqrt(menbi[item] + 1e-8)
                    
                    for k in xrange(self.K):
                        if user not in dp:
                            dp.setdefault(user,{})
                            menp.setdefault(user,{})
                        if item not in dq:
                            dq.setdefault(item,{})
                            menq.setdefault(item,{})
                        dp[user][k] = (err_ui * self.Q[item][k] - self.lmbd_user * self.P[user][k])
                        dq[item][k] = (err_ui * self.P[user][k] - self.lmbd_item * self.Q[item][k])
                        menp[user][k] = menp[user].get(k,0) + dp[user][k] ** 2
                        menq[item][k] = menq[item].get(k,0) + dq[item][k] ** 2
                        self.P[user][k] += self.alpha * dp[user][k] / np.sqrt(menp[user][k] + 1e-8)
                        self.Q[item][k] += self.alpha * dq[item][k] / np.sqrt(menq[item][k] + 1e-8)

            print 'BiasLFM iteraions: %04d, loss: %.4f' % (step, loss)
            all_loss.append(loss)
            #self.alpha *= 0.9 # 每次迭代次数缩减
        plt.figure()
        plt.plot(list(range(len(all_loss))), all_loss, color = 'b')
        plt.show()

    def save(self, path):
        allVar = [self.P, self.Q, self.bu, self.bi, self.mu]
        with open(path, 'w') as f:
             pickle.dump(allVar, f)
        print 'save the key nodes successfully'

    def load(self, path):
        with open(path, 'r') as f:
            self.P, self.Q, self.bu, self.bi, self.mu = pickle.load(f)
        print 'load the key nodes successfully'
    
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
    biaslfm = BiasLFM(K = 2)
    temp = {'A':{'a':1.0,'b':5.0},'B':{'b':2.0,'c':1.0},'C':{'c':3.0,'d':2.0}}
    biaslfm.trainset = temp
    biaslfm.train()
    for item in ['a','b','c','d']:
        print item,biaslfm.predict('A',item)