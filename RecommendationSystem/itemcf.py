#-*- coding: utf-8 -*-
'''
Created on 2017-07-21

@author: Ivan
'''

import sys
import random
import math
import os
from operator import itemgetter

random.seed(0)

class ItemBasedCF(object):
    # TopN recommendation - Item Based Collaborative Filtering 

    def __init__(self):
        self.trainset = {}
        self.testset = {}

        self.n_sim_movie = 20 # 取最相似的k个物品
        self.n_rec_movie = 10 # 推荐得分最高的k个物品

        self.movie_sim_mat = {} # 物品相似度矩阵
        self.movie_popular = {} # 每个物品被产生过行为的用户数
        self.movie_count = 0 # 记录一共有几个物品

        print 'Similar movie number = %d' % self.n_sim_movie
        print 'Recommended movie number = %d' % self.n_rec_movie
    
    # load a file, return a generator
    @staticmethod
    def loadfile(filename):
        fp = open(filename, 'r')
        for i, line in enumerate(fp):
            yield line.strip('\r\n')
            if i % 100000 == 0:
                print 'loading %s(%s)' % (filename, i)
        fp.close()
        print 'load %s successfully' % filename
    
    # load rating data and split it to training set and test set
    def generate_dataset(self, filename, pivot = 0.7):
        
        trainset_len = 0
        testset_len = 0
        # 建立用户-物品倒排表
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
    
    # calculate movie similarity matrix 
    def calc_movie_sim(self):
        
        print 'counting movies number and popularity...'

        for user, movies in self.trainset.iteritems(): # 对每个用户的记录进行遍历
            for movie in movies: # 对每个用户的物品进行遍历
                # count item popularity
                if movie not in self.movie_popular:
                    self.movie_popular[movie] = 0
                self.movie_popular[movie] += 1 # 记录每个物品被多少个用户产生过行为

        print 'count movies number and popularity successfully'

        # save the total number of movies
        self.movie_count = len(self.movie_popular)
        print 'total movie number = %d' % self.movie_count

        # count co-rated users between items
        itemsim_mat = self.movie_sim_mat
        print 'building co-rated users matrix...'

        for user, movies in self.trainset.iteritems(): # 物品的相似度矩阵
            for m1 in movies:
                for m2 in movies:
                    if m1 == m2:
                        continue
                    itemsim_mat.setdefault(m1, {})
                    itemsim_mat[m1].setdefault(m2, 0)
                    itemsim_mat[m1][m2] += 1 #惩罚过于活跃的用户 1 / math.log(1 + len(movies) * 1.0) #时间信息 1 / (1 + alpha * abs(tui - tuj))

        print 'build co-rated users matrix successfully'

        # calculate similarity matrix
        print 'calculating movie similarity matrix...'
        simfactor_count = 0
        PRINT_STEP = 2000000
        # 物品相似度矩阵
        for m1, related_movies in itemsim_mat.iteritems():
            for m2, count in related_movies.iteritems():
                itemsim_mat[m1][m2] = count / math.sqrt(
                    self.movie_popular[m1] * self.movie_popular[m2])
                simfactor_count += 1
                if simfactor_count % PRINT_STEP == 0:
                    print 'calculating movie similarity factor(%d)' % simfactor_count

        print 'calculate movie similarity matrix(similarity factor) successfully'
        print 'Total similarity factor number = %d' % simfactor_count
    
    # Find K similar movies and recommend N movies.
    def recommend(self, user):
        
        K = self.n_sim_movie
        N = self.n_rec_movie
        rank = {}
        watched_movies = self.trainset[user] # 要推荐的1个用户，用户-物品表

        for movie, rating in watched_movies.iteritems(): # 对用户看过的物品遍历
            for related_movie, similarity_factor in sorted(self.movie_sim_mat[movie].items(),
                                                           key = itemgetter(1), reverse = True)[0:K]:
                if related_movie in watched_movies: # 只推荐用户没有行为的物品
                    continue
                rank.setdefault(related_movie, 0)
                rank[related_movie] += similarity_factor * rating # 时间信息:similarity_factor * rating / (1 + alpha * (t0 - tui)) # 惩罚热门物品(提高新颖度)：pui / math.log(1 + alpha * popularity(i))
        # return the N best movies
        return sorted(rank.items(), key = itemgetter(1), reverse = True)[:N]
    
    # print evaluation result: precision, recall, coverage and popularity
    def evaluate(self):
        
        print 'Evaluation start...'

        N = self.n_rec_movie
        #  varables for precision and recall
        hit = 0
        rec_count = 0
        test_count = 0
        # varables for coverage
        all_rec_movies = set()
        # varables for popularity
        popular_sum = 0

        for i, user in enumerate(self.trainset):
            if i % 500 == 0:
                print 'recommended for %d users' % i
            test_movies = self.testset.get(user, {})
            rec_movies = self.recommend(user)
            for movie, _ in rec_movies:
                if movie in test_movies:
                    hit += 1
                all_rec_movies.add(movie)
                popular_sum += math.log(1 + self.movie_popular[movie])
            rec_count += N
            test_count += len(test_movies)

        precision = hit / (1.0 * rec_count)
        recall = hit / (1.0 * test_count)
        coverage = len(all_rec_movies) / (1.0 * self.movie_count)
        popularity = popular_sum / (1.0 * rec_count)

        print 'precision = %.4f\nrecall = %.4f\ncoverage = %.4f\npopularity = %.4f' \
            % (precision, recall, coverage, popularity)


if __name__ == '__main__':
    path = r'D:\file\RS\MovieLens-RecSys-python2'
    ratingfile = os.path.join(path,'ml-1m/ratings.dat')
    itemcf = ItemBasedCF()
    itemcf.generate_dataset(ratingfile)
    itemcf.calc_movie_sim()
    itemcf.evaluate()
