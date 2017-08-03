#-*- coding: utf-8 -*-
'''
Created on 2017-07-22

@author: Ivan
'''
import sys
import random
import math
import os
from operator import itemgetter


random.seed(0)


class UserBasedCF(object):
    # TopN recommendation - User Based Collaborative Filtering 

    def __init__(self):
        self.trainset = {}
        self.testset = {}

        self.n_sim_user = 20 # 取得分相似的k个用户计算
        self.n_rec_movie = 10 # 推荐得分最高的k个物品

        self.user_sim_mat = {} # 用户相似度矩阵
        self.movie_popular = {} # 每个物品被产生过行为的用户数
        self.movie_count = 0 # 记录一共有几个物品

        print 'Similar user number = %d' % self.n_sim_user
        print 'recommended movie number = %d' % self.n_rec_movie

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
        # load rating data and split it to training set and test set
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

    def calc_user_sim(self):
        # calculate user similarity matrix 
        # build inverse table for item-users
        # key = movieID, value = list of userIDs who have seen this movie
        print 'building movie-users inverse table...'
        movie2users = dict()

        for user, movies in self.trainset.iteritems(): # 对每个用户记录进行遍历
            for movie in movies: # 对一个用户的每个物品进行遍历
                # inverse table for item-users
                if movie not in movie2users: # 建立物品-用户倒排表
                    movie2users[movie] = set()
                movie2users[movie].add(user)
                # count item popularity at the same time
                if movie not in self.movie_popular: # 记录每个物品被多少个用户产生过行为/流行度
                    self.movie_popular[movie] = 0
                self.movie_popular[movie] += 1 # 时间信息：1 / 1 + alpha * (T - t),T为当前时间
        print 'build movie-users inverse table successfully'

        # save the total movie number, which will be used in evaluation
        self.movie_count = len(movie2users) # 物品-用户到排表长度
        print 'total movie number = %d' % self.movie_count
#        print 'total user number = %d' % self.movie_count
        
        # count co-rated items between users
        usersim_mat = self.user_sim_mat
        print 'building user co-rated movies matrix...'

        for movie, users in movie2users.iteritems():
            for u in users:
                for v in users:
                    if u == v:
                        continue
                    usersim_mat.setdefault(u, {})
                    usersim_mat[u].setdefault(v, 0)
                    usersim_mat[u][v] += 1 #惩罚热门物品 1 / math.log(1 + len(users)) # 时间信息：1 / (1 + alpha * abs(tui - tvi))
        print 'build user co-rated movies matrix successfully'

        # calculate similarity matrix
        print 'calculating user similarity matrix...'
        simfactor_count = 0
        PRINT_STEP = 2000000
        # 用户相似度矩阵
        for u, related_users in usersim_mat.iteritems():
            for v, count in related_users.iteritems():
                usersim_mat[u][v] = count / math.sqrt(
                    len(self.trainset[u]) * len(self.trainset[v]))
                simfactor_count += 1
                if simfactor_count % PRINT_STEP == 0:
                    print 'calculating user similarity factor(%d)' % simfactor_count

        print 'calculate user similarity matrix(similarity factor) successfully'
        print 'Total similarity factor number = %d' % simfactor_count

    def recommend(self, user):
        # Find K similar users and recommend N movies
        K = self.n_sim_user
        N = self.n_rec_movie
        rank = dict()
        watched_movies = self.trainset[user] # 推荐一个用户，用户-物品表

        for similar_user, similarity_factor in sorted(self.user_sim_mat[user].items(), # 最相似的k个用户
                                                      key = itemgetter(1), reverse = True)[0:K]:
            for movie, rating in self.trainset[similar_user].items(): # 对每个相似的用户的所有物品遍历
                if movie in watched_movies:
                    continue
                # predict the user's "interest" for each movie
                rank.setdefault(movie, 0)
                rank[movie] += similarity_factor * rating #时间信息 similarity_factor * rating / (1 + alpha * (t0 - yvi)) # 惩罚热门物品(提高新颖度)：pui / math.log(1 + alpha * popularity(i))
        
        # return the N best movies
        return sorted(rank.items(), key = itemgetter(1), reverse = True)[0:N]

    def evaluate(self):
        # print evaluation result: precision, recall, coverage and popularity
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

        print 'precision = %.4f\nrecall = %.4f\ncoverage = %.4f\npopularity = %.4f' % \
            (precision, recall, coverage, popularity)


if __name__ == '__main__':
    path = r'D:\file\RS\MovieLens-RecSys-python2'
    ratingfile = os.path.join(path,'ml-1m/ratings.dat')
    usercf = UserBasedCF()
    usercf.generate_dataset(ratingfile)
    usercf.calc_user_sim()
    usercf.evaluate()
