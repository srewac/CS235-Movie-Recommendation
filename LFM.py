import numpy as np
import pandas as pd
from scipy import sparse
from time import time
from numpy import matrix
from numpy.random import rand


class MF_LFM(object):
    # rating_mat is input
    # features is hidden vector
    # alpha stands for learning rate
    # lamb stands for regularition parameter
    # criterion is judge factor

    def learningLFM(self, ratings_mat, features, learn_loops, alpha, lamb, criterion):
        self.ratings_mat = ratings_mat
        self.n_users = ratings_mat.shape[0]
        self.n_items = ratings_mat.shape[1]
        self.n_have_rated = ratings_mat.nonzero()[0].size
        self.user_mat = matrix(rand(self.n_users, features))
        self.movie_mat = matrix(rand(features, self.n_items))
        iteration_fix_count = 0
        curren_error = 100
        for step in xrange(0, 5000, 1):
            previous_error = curren_error
            curren_error = 0
            for user_index in range(self.n_users):
                for movie_index in range(self.n_items):
                    error = self.ratings_mat[user_index, movie_index]
                    if self.ratings_mat[user_index, movie_index] > 0:
                        row = self.user_mat[user_index, :]
                        column = self.movie_mat[:, movie_index]
                        error -= np.dot(row, column)
                        curren_error += pow(error, 2)
                        for features_index in range(features):
                            user_value = self.user_mat[user_index, features_index]
                            movie_value = self.movie_mat[features_index, movie_index]

                            delta_user = alpha * (error * movie_value - lamb * user_value)
                            self.user_mat[user_index, features_index] += delta_user

                            delta_movie = alpha * (error * user_value - lamb * movie_value)
                            self.movie_mat[features_index, movie_index] += delta_movie
            percentage = float(previous_error - curren_error) / previous_error
            print("Learning LFM Process")
            #print("Loops | MSE  | Improved percentage")
            #print("%d\t%f\t%f" % (iteration_fix_count + 1, curren_error / self.n_have_rated, percentage))
            previous_error = curren_error

            if (iteration_fix_count > learn_loops) and (percentage < criterion):
                break
            iteration_fix_count += 1
        print("Latent Factor Model done!")
