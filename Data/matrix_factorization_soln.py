import numpy as np
import pandas as pd
from scipy import sparse
from time import time
from numpy import matrix
from numpy.random import rand


class MatrixFactorizationRec(object):

    def __init__(self, n_features=8, learn_rate=0.005,
                 regularization_param=0.02, optimizer_pct_improvement_criterion=2):
        self.n_features = n_features
        self.learn_rate = learn_rate
        self.regularization_param = regularization_param
        self.optimizer_pct_improvement_criterion = optimizer_pct_improvement_criterion

    def fit(self, ratings_mat):
        self.ratings_mat = ratings_mat
        self.average_rating = ratings_mat.mean()
        self.n_users = ratings_mat.shape[0]
        self.n_items = ratings_mat.shape[1]
        self.n_rated = ratings_mat.nonzero()[0].size
        self.user_mat = matrix(
            rand(self.n_users*self.n_features).reshape([self.n_users, self.n_features]))
        self.movie_mat = matrix(
            rand(self.n_items*self.n_features).reshape([self.n_features, self.n_items]))
        optimizer_iteration_count = 0
        sse_accum = 0
        print("Optimizaiton Statistics")
        print("Iterations | Mean Squared Error  |  Percent Improvement")
        while ((optimizer_iteration_count < 2) or (pct_improvement > self.optimizer_pct_improvement_criterion)):
            old_sse = sse_accum
            sse_accum = 0
            for i in range(self.n_users):
                for j in range(self.n_items):
                    if self.ratings_mat[i, j] > 0:
                        error = self.ratings_mat[i, j] - np.dot(self.user_mat[i, :], self.movie_mat[:, j])
                        sse_accum += error**2
                        for k in range(self.n_features):
                            self.user_mat[i, k] = self.user_mat[i, k] + self.learn_rate * \
                                (2 * error * self.movie_mat[k, j] - self.regularization_param * self.user_mat[i, k])
                            self.movie_mat[k, j] = self.movie_mat[k, j] + self.learn_rate * \
                                (2 * error * self.user_mat[
                                 i, k] - self.regularization_param * self.movie_mat[k, j])
            pct_improvement = 100 * (old_sse - sse_accum) / old_sse
            print("%d \t\t %f \t\t %f" % (
                optimizer_iteration_count, sse_accum / self.n_rated, pct_improvement))
            old_sse = sse_accum
            optimizer_iteration_count += 1
        print("Fitting of latent feature matrices completed")

    def pred_one_user(self, user_id, report_run_time=False):
        start_time = time()
        out = self.user_mat[user_id] * self.movie_mat
        if report_run_time:
            print("Execution time: %f seconds" % (time()-start_time))
        return out

    def pred_all_users(self, report_run_time=False):
        start_time = time()
        out = self.user_mat * self.movie_mat
        if report_run_time:
            print("Execution time: %f seconds" % (time()-start_time))
        return out

    def top_n_recs(self, user_id, n):
        pred_ratings = self.pred_one_user(user_id)
        item_index_sorted_by_pred_rating = list(np.argsort(pred_ratings))
        items_rated_by_this_user = self.ratings_mat[user_id].nonzero()[1]
        unrated_items_by_pred_rating = [item for item in item_index_sorted_by_pred_rating
                                        if item not in items_rated_by_this_user]
        return unrated_items_by_pred_rating[-n:]

