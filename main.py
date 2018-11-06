import pandas as pd
import numpy as np
from scipy import sparse
import csv
from Data.matrix_factorization_soln import *

def read_rating(fname):
    rating_file = pd.read_csv(fname)
    user_mat_len = rating_file.user.max() - rating_file.user.min() + 1
    # We want the users with reviews and ratings
    offset = rating_file.user.min()
    movie_mat_len = rating_file.movie.max()
    ratings_as_mat = sparse.lil_matrix((user_mat_len, movie_mat_len))
    for _, row in rating_file.iterrows():
        ratings_as_mat[row.user - offset, row.movie - 1] = row.rating
    print("read file {0} complete").format(fname)
    return offset, ratings_as_mat


def output_rating_CF(prediction_mat, ofname, first_user_idx):
    with open(ofname, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['user', 'movie', 'rating'])

        row, col = prediction_mat.nonzero()
        for i in range(len(row)):
            writer.writerow([row[i] + first_user_idx, col[i]+1, prediction_mat[row[i], col[i]]])
        print("write file {0} complete").format(ofname)
    return

def pred_one_user(recommender, user_id, offset):
    out = recommender.user_mat[user_id]* recommender.movie_mat
    print("predict user {} using {} complete.").format(user_id+offset, recommender.name)
    return out

def pred_all_user(recommender):
    out = recommender.user_mat * recommender.movie_mat
    print("predict all user using {} complete.").format(recommender.name)
    return out

if __name__ == "__main__":
    offset, rating_mat = read_rating("Data/input.csv")
    output_rating_CF(rating_mat, "Data/test.csv", offset)
    recommender = MatrixFactorizationRec()
    recommender.fit(rating_mat)
    predictions_mat = pred_one_user(recommender, 0, offset)
    print(predictions_mat)
    output_rating_CF(predictions_mat, "Data/test.csv", offset)