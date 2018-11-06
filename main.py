import pandas as pd
import numpy as np
from scipy import sparse
import csv
from Data.matrix_factorization_soln import *
from enum import Enum


class Genre(Enum):
    Action = 0
    Adventure = 1
    Animation = 2
    Children = 3
    Comedy = 4
    Crime = 5
    Documentary = 6
    Drama = 7
    Fantasy = 8
    FilmNoir = 9
    Horror = 10
    Musical = 11
    Mystery = 12
    Romance = 13
    SciFi = 14
    Thriller = 15
    War = 16
    Western = 17


def read_rating(fname, movie_len):
    rating_file = pd.read_csv(fname)
    user_mat_len = rating_file.user.max() - rating_file.user.min() + 1
    # We want the users with reviews and ratings
    offset = rating_file.user.min()
    ratings_as_mat = sparse.lil_matrix((user_mat_len, movie_len))
    for _, row in rating_file.iterrows():
        ratings_as_mat[row.user - offset, row.movie - 1] = row.rating
    print("read file {0} complete").format(fname)
    return user_mat_len, ratings_as_mat, offset


def read_movie_genre(fname):
    with open(fname, 'rb') as f:
        movie_list = f.read().splitlines()
        movie_len = int(movie_list[movie_list.__len__() - 1].split("::")[0])
        movie_genre_mat = sparse.lil_matrix((movie_len, Genre.__len__()))
        for movie in movie_list:
            movie_genre = movie.split("::")
            for genre in movie_genre[2].split('|'):
                if (genre == "Children's"):
                    genre = "Children"
                if (genre == "Sci-Fi"):
                    genre = "SciFi"
                if (genre == "Film-Noir"):
                    genre = "FilmNoir"
                movie_genre_mat[int(movie_genre[0]) - 1, Genre[genre].value] = 1
    print("movie genre read complete")
    return movie_len, movie_genre_mat

def read_user_genre(rating_mat, movie_genre_mat):
#Todo: compute user_genre base on their rating habit
    return


def output_rating_CF(prediction_mat, ofname, first_user_idx):
    with open(ofname, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['user', 'movie', 'rating'])

        row, col = prediction_mat.nonzero()
        for i in range(len(row)):
            writer.writerow([row[i] + first_user_idx, col[i] + 1, prediction_mat[row[i], col[i]]])
        print("write file {0} complete").format(ofname)
    return


def pred_one_user(recommender, user_id, offset):
    out = recommender.user_mat[user_id] * recommender.movie_mat
    # print("predict user {} using {} complete.").format(user_id+offset, recommender.name)
    return out


def pred_all_user(recommender):
    out = recommender.user_mat * recommender.movie_mat
    # print("predict all user using {} complete.").format(recommender.name)
    return out


if __name__ == "__main__":
    movie_len, movie_genre_mat = read_movie_genre("Data/movies.dat")
    read_rating("Data/input.csv", movie_len)
