import pandas as pd
import numpy as np
from scipy import sparse
import csv
from enum import Enum
from LFM import *
import matplotlib.pyplot as plt
from UserBasedModel import *

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


def genre_array():
    genre_array = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', \
                   'Documentary', 'Drama', 'Fantasy', 'FilmNoir', 'Horror', 'Musical', \
                   'Mystery', 'Romance', 'SciFi', 'Thriller', 'War', 'Western']
    return genre_array


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
    movie = pd.read_csv(fname,sep='::')
    #movie_genre_mat = sparse.lil_matrix((movie.movieId.max()+1, Genre.__len__()))
    movie_genre_mat = np.random.rand(Genre.__len__(), movie.movieId.max()+1)
    genre_distribute = [0 for i in range(18)]
    for _, row in movie.iterrows():
        for genre in row.genres.split('|'):
            if (genre == "Children's"):
                genre = "Children"
            if (genre == "Sci-Fi"):
                genre = "SciFi"
            if (genre == "Film-Noir"):
                genre = "FilmNoir"
            movie_genre_mat[Genre[genre].value, row.movieId] = 1.2
            genre_distribute[Genre[genre].value] += 1
    print("movie genre read complete")
    return movie.movieId.max()+1, movie_genre_mat, genre_distribute

def get_user_genre(rating_mat, movie_genre_mat, offset):
    user_genre =np.zeros((user_len, Genre.__len__()))
    with open('feature.txt','wb') as f:
        for user in range(user_len):
            print('\n'+str(user+offset)+'\t')
            f.write(str(user+offset)+'\t')
            count=1
            for movie in range(movie_len):
                if rating_mat[user,movie]!=0:
                    count+=1
                    for feature in range(Genre.__len__()):
                        user_genre[user,feature] += rating_mat[user,movie]*movie_genre_mat[feature,movie]
            for feature in range(Genre.__len__()):
                user_genre[user, feature]=user_genre[user, feature]/count
                user_genre[user, feature] = round(user_genre[user, feature],3)
                f.write(str(user_genre[user,feature])+'\t')

        f.write('\n')
        print('\n')
    return user_genre

def rating_user_info(user_genre,user_info):
    occu = np.zeros((21,Genre.__len__()))
    age = np.zeros((7,Genre.__len__()))
    gender = np.zeros((2,Genre.__len__()))
    occu_counter=[1 for i in range(21)]
    age_counter = [1 for i in range(7)]
    gender_counter = [1 for i in range(2)]
    for _, row in user_info.iterrows():
        age_idx=0
        if row.Age==1:
            age_idx=0
        elif row.Age==18:
            age_idx=1
        elif row.Age==25:
            age_idx=2
        elif row.Age == 35:
            age_idx = 3
        elif row.Age == 45:
            age_idx = 4
        elif row.Age == 50:
            age_idx = 5

        genderidf =0
        if row.Gender=='M':
            genderidf=0
        else:
            genderidf=1
        for i in range(Genre.__len__()):
            occu[row.Occupation,i] += user_genre[row.UserID-offset,i]
            gender[genderidf,i]+=user_genre[row.UserID-offset,i]
            age[age_idx,i]+=user_genre[row.UserID-offset,i]

        occu_counter[row.Occupation]+=1
        age_counter[age_idx]+=1
        gender_counter[genderidf]+=1
    '''
    for i in range(Genre.__len__()):
        for j in range(21):
            occu[j,i]/=occu_counter[j]
        for j in range(7):
            age[j,i]/=age_counter[j]
        for j in range(2):
            gender[j,i]/=age_counter[j]
    '''
    with open('user_info_based.txt', 'wb') as f:
        f.write("Job\n")
        #print("Job\n")
        for j in range(21):
            f.write('\n'+str(j))
            for i in range(Genre.__len__()):
                occu[j, i] /= occu_counter[j]
                f.write(str(round(occu[j,i],3))+'\t')
                #print(str(round(occu[j,i],3))+'\t')
        '''
        f.write("Age\n")
        for j in range(7):
            f.write(str(j) + '\n')
            for i in range(Genre.__len__()):
                age[j, i] /= age_counter[j]
                f.write(str(age(gender[j,i],3))+'\t')
        f.write("Gender\n")
        for j in range(2):
            f.write(str(j) + '\n')
            for i in range(Genre.__len__()):
                gender[j, i] /= age_counter[j]
                f.write(str(round(gender[j,i],3))+'\t')
        '''
    return occu

def plot_genre(genre):
    plt.title("genre distribution")
    plt.ylabel("occurance")
    plt.xlabel("genre")
    x = np.arange(18)
    plt.bar(x, height=genre)
    plt.xticks(x, genre_array())
    plt.show()
    #print(genre)
    return


def output_rating_CF(prediction_mat, ofname, offset):
    with open(ofname, 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(['user', 'movie', 'rating'])
        row, col = prediction_mat.nonzero()
        for i in range(len(row)):
            writer.writerow([row[i] + offset, col[i] + 1, prediction_mat[row[i], col[i]]])
            print("write file {0} complete").format(ofname)
    return

if __name__ == "__main__":

    movie_len, movie_genre_mat, movie_genre_distribute = read_movie_genre("movies.dat")
    user_len, rating_mat, offset = read_rating("rating.csv", movie_len)

    plot_genre(movie_genre_distribute)

    user_genre =  get_user_genre(rating_mat,movie_genre_mat,offset)
    user_info = pd.read_csv("users.dat",sep='::')
    rating_user_info(user_genre,user_info)
    rate_engine = MF_LFM()
    rate_engine.learningLFM(rating_mat,movie_genre_mat, 18, 2, 0.05, 0.02, 0.02)
    print(rate_engine.top_n_recs(2789,10))

    '''
    rate_engine2= UserBasedModel()
    movies = pd.read_csv("movies.dat",sep='::')
    ratings = pd.read_csv("rating.csv")
    rate_engine2.learning(ratings, movies, neighbourNumber=40, recommendationNumber=20)
    for i in range(2790,2990):
        rate_engine2.recommendByUser(i)
        rate_engine2.showTable(i)
        print(i)
    '''

