
from numpy import *
import pandas as pd
from texttable import Texttable

class UserBasedModel(object):

    def learning(self,ratings,movies,neighbourNumber,recommendationNumber):
        self.ratings = ratings
        self.movies = movies
        self.k = neighbourNumber
        self.n = recommendationNumber
        self.userDict = {}
        self.itemDIct = {}
        self.neighbour = []
        self.recommandList = []

    def recommendByUser(self,userId):
        self.formatRate()
        self.getNearestNeighbor(userId)
        self.getRecommandList()
        self.getPrecision(userId)
    
    def formatRate(self):
        self.userDict = {}
        self.ItemUser = {}

        for _, row in self.ratings.iterrows():
            temp = (row.movie, float(row.rating) / 5)

            if (row.user in self.userDict):
                self.userDict[row.user].append(temp)
            else:
                self.userDict[row.user] = [temp]
            if (row.movie in self.ItemUser):
                self.ItemUser[row.movie].append(row.user)
            else:
                self.ItemUser[row.movie] = [row.user]

    def getNearestNeighbor(self, userId):
        neighbors = []
        self.neighbors = []
        for i in self.userDict[userId]:
            for j in self.ItemUser[i[0]]:
                if(j != userId and j not in neighbors):
                    neighbors.append(j)
        for i in neighbors:
            dist = self.getCost(userId, i)
            self.neighbors.append([dist, i])
        self.neighbors.sort(reverse=True)
        self.neighbors = self.neighbors[:self.k]

    def getCost(self, userId, l):
        user = self.formatuserDict(userId, l)
        x = 0.0
        y = 0.0
        z = 0.0
        for k, v in user.items():
            x += float(v[0]) * float(v[0])
            y += float(v[1]) * float(v[1])
            z += float(v[0]) * float(v[1])
        if (z == 0.0):
            return 0
        return z / sqrt(x * y)

    def formatuserDict(self, userId, l):
        user = {}
        for i in self.userDict[userId]:
            user[i[0]] = [i[1], 0]
        for j in self.userDict[l]:
            if (j[0] not in user):
                user[j[0]] = [0, j[1]]
            else:
                user[j[0]][1] = j[1]
        return user

    def getRecommandList(self):
        self.recommandList = []

        recommandDict = {}
        for neighbor in self.neighbors:
            movies = self.userDict[neighbor[1]]
            for movie in movies:
                if (movie[0] in recommandDict):
                    recommandDict[movie[0]] += neighbor[0]
                else:
                    recommandDict[movie[0]] = neighbor[0]


        for key in recommandDict:
            self.recommandList.append([recommandDict[key], key])
        self.recommandList.sort(reverse=True)
        self.recommandList = self.recommandList[:self.n]

    def getPrecision(self, userId):
        user = [i[0] for i in self.userDict[userId]]
        recommand = [i[1] for i in self.recommandList]
        count = 0.0
        if(len(user) >= len(recommand)):
            for i in recommand:
                if(i in user):
                    count += 1.0
            self.cost = count / len(recommand)
        else:
            for i in user:
                if(i in recommand):
                    count += 1.0
            self.cost = count / len(user)

    def showTable(self,userId):
        neighbors_id = [i[1] for i in self.neighbors]
        table = Texttable()
        table.set_deco(Texttable.HEADER)
        table.set_cols_dtype(["t", "t", "t", "t"])
        table.set_cols_align(["l", "l", "l", "l"])
        rows = []
        rows.append([u"movie ID", u"Name", u"genre", u"from userID"])
        nodes = []
        for item in self.recommandList:

            node = []

            fromID = []

            for _,row in self.movies.iterrows():
                if row.movieId == item[1]:
                    movie = [row.movieId,row.title,row.genres]
                    break

            for i in self.ItemUser[item[1]]:
                if i in neighbors_id:
                    fromID.append(i)
            movie.append(fromID)
            rows.append(movie)

            with open('neighbour.txt','ab') as f:
                for first in fromID:
                    temp =str(userId)+','+str(first)+'\n'
                    f.write(temp)

        table.add_rows(rows)
'''
demo = UserBasedModel()
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")
demo.learning(ratings,movies,neighbourNumber=10,recommendationNumber=20)
demo.recommendByUser(100)
demo.showTable()

'''
