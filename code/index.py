import pandas as pd
import numpy as np
import math
from math import sqrt
import operator

def edist(instance1, instance2, length):
    distance = 0
    for i in range(length):
        distance += pow((instance1[i] - instance2[i]), 2)
    return math.sqrt(distance)


def catdist(instance1, instance2, length):
    for i in range(length):
        if instance1[i] == instance2[i]:
            return 1
        else:
            return 0


def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for j in range(len(trainingSet)):
        dist = edist(testInstance, trainingSet[j], length)
        distances.append((trainingSet[j], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for l in range(k):
        neighbors.append(distances[l][0])
    return neighbors

def pearson(person1, person2, dataset):

    return 0

def pearson_correlation(person1, person2, dataset):
    # To verify both items exist
    both_rated = {}
    for item in dataset.loc[dataset['userID'] == person1]['id']:
        if (dataset.loc[dataset['userID'] == person2]['id'] == item).any():
            both_rated[item] = 1
    # print('both_rated: ', both_rated)
    number_of_ratings = len(both_rated)
    # print('number of ratings: ', number_of_ratings)
    # Checking for number of ratings in common
    if number_of_ratings == 0:
        return 0

    # Add up all the preferences of each user
    l = 0
    for item in both_rated:
        l += ratings.loc[ratings['userID'] == person1].loc[ratings['id'] == item]['rating'].sum()
    person1_ps = l
    # print('person1_preferences_sum: ', person1_preferences_sum)
    l = 0
    for item in both_rated:
        l += ratings.loc[ratings['userID'] == person2].loc[ratings['id'] == item]['rating'].sum()
    person2_ps = l
    # print('person2: ', person2)
    # print('person2_preferences_sum: ', person2_preferences_sum)
    l = 0
    # Sum up the squares of preferences of each user
    for item in both_rated:
        l += dataset.loc[dataset['userID'] == person1].loc[dataset['id'] == item]['rating'].pow(2).sum()
    person1_square_ps = l
    # print('person1_square_preferences_sum: ', person1_square_preferences_sum)
    l = 0
    # Sum up the squares of preferences of each user
    for item in both_rated:
        l += dataset.loc[dataset['userID'] == person2].loc[dataset['id'] == item]['rating'].pow(2).sum()
    person2_square_ps = l

    l = 0

    # Sum up the product value of both preferences for each item
    for item in both_rated:
        df = dataset.loc[dataset['userID'] == person1].loc[dataset['id'] == item]['rating'].values*dataset.loc[dataset['userID'] == person2].loc[dataset['id'] == item]['rating'].values
        l += df.sum()
        # print('l :', l)
        # print(dataset.loc[dataset['userID'] == person1].loc[dataset['id'] == item]['rating'])
        # print(dataset.loc[dataset['userID'] == person2].loc[dataset['id'] == item]['rating'])
    product_sum_of_both_users = l
    # print('product_sum_of_both_users: ', product_sum_of_both_users)

    # Calculate the pearson score
    numerator_value = product_sum_of_both_users - (person1_ps * person2_ps / number_of_ratings)
    # print('numerator_value: ', numerator_value)
    denominator_value = sqrt((person1_square_ps - pow(person1_ps, 2) / number_of_ratings) * (
        person2_square_ps - pow(person2_ps, 2) / number_of_ratings))
    # print('denominator_value: ', denominator_value)
    if denominator_value == 0:
        return 0
    else:
        r = numerator_value / denominator_value
        return r


def most_similar_users(person, number_of_users, dataset):
    # returns the number_of_users (similar persons) for a given specific person.
    scores = [(pearson_correlation(person, other_person, dataset), other_person) for other_person in np.unique(dataset['userID'][:500].values.ravel()) if
              other_person != person]

    # Sort the similar persons so that highest scores person will appear at the first
    scores.sort()
    scores.reverse()
    return scores[:]

def user_reommendations(person, dataset):
    # Gets recommendations for a person by using a weighted average of every other user's rankings
    totals = {}
    simSums = {}
    rankings_list = []
    for other in dataset['userID']:
        # don't compare me to myself
        if other == person:
            continue
        sim = pearson_correlation(person, other, dataset)

        # ignore scores of zero or lower
        if sim <= 0:
            continue
        for item in dataset.loc[dataset['userID'] == other]['id']:

            # only score movies i haven't seen yet
            if (dataset.loc[dataset['userID'] == person]['id'] != item).any():
                #   if item not in dataset[person] or dataset[person][item] == 0:
                # Similrity * score
                totals.setdefault(item, 0)
                totals[item] = dataset.loc[dataset['userID'] == other].loc[dataset['id'] == item]['rating'] * sim
                # sum of similarities
                # simSums.setdefault(item, 0)
                # simSums[item] += sim

            # Create the normalized list

    result = [(ranking, item) for item, ranking in totals.items()]
    # returns the recommended items
    rec_list = [recommend_item for ranking, recommend_item in result]
    return rec_list


def show_result(person, ratings, movies):

    res = user_reommendations(person, ratings[:300])
    print(len(res))
    for i in range(len(res)):
        print(movies.loc[movies['id'] == res[i]]['title'])

# input
users = pd.read_csv("../data/users.csv")  # , index_col="userID")
ratings = pd.read_csv("../data/ratings.csv")  # , index_col="userID")
movies = pd.read_csv("../data/movies.csv", encoding="iso8859-2")
# print(users.head(5))
# print(movies.head(5))
# print(ratings.head(5))
# t["test"] = [t[i] for x in movies["genres"]for i in x.split('|')]
s = {}
k = 1
both_rated = {}
both_rated[1197] = 1
# f = [item for item in np.unique(ratings['userID'].values.ravel())]
# print(f)
# users['items'] = {}





    # Checking for number of ratings in common

    # Add up all the preferences of each user

# person1_preferences_sum = [ratings.loc[ratings['userID'] == 1].loc[ratings['id'] == item]['rating'].sum(axis=1) for item in both_rated]

print(both_rated)
print([item for item in both_rated])
print('******')
print(show_result(3, ratings, movies))




'''
for index, x in ratings.iterrows():
    if ratings.loc[index, 'userID'] == k+1:
        users.loc[k-1, 'items'] = 1
        s = {}
        k += 1
    s[ratings.loc[index, 'id']] = ratings.loc[index, 'rating']

print(users.head())
'''

'''

# Movies cleaning
s = []
1
for index, x in movies.iterrows():
    s = x["genres"].split('|')
    movies.loc[index, 2:] = 0
    for i in s:
        # print(t.iloc[index])
        movies.loc[index, i] = 1
movies = movies.fillna(0)
movies = movies.drop(['title', 'genres'], axis=1)
# print(movies.head(5))
'''

'''
# Users cleaning
users.loc[users['gender'] == 'F', 'gender'] = 0
users.loc[users['gender'] == 'M', 'gender'] = 1
users['gender'] = users['gender'].astype(int)

one_hot = pd.get_dummies(users['occupation'], prefix='occ')
users = users.join(one_hot)
users.drop(['occupation'], axis=1, inplace=True)


# Ratings cleaning
ratings.drop(['time'], axis=1, inplace=True)

# Final dataset

dataset = users
dataset.drop(['zip'], axis=1, inplace=True)

k = 0
s = {}
for i in range(6040):
    if ratings.loc[k, 'userID'] == i+1:
        s[ratings[k]['id']] = ratings[k]['rating']
    else:
        users[k]['items'] = s
        k += 1
        s = {}
        s[ratings[k]['id']] = ratings[k]['rating']
print(users.head())


'''

