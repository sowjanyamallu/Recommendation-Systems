# coding: utf-8

# # Assignment 3:  Recommendation systems
#
# Here we'll implement a content-based recommendation algorithm.
# It will use the list of genres for a movie as the content.
# The data come from the MovieLens project: http://grouplens.org/datasets/movielens/

# Please only use these imports.
from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile

def download_data():
    """ DONE. Download and unzip data.
    """
    url = 'https://www.dropbox.com/s/h9ubx22ftdkyvd5/ml-latest-small.zip?dl=1'
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')
    zfile = zipfile.ZipFile('ml-latest-small.zip')
    zfile.extractall()
    zfile.close()

def tokenize_string(my_string):
    """ DONE. You should use this in your tokenize function.
    """
    return re.findall('[\w\-]+', my_string.lower())


def tokenize(movies):
    """
    Append a new column to the movies DataFrame with header 'tokens'.
    This will contain a list of strings, one per token, extracted
    from the 'genre' field of each movie. Use the tokenize_string method above.

    Note: you may modify the movies parameter directly; no need to make
    a new copy.
    Params:
      movies...The movies DataFrame
    Returns:
      The movies DataFrame, augmented to include a new column called 'tokens'.

    >>> movies = pd.DataFrame([[123, 'Horror|Romance'], [456, 'Sci-Fi']], columns=['movieId', 'genres'])
    >>> movies = tokenize(movies)
    >>> movies['tokens'].tolist()
    [['horror', 'romance'], ['sci-fi']]
    """
    ###TODO
    t = []
    for r in (movies['genres']):
        c = tokenize_string(r)
        t.append(c)
    movies['tokens'] = t
    return movies


def featurize(movies):
    """
    Append a new column to the movies DataFrame with header 'features'.
    Each row will contain a csr_matrix of shape (1, num_features). Each
    entry in this matrix will contain the tf-idf value of the term, as
    defined in class:
    tfidf(i, d) := tf(i, d) / max_k tf(k, d) * log10(N/df(i))
    where:
    i is a term
    d is a document (movie)
    tf(i, d) is the frequency of term i in document d
    max_k tf(k, d) is the maximum frequency of any term in document d
    N is the number of documents (movies)
    df(i) is the number of unique documents containing term i

    Params:
      movies...The movies DataFrame
    Returns:
      The movies DataFrame, which has been modified to include a column named 'features'.
    """
    ###TODO
    v = []
    kc = 0
    h = defaultdict(list)
    vocab = defaultdict(lambda: 0)
    for r1 in movies['tokens']:
        for x in r1:
            v.append(x)
    counter = Counter(v)
    for k1, val in sorted(counter.items()):
        vocab[k1] += kc
        kc += 1
    csr = []
    for j in range(len(movies['tokens'])):
        for e1 in movies.tokens[j]:
            h[e1].append(movies.title[j])
    for d in movies['tokens']:
        m = 0
        data = []
        row = []
        col = []
        count = Counter(sorted(d))
        MAX = max(count.values())
        N = len(movies['movieId'])
        for k4 in sorted(d):
            col.append(vocab[k4])
            row.append(m)
            for k5, vl in (h.items()):
                if k5 == k4:
                    df = len(vl)
            tf_idf = ((count[k4]) / MAX) * math.log10(N / df)
            data.append(tf_idf)
        X = csr_matrix((data, (row, col)), shape=(1, len(counter.keys())))
        csr.append(X)
    movies['features'] = csr
    #print(movies['features'])
    return tuple((movies, vocab))

def train_test_split(ratings):
    """DONE.
    Returns a random split of the ratings matrix into a training and testing set.
    """
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]


def cosine_sim(a, b):
    """
    Compute the cosine similarity between two 1-d csr_matrices.
    Each matrix represents the tf-idf feature vector of a movie.
    Params:
      a...A csr_matrix with shape (1, number_features)
      b...A csr_matrix with shape (1, number_features)
    Returns:
      The cosine similarity, defined as: dot(a, b) / ||a|| * ||b||
      where ||a|| indicates the Euclidean norm (aka L2 norm) of vector a.
    """
    ###TODO
    #def Euclidean_norm(a1):
     #   return np.sqrt(sum([m*m for m in a1]))
    #denominator= Euclidean_norm(a)*Euclidean_norm(b)
    #dist= math.sqrt(a.dot(a) - 2 * a.dot(b) + b.dot(b))
    return ((a.dot(b.transpose()))/(np.sqrt(a*a.transpose())*np.sqrt(b*b.transpose()))).tolist()[0][0]


def make_predictions(movies, ratings_train, ratings_test):
    """
    Using the ratings in ratings_train, predict the ratings for each
    row in ratings_test.

    To predict the rating of user u for movie i: Compute the weighted average
    rating for every other movie that u has rated.  Restrict this weighted
    average to movies that have a positive cosine similarity with movie
    i. The weight for movie m corresponds to the cosine similarity between m
    and i.

    Params:
      movies..........The movies DataFrame.
      ratings_train...The subset of ratings used for making predictions. These are the "historical" data.
      ratings_test....The subset of ratings that need to predicted. These are the "future" data.
    Returns:
      A numpy array containing one predicted rating for each element of ratings_test.
    """
    ###TODO
    prediction = []

    rating = []

    #print(ratings_train)
    #print(ratings_test)
    for utest in ratings_test.itertuples():
        z = []
        product = []
        atest = ((movies[movies.movieId == utest.movieId]).features.values[0])
        for utrain in ratings_train.itertuples():
            #print(ratings_train['userId==1'])
            #print(ratings_train)
            if utest.userId == utrain.userId :
                #print(utest.userId)
                #print(ratings_train)
                #vc.append(jl)
                #print(utrain.movieId)
                #print(ratings_train[utrain.userId])
                btrain = ((movies[movies.movieId == utrain.movieId]).features.values[0])
                #print(btrain)
                cosine = cosine_sim(atest, btrain)
                #print(cosine)
                c = ((ratings_train[ratings_train.movieId == utrain.movieId]).rating.values[0])
                #print(c)

                #if (cosine >= 0):
                z.append(cosine)
                product.append(cosine*c)
                #print(product)
                rating.append(c)
        #print(len(z))
        #vc.append(jl)
        if (sum(z))>0:
            h1 = sum(z)
            #print(h1)
            j = sum(product)
            #print(j)
            d=j/h1
            prediction.append(d)
        else:
            prediction.append((np.mean(rating)))
    return prediction



def mean_absolute_error(predictions, ratings_test):
    """DONE.
    Return the mean absolute error of the predictions.
    """
    return np.abs(predictions - np.array(ratings_test.rating)).mean()


def main():
    download_data()
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    movies, vocab = featurize(movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test)
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])


if __name__ == '__main__':
    main()
