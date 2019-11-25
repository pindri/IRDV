# Set of functions used to read the dataset and build/update the
# appropriate matrices. 


from pathlib import Path
from scipy.sparse import coo_matrix
import csv
import pandas as pd
import numpy as np


def importDataset():
    """
    Imports the ml-latest-small dataset, reading ratings and movies.
    From that, it builds and returns a rating dataframe and a movies
    dataframe.
    """
    
    __file__ = 'recommender.ipynb'
    base_path = Path(__file__).parent

    file_path = (base_path / '../ml-latest-small/ratings.csv').resolve()
    with open(file_path) as f:
        ratings = [line for line in csv.reader(f)]

    file_path = (base_path / '../ml-latest-small/movies.csv').resolve()
    with open(file_path) as f:
        movies = [line for line in csv.reader(f)]
        

    # Building dataframes, fixing types, dropping useless columns.
    # The `- 1` fixes indices, making them start at 0.

    ratings_df = pd.DataFrame(ratings,columns = ['UserID', 'MovieID','Rating',
                                                 'Timestamp']).iloc[1:]
    ratings_df[['UserID', 'MovieID']] = ratings_df[['UserID',
                                                    'MovieID']].astype(int) - 1
    ratings_df[['Rating']] = ratings_df[['Rating']].astype(float)
    ratings_df.drop(['Timestamp'], inplace = True, axis = 1)


    movies_df = pd.DataFrame(movies, columns = ['MovieID', 'Title',
                                                'Genres']).iloc[1:3000]
    movies_df[['MovieID']] = movies_df[['MovieID']].astype(int) - 1
    
    
    # Movie index correction.
    # In the _ml-latest-small_ dataset `MovieId`s do not increase continuously. 
    # Even if less than 10000 movies are present, the index goes up to ~19000. 
    # In order to fix this unconvenience and make the dataframe indexing more
    # intuitive, a more appropriate index has been built.
    # If necessary, a reverse conversion to the original one could be achieved
    # by storing a two column conversion dataframe.
    
    n_movies = movies_df['MovieID'].shape[0]
    movie_index = pd.DataFrame([i for i in 
                                range(0, n_movies)], columns = ['NewID'])
    movie_index['MovieID'] = movies_df['MovieID'].to_numpy()

    # Fix the MovieIDs of the movies_df dataframe.
    movies_df = pd.merge(movie_index, movies_df, on = 'MovieID', 
                         how = 'inner').drop(['MovieID'], axis = 1)
    movies_df.columns = ['MovieID', 'Title', 'Genres']

    # Fix the MovieIDs of the ratings_df dataframe.
    ratings_df = pd.merge(movie_index, ratings_df, on = 'MovieID',
                          how = 'inner').drop(['MovieID'], axis = 1)
    ratings_df.columns = ['MovieID', 'UserID', 'Rating']
    
    return movies_df, ratings_df


def buildR(movies_df, ratings_df):
    """
    Builds the sparse rating dataframe and matrix starting from the
    movies/rating dataframes.
    """
    
    # Dataframe.
    
    R_df = pd.merge(ratings_df, movies_df, on = 'MovieID', how = 'inner')
    R_df = pd.pivot_table(R_df, index = ['MovieID', 'UserID', 'Genres',
                                         'Title'])
    R_df = pd.DataFrame(R_df.to_records())
    
    # R matrix.
    
    R_users = R_df['UserID'].to_numpy().flatten()
    R_movies = R_df['MovieID'].to_numpy().flatten()
    R_ratings = R_df['Rating'].to_numpy().flatten()

    # Matrices in COOrdinate formate can be built using
    # the syntax: csr_matrix((dat, (row, col))).
    R = coo_matrix((R_ratings, (R_users, R_movies)))
    R = R.toarray()

    print("The dataframe contains {} users and {} items."
          .format(np.shape(R)[0], np.shape(R)[1]))
    
    return R_df, R


def buildWeightMatrix(R, alpha = 10, w0 = 1):
    """
    Builds a weight matrix.
    """
    # The commented lines suggest a viable alternative.
    
    #c = [np.count_nonzero(R[:, i]) for i in range(0, np.shape(R)[1])]
    #C = R * c + w0
    C = 1 + alpha * R

    return C


def updateMatrices(new_user, R, C, X):
    """
    Updates the ratings, weight and user-embedding matrices when a new user
    is added into the dataset.
    """
    R = np.vstack((R, new_user))
    C = buildWeightMatrix(R, alpha = 10)
    X = np.vstack((X, np.random.rand(np.shape(X)[1])))
    
    return R, C, X


def updateDataFrame(new_user, R_df, movies_df):
    """
    Updates the ratings dataframes when a new user is added into the dataset.
    """
    # First, create a new dataframe for the new_user.
    new_df = pd.DataFrame(new_user, columns=['Rating'])
    new_df['MovieID'] = range(0, len(new_user))
    new_df['UserID'] = R_df['UserID'].max() + 1
    new_df = new_df[new_df['Rating'] != 0]
    new_df = pd.merge(new_df, movies_df, on = 'MovieID', how = 'inner')
    new_df = new_df[['MovieID', 'UserID', 'Genres', 'Title', 'Rating']]
    
    # Then, append the new dataframe to the former R_df.
    R_df = R_df.append(new_df, ignore_index = True).sort_values(by = ['MovieID', 'UserID'])
    
    return R_df
