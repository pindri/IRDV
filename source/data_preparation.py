# Set of functions used to read the dataset and build/update the
# appropriate matrices. 


from pathlib import Path
from scipy.sparse import coo_matrix
import csv
import pandas as pd
import numpy as np


def importDataset(dataset_fraction = 1.0):
    """
    Imports the ml-latest-small dataset, reading ratings and movies.
    From that, it builds and returns a rating dataframe and a movies
    dataframe.
    Only a fraction of the dataset can be imported if a number between
    0 and 1 is passed as an argument. By default, it imports the whole
    dataset.
    
    Training and test sets are divided as follows: for each
    user, the most recent 10 ratings constitute the test set.
    """
    
    __file__ = 'recommender.ipynb'
    base_path = Path(__file__).parent

    file_path = (base_path / '../ml-latest-small/ratings.csv').resolve()
    with open(file_path) as f:
        ratings = [line for line in csv.reader(f)]

    file_path = (base_path / '../ml-latest-small/movies.csv').resolve()
    with open(file_path) as f:
        movies = [line for line in csv.reader(f)]
        
    n_movies = int(dataset_fraction * len(movies))    

    # Building dataframes, fixing types, dropping useless columns.
    # The `- 1` fixes indices, making them start at 0.

    ratings_df = pd.DataFrame(ratings,columns = ['UserID', 'MovieID','Rating',
                                                 'Timestamp']).iloc[1:]
    ratings_df[['UserID', 'MovieID']] = ratings_df[['UserID',
                                                    'MovieID']].astype(int)-1
    ratings_df[['Rating']] = ratings_df[['Rating']].astype(float)


    movies_df = pd.DataFrame(movies, columns = ['MovieID', 'Title',
                                                'Genres']).iloc[1:n_movies]
    movies_df[['MovieID']] = movies_df[['MovieID']].astype(int)-1
    
    
    # Movie index corre-small_ dataset `MovieId`s do not increase
    # continuously. Even if less than 10000 movies are present,
    # the index goes up to ~19000. In order to fix this unconvenience
    # and make the dataframe indexing more intuitive,
    # a more appropriate index has been built.
    # If necessary, a reverse conversion to the original one
    # could be achieved by storing a two column conversion dataframe.
    
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
    ratings_df.columns = ['MovieID', 'UserID', 'Rating', 'Timestamp']
    
    
    # Extracting test ratings (10 most recent ratings for each user).
    ratings_df.sort_values(by = ['UserID', 'Timestamp'])
    
    ratings_df_test = pd.DataFrame(columns = ratings_df.columns)
    
    for i in range(ratings_df['UserID'].nunique()):
        # Test set is 10 of observations.
        #n_test = int(0.2 * len(ratings_df[ratings_df['UserID'] == i]))
        n_test = 10
        ratings_df_test = ratings_df_test.append(ratings_df[ratings_df
                                                            ['UserID'] ==
                                                            i].tail(n_test),
                                                ignore_index = True)
        ratings_df.drop(ratings_df[ratings_df['UserID'] == i]
                        .tail(n_test).index,
                        inplace = True)
    
    # Delete now useless timestamp columns.
    ratings_df.drop(['Timestamp'], inplace = True, axis = 1)
    ratings_df_test.drop(['Timestamp'], inplace = True, axis = 1)
    
    return movies_df, ratings_df, ratings_df_test


def buildR(movies_df, ratings_df, is_test = False):
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

    if is_test == False:
        print("The dataframe contains {} users and {} items."
              .format(np.shape(R)[0], np.shape(R)[1]))
    
    return R_df, R


def buildWeightMatrix(R, alpha = 10, w0 = 0.1):
    """
    Builds the weight matrix.
    """
    # The commented lines suggest a viable alternative.
    
    c = [np.count_nonzero(R[:, i]) for i in range(0, np.shape(R)[1])]
    C = R * c + w0
    #C = 1 + alpha * R

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
    R_df = R_df.append(new_df,
                       ignore_index = True).sort_values(by = ['MovieID',
                                                              'UserID'])
    
    return R_df
