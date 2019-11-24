import numpy as np
import pandas as pd
import numpy.linalg as lin
from random import sample, choice
from functools import reduce

import data_preparation # Load dataset and build required matrices.
import factorisation # WALS factorisation.


class recommenderSystem():
    
    
    """
    TAKE CARE OF TEST AND TRAIN
    """
    
    def __init__(self, movies_df, ratings_df):
        self.movies_df = movies_df
        self.ratings_df = ratings_df
        self.R_df, self.R = data_preparation.buildR(movies_df, ratings_df)
        
        self.C = data_preparation.buildWeightMatrix(self.R)
        
        self.K = 100
        self.X = np.random.rand(np.shape(self.R)[0], self.K)
        self.Y = np.random.rand(np.shape(self.R)[1], self.K)
        
    def getUserMovies(self, user_id):
        return self.R_df[self.R_df['UserID'] == user_id]
    
    def predictionError(self):
        predicted_ratings = factorisation.predict(self.X, self.Y)
        prediction_error = factorisation.error(predicted_ratings, self.R)
        print("Prediction error: {}".format(prediction_error))
    
    def performFactorisation(self, reg_lambda, n_iter):
        factorisation.WALS(self.R, self.R, self.X, self.Y, self.C, reg_lambda, n_iter)
    
    def answerQueryAux(self, user_id):
        """
        Produces a dataframe containing the ranked recommendations for the
        unobserved items using the predicted ratings. 
        The average rating for the recommended item is displayed as well.
        """
        
        """REWRITE"""
        pred = np.matrix.round(factorisation.predict(self.X, self.Y), 2)[user_id]

        # Unseen movies.
        idx = np.where(self.R[user_id] == 0)[0]
        movie_pred = list(zip(idx, pred[idx]))

        # Build predictions and avg ratings dataframes.
        predictions_df = pd.DataFrame(movie_pred,
                                      columns = ['MovieID', 'Prediction'])
        avg_rat = self.ratings_df.groupby('MovieID').mean()
        
        dfs = [predictions_df, self.movies_df, avg_rat]
        
        recom_df = reduce(lambda left, right: 
                          pd.merge(left, right, on = "MovieID"), dfs)
        
        recom_df.drop(['UserID'], inplace = True, axis = 1)
        recom_df.round({'Rating': 1})
        recom_df.rename(columns = {'Rating':'AVG_Rating'}, inplace = True)

        return recom_df.sort_values(by = "Prediction", ascending = False)
    
    def mostPopular(self):
        """
        Produces a dataframe containing the ranked most popular items.
        """
        # movie title genre avg rating
        movie_count_df = (self.ratings_df.groupby("MovieID").size()
                          .reset_index(name = "Counts"))
        avg_rat = self.ratings_df.groupby("MovieID").mean()
        
        dfs = [self.movies_df, avg_rat, movie_count_df]
        
        recom_df = reduce(lambda left, right: 
                          pd.merge(left, right, on = "MovieID"), dfs)
        
        recom_df.drop(["UserID"], inplace = True, axis = 1)
        recom_df.rename(columns = {'Rating':'AVG_Rating'}, inplace = True)
        
        return recom_df.sort_values(by = "Counts", ascending = False)
    
    def answerQuery(self, user_id):
        """
        TODO
        """
        n_seen = len(np.where(self.R[user_id] != 0)[0])
        
        if n_seen > 10:
            recom_df = self.answerQueryAux(user_id)
        else:
            print("Too few movies! Most poular movies will be suggested.")
            recom_df = self.mostPopular()
            
        return recom_df
    
    
    # Suggesting similar items.

    def cosineSimilarity(self, d_1, d_2):
        """
        TODO
        """
        len_1 = lin.norm(d_1)
        len_2 = lin.norm(d_2)
        if len_1 == 0 or len_2 == 0:
            return -1
        return np.dot(d_1, d_2) / (len_1 * len_2)

    def similarItems(self, movie_id):
        """
        TODO
        """
        # Y is the item embedding
        d_1 = self.Y[movie_id]
        similarity = [self.cosineSimilarity(self.Y[movie_id], self.Y[i]) 
                      for i in range(0, np.shape(self.Y)[0])]
        return similarity
    
    def suggestSimilar(self, movie_id):
        """
        TODO
        """
        similarities = pd.DataFrame(self.similarItems(movie_id),
                                    columns = ["Similarity"])
        similarities_df = pd.concat([self.movies_df, similarities], axis = 1)
        return similarities_df.sort_values(by = 'Similarity',
                                           ascending = False).head(10)
    
    
    # Recommendations for new users. 

    def generateNewUser(self, n_movies):
        """
        TODO
        """
        new_user = []
        dim = np.shape(self.R)[1]

        new_user = np.zeros(dim)
        new_user_id = len(self.R)

        # Get indices of watched movies.
        obs = sample(range(dim), n_movies)
        avail_ratings = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        for i in obs:
            new_user[i] = choice(avail_ratings)

        return new_user, new_user_id
    
    def addNewUser(self, new_user, reg_lambda):
        """
        TODO
        """
        self.R, self.C, self.X = data_preparation.updateMatrices(new_user, self.R, 
                                                                 self.C, self.X)
        self.R_df = data_preparation.updateDataFrame(new_user, self.R_df,
                                                     self.movies_df)
        factorisation.newUserSinglePassWALS(new_user, self.R, self.C, self.X,
                                            self.Y, reg_lambda)