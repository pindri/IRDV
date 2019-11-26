import numpy as np
import pandas as pd
import numpy.linalg as lin
from random import sample, choice
from functools import reduce

import data_preparation # Load dataset and build required matrices.
import factorisation # WALS factorisation.


class recommenderSystem():
    
    
    """
    TODO CLASS DOCUMENTATION
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
        prediction_error = factorisation.MAE(predicted_ratings, self.R) 
        return prediction_error
    
    def performFactorisation(self, reg_lambda, n_iter):
        train_err, test_err = factorisation.WALS(self.R, self.R, self.X,
                                                 self.Y, self.C,
                                                 reg_lambda, n_iter)
        return train_err, test_err
    
    def answerQueryAux(self, user_id):
        """
        Produces a dataframe containing the ranked recommendations for the
        unobserved items using the predicted ratings. 
        The average rating for the recommended item is displayed as well.
        """
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
        Returns a dataframe of ranked recommendations for user_id.
        If user_id has rated less than 10 movies, the most popular
        movies will be returned.
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
        
        
    def computeFolds(self, n_folds):
        """
        TODO
        """
        folds_indices = [i for i in range(n_folds)]
        p_folds = [1./(n_folds) for _ in range(n_folds)]

        # Mask used to determine the fold of each element.
        mask = np.random.choice(a = folds_indices, size = self.R.size,
                                p = p_folds).reshape(self.R.shape)

        # These will hold the k folds.
        k_folds = [np.zeros(self.R.shape) for _ in range(n_folds)]
        for i in range(n_folds):
            k_folds[i][mask == i] = self.R[mask == i]

        return k_folds

        
    def kFoldCV(self, n_folds, n_iter, reg_lambda):
        """
        TODO
        """
        
        k_folds = self.computeFolds(n_folds)
        k_train_err = []
        k_test_err = []

        for i in range(n_folds):
            R_test = k_folds[i]
            R_train = sum(k_folds) - k_folds[i]
            train_err, test_err = factorisation.WALS(R_train, R_test, self.X,
                                                     self.Y, self.C,
                                                     reg_lambda, n_iter)

            # Appending the last train/test errors from WALS.
            k_train_err.append(train_err[-1])
            k_test_err.append(test_err[-1])
        

        return (sum(k_train_err) / len(k_train_err),
                sum(k_test_err) / len(k_test_err))
    
    
    def bestLambdaCV(self, n_folds, n_iter, reg_lambda):
        """
        NOTE: requires reg_lambda to be a list
        """
        print("Performing {} fold CV...".format(n_folds))
        
        errors = []
        for l in reg_lambda:
            train_err, test_err = self.kFoldCV(n_folds, n_iter, l)
            errors.append(test_err)
            
        print("...Done!")
        
        return reg_lambda[np.argmin(errors)]