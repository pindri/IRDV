import numpy as np
import pandas as pd
import numpy.linalg as lin
from random import sample, choice
from functools import reduce

import data_preparation # Load dataset and build required matrices.
import factorisation # WALS factorisation.


class recommenderSystem():
    
    """
    Implements a recommender system. It is constructed from a movies
    dataframe and ratings dataframes (for both training and test).
    The number of latent factors 'k' can be specified.
    
    It includes methods to perform WALS factorisation and provide
    recommendations.
    
    It includes methods to compute the k-fold CV error and
    precision/recall/AP. Theese can be used to assess the recommander
    system performance.
    
    For each item, using cosine similarity, similar items can be
    recommended.
    
    Additionally, the class can randomly generate new users and provide
    recommendations for them.
    """
    
    def __init__(self, movies_df, ratings_df, ratings_df_test, k = 100):
        self.movies_df = movies_df
        self.ratings_df = ratings_df
        self.ratings_df_test = ratings_df_test
        self.R_df, self.R = data_preparation.buildR(movies_df, ratings_df)
        self.R_df_test, self.R_test = data_preparation.buildR(movies_df, 
                                                              ratings_df_test,
                                                              is_test = True)
        
        self.C = data_preparation.buildWeightMatrix(self.R)
        
        self.K = k
        self.X = np.random.rand(np.shape(self.R)[0], self.K)
        self.Y = np.random.rand(np.shape(self.R)[1], self.K)
        
        
    def getUserMovies(self, user_id):
        
        return self.R_df[self.R_df['UserID'] == user_id]
    
    
    def predictionError(self):
        """
        Computes the MAE for the current values of X and Y.
        """
        
        predicted_ratings = factorisation.predict(self.X, self.Y)
        prediction_error = factorisation.MAE(predicted_ratings, self.R) 
        
        return prediction_error
    
    
    def performFactorisation(self, reg_lambda, n_iter):
        """
        Performs n_iter iterations of the WALS algorithm.
        """
        
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
        
        pred = np.matrix.round(factorisation.predict(self.X,
                                                     self.Y), 2)[user_id]

        # Unseen movies.
        idx = np.where(self.R[user_id] == 0)[0]
        movie_pred = list(zip(idx, pred[idx]))

        # Build predictions and avg ratings dataframes.
        predictions_df = pd.DataFrame(movie_pred,
                                      columns = ['MovieID', 'Prediction'])
        avg_rat = self.ratings_df.groupby('MovieID').mean()
        
        dfs = [predictions_df, self.movies_df, avg_rat]
        
        recom_df = reduce(lambda left, right: 
                          pd.merge(left, right, on = 'MovieID'), dfs)
        
        recom_df.drop(['UserID'], inplace = True, axis = 1)
        recom_df.round({'Rating': 1})
        recom_df.rename(columns = {'Rating':'AVG_Rating'}, inplace = True)

        return recom_df.sort_values(by = 'Prediction', ascending = False)
    
    
    def mostPopular(self):
        """
        Produces a dataframe containing the ranked most popular items.
        """
        
        # movie title genre avg rating
        movie_count_df = (self.ratings_df.groupby('MovieID').size()
                          .reset_index(name = 'Counts'))
        avg_rat = self.ratings_df.groupby('MovieID').mean()
        
        dfs = [self.movies_df, avg_rat, movie_count_df]
        
        recom_df = reduce(lambda left, right: 
                          pd.merge(left, right, on = 'MovieID'), dfs)
        
        recom_df.drop(['UserID'], inplace = True, axis = 1)
        recom_df.rename(columns = {'Rating':'AVG_Rating'}, inplace = True)
        
        return recom_df.sort_values(by = 'Counts', ascending = False)
    
    
    def answerQuery(self, user_id):
        """
        Returns a dataframe of ranked recommendations for user_id.
        If user_id has rated less than 10 movies, the most popular
        movies will be returned.
        """
        
        n_seen = len(np.where(self.R[user_id] != 0)[0])
        
        if n_seen >= 10:
            recom_df = self.answerQueryAux(user_id)
        else:
            print("Too few movies! Most poular movies will be suggested.")
            recom_df = self.mostPopular()
            
        return recom_df
    
    
    #===============================#
    #== Suggesting similar items. ==#
    #===============================#
    
    def cosineSimilarity(self, d_1, d_2):
        """
        Computes the cosine similarity between two arrays.
        """
        
        len_1 = lin.norm(d_1)
        len_2 = lin.norm(d_2)
        if len_1 == 0 or len_2 == 0:
            return -1
        
        return np.dot(d_1, d_2) / (len_1 * len_2)

    
    def similarItems(self, movie_id):
        """
        Computes the similarity beween the current item (movie_id) and
        all other items.
        """
        
        # Y is the item embedding
        d_1 = self.Y[movie_id]
        similarity = [self.cosineSimilarity(self.Y[movie_id], self.Y[i]) 
                      for i in range(0, np.shape(self.Y)[0])]
        
        return similarity
    
    
    def suggestSimilar(self, movie_id):
        """
        Given a movie_id, it retuns a ranked dataframe of similar items.
        """
        
        similarities = pd.DataFrame(self.similarItems(movie_id),
                                    columns = ['Similarity'])
        similarities_df = pd.concat([self.movies_df, similarities], axis = 1)
        
        return similarities_df.sort_values(by = 'Similarity',
                                           ascending = False).head(10)
   

    #===============================#
    #== New users recommendations ==#
    #===============================#

    def generateNewUser(self, n_movies):
        """
        Randomly generates a new user who has rated n_movies.
        It returns the user array and a new_user_id.
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
        Adds the new_user updating the ratings and users matrices.
        It does not recompute the item matrix, which is assumed not to
        significantly change when a few users are added
        
        It performs a single-step WALS for the new user. If many users
        are added, a complete WALS step should be performed to update
        the item matrix as well.
        """
        
        self.R, self.C, self.X = data_preparation.updateMatrices(new_user,
                                                                 self.R, 
                                                                 self.C,
                                                                 self.X)
        self.R_df = data_preparation.updateDataFrame(new_user, self.R_df,
                                                     self.movies_df)
        factorisation.newUserSinglePassWALS(new_user, self.R, self.C, self.X,
                                            self.Y, reg_lambda)
        
        
    def computeFolds(self, n_folds):
        """
        Subdivides the R matrix in n_folds folds.
        It uses a mask to randomly choose (user, item) pairs.
        It returns an array containing the folds.
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

    
    #=======================#
    #== Cross validation. ==#
    #=======================#
        
    def kFoldCV(self, n_folds, n_iter, reg_lambda):
        """
        Computes the k-fold CV error using n_folds, reg_lambda 
        regression coefficient and n_iter iterations of the WALS
        algorithm.
        """
        
        k_folds = self.computeFolds(n_folds)
        # Initialising training and test errors across iterations.
        k_train_err = [0 for _ in range(n_iter)]
        k_test_err = [0 for _ in range(n_iter)]
        
        for i in range(n_folds):
            R_test = k_folds[i]
            R_train = sum(k_folds) - k_folds[i]
            train_err, test_err = factorisation.WALS(R_train, R_test, self.X,
                                                     self.Y, self.C,
                                                     reg_lambda, n_iter)

            # Updating cumulative train/test errors from WALS.
            k_train_err = [sum(x) for x in zip(k_train_err, train_err)]
            k_test_err = [sum(x) for x in zip(k_test_err, test_err)]
            
            
        # Returning average error across folds.
        return ([x / n_folds for x in k_train_err],
                [x / n_folds for x in k_test_err])
    
    
    def bestLambdaCV(self, n_folds, n_iter, reg_lambda):
        """
        Computes the k-fold CV error (using n_folds folds and n_iter
        iterations of the WALS algorithm) for each of the reg_lambda
        values. Return the regression coefficient associated with
        the smallest CV error.
        
        NOTE: requires reg_lambda to be a list
        """
        
        print("Performing {} fold CV...".format(n_folds))
        
        lambda_errors = []
        error_history = []
        for l in reg_lambda:
            train_err, test_err = self.kFoldCV(n_folds, n_iter, l)
            # Appending the last iteration kfold error.
            lambda_errors.append(test_err[-1])
            # Updating error history.
            error_history.append([train_err, test_err])
            
        print("...Done!")
        
        return reg_lambda[np.argmin(lambda_errors)], error_history
    
    
    #=====================#
    #== Error measures. ==#
    #=====================#
        
    def precision(self, user_id, n_recom):
        """
        Computes the number of relevant recommendations for user_id
        when n_recom recommendations are suggested.
        """
        
        recom = self.answerQuery(user_id).head(n_recom)
        actual = self.R_df_test[self.R_df_test['UserID'] == user_id]
        # Number of relevant recommendations.
        prec = len(pd.merge(recom, 
                            actual, on = 'MovieID')) / n_recom
        
        return prec

    
    def meanPrecision(self, n_recom):
        """
        Computes the mean precision across all users when n_recom
        recommendations are suggested.
        """
        
        prec = 0.0
        n_users = self.R_df['UserID'].nunique()
        for u in range(n_users):
            prec += self.precision(u, n_recom)
        
        return prec / n_users
    
    
    def recall(self, user_id, n_recom):
        """
        Computes the recall for user_id when n_recom recommendations
        are suggested.
        """
        
        recom = self.answerQuery(user_id).head(n_recom)
        actual = self.R_df_test[self.R_df_test['UserID'] == user_id]
        # Number of relevant reccomendations.
        recall = len(pd.merge(recom, 
                             actual, on = 'MovieID')) / len(actual)
        
        return recall
        
        
    def meanRecall(self, n_recom):
        """
        Computes the mean recall across all users when n_recom
        recommendations are suggested.
        """
        
        recall = 0.0
        n_users = self.R_df['UserID'].nunique()
        for u in range(n_users):
            recall += self.recall(u, n_recom)
        
        return recall / n_users
