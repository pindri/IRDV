import numpy as np
import pandas as pd
import numpy.linalg as lin

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
        
    def predictionError(self):
        predicted_ratings = factorisation.predict(self.X, self.Y)
        prediction_error = factorisation.error(predicted_ratings, self.R)
        print("Prediction error: {}".format(prediction_error))
        
    
    def performFactorisation(self, reg_lambda, n_iter):
        factorisation.WALS(self.R, self.R, self.X, self.Y, self.C, reg_lambda, n_iter)
        
    
    def answerQuery(self, user_id):
        """
        Produces a dataframe containing the ranked recommendations for the
        unobserved items using the predicted ratings. 
        The average rating for the recommended item is displayed as well.
        """
        pred = np.matrix.round(factorisation.predict(self.X, self.Y), 2)[user_id]

        # Unseen movies.
        idx = np.where(self.R[user_id] == 0)[0]
        movie_pred = list(zip(idx, pred[idx]))

        # Build prediction dataframe.
        recom_df = pd.DataFrame(movie_pred,
                                columns = ['MovieID', 'Prediction'])
        recom_df = pd.merge(recom_df, self.movies_df,
                            on = "MovieID", how = "inner")
        recom_df = recom_df.sort_values(by = 'Prediction', ascending = False)

        # Add comparison with average ratings.
        avg_rat = self.ratings_df.groupby('MovieID').mean()
        recom_df = pd.merge(recom_df, avg_rat,
                            on = "MovieID", how = "inner")
        recom_df.drop(['UserID'], inplace = True, axis = 1)
        recom_df.round({'Rating': 1})
        recom_df.rename(columns = {'Rating':'AVG_Rating'}, inplace = True)

        return recom_df
    
    def getUserMovies(self, user_id):
        return self.R_df[self.R_df['UserID'] == user_id]
    
    
    # Suggesting similar items.

    def cosineSimilarity(self, d_1, d_2):
        len_1 = lin.norm(d_1)
        len_2 = lin.norm(d_2)
        if len_1 == 0 or len_2 == 0:
            return -1
        return np.dot(d_1, d_2) / (len_1 * len_2)

    def similarItems(self, movie_id):
        # Y is the item embedding
        d_1 = self.Y[movie_id]
        similarity = [self.cosineSimilarity(self.Y[movie_id], self.Y[i]) 
                      for i in range(0, np.shape(self.Y)[0])]
        return similarity
    
    def suggestSimilar(self, movie_id):
        similarities = pd.DataFrame(self.similarItems(movie_id),
                                    columns = ["Similarity"])
        similarities_df = pd.concat([self.movies_df, similarities], axis = 1)
        return similarities_df.sort_values(by = 'Similarity',
                                           ascending = False).head(10)