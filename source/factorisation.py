# Set of functions used to perform and test the Weighted Alternating
# Least Squares algorithm.

import numpy as np
import numpy.linalg as lin
import pandas as pd


# Define functions to compute approximated ratings and error.

def predict(X, Y):
    """
    Computes the dot product between X and Y.T, producing a prediction of
    the ratings.
    """
    
    return np.dot(X, Y.T)


def MAE(predicted_ratings, R, w0 = 1):
    """
    By default, the weight of the error on the unobserved items is one.
    """
    
    obs_idx = np.where(R > 0)
    n_obs = np.count_nonzero(R)
    nobs_idx = np.where(R == 0)
    n_nobs = np.count_nonzero(R == 0)
    obs_error = sum(abs(R[obs_idx] - predicted_ratings[obs_idx])) / n_obs
    nobs_error = sum(abs(R[nobs_idx] - predicted_ratings[nobs_idx])) / n_nobs

    return obs_error + w0 * nobs_error


#import scipy.optimize.nnls as nnls

def singlePassWALS(R, X, Y, C, reg_lambda):
    """
    A single pass of the Weighted Alternating Least Squares algorithm.
    As presented, it solves the linear systems of the form Ax = b without
    constraints.
    If desired, `nnls` can be used to compute a non-negative solution.
    """
    
    M = np.shape(X)[0]
    K = np.shape(X)[1]
    N = np.shape(Y)[0]
    
    for u in range(1, M):
        Cu = np.diag(C[u,:])
        A = lin.multi_dot([Y.T, Cu, Y]) + reg_lambda * np.eye(K)
        b = lin.multi_dot([Y.T, Cu, R[u, :]])
        X_u = lin.solve(A, b)
        #X_u = nnls(A, b)[0]
        
        X[u,] = X_u
        
    for i in range(1, N):
        Ci = np.diag(C[:,i])
        A = lin.multi_dot([X.T, Ci, X]) + reg_lambda * np.eye(K)
        b = lin.multi_dot([X.T, Ci, R[:, i]])
        Y_i = lin.solve(A, b)
        #Y_i = nnls(A, b)[0]
        
        Y[i,] = Y_i
        
        
def WALS(R_train, R_test, X, Y, C, reg_lambda, n_iter):
    """
    Performs `n_iter` passes of the WALS algorithm.
    It returns two lists, one for training and one for test errors.
    """
    
    print("Performing WALS algoritm...")
    
    train_error = []
    test_error = []
    for j in range(n_iter):
        singlePassWALS(R_train, X, Y, C, reg_lambda)
        predicted_ratings = predict(X, Y)
        train_error.append(MAE(predicted_ratings, R_train))
        test_error.append(MAE(predicted_ratings, R_test))
        #print("Train error: {}".format(train_error[j]))
        #print("Test error: {}".format(test_error[j]))
    
    print("...Done!")
    
    return train_error, test_error
        
        
def newUserSinglePassWALS(new_user, R, C, X, Y, reg_lambda):
    """
    A single pass of the Weighted Alternating Least Squares algorithm
    for the addition of a new user. Assumes the item-embedding matrix
    to be already optimised and proceeds to minimise the error for the
    newly added user row.
    As presented, it solves the linear systems of the form Ax = b without
    constraints.
    If desired, `nnls` can be used to compute a non-negative solution.
    """
    
    M = np.shape(X)[0]
    K = np.shape(X)[1]    
    
    # Perform new_user optimisation.
    u = M - 1 # Last user.
    Cu = np.diag(C[u, :])
    A = lin.multi_dot([Y.T, Cu, Y]) + reg_lambda * np.eye(K)
    b = lin.multi_dot([Y.T, Cu, R[u, :]])
    X_u = np.linalg.solve(A, b)
    #X_u = nnls(A, b)[0]

    X[u,] = X_u