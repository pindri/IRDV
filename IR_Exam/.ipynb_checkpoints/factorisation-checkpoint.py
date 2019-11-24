import numpy as np
import numpy.linalg as lin
import pandas as pd


# Define functions to compute approximated ratings and error.

def predict(X, Y):
    return np.dot(X, Y.T)

def error(predicted_ratings, R, w0 = 1):
    """
    By default, the weight of the error on the unobserved items is the same as
    that on the obserbed ones.
    """
    obs_idx = np.where(R > 0)
    nobs_idx = np.where(R == 0)
    obs_error = sum( (R[obs_idx] - predicted_ratings[obs_idx]) ** 2 )
    nobs_error = sum( (R[nobs_idx] - predicted_ratings[nobs_idx]) ** 2 )
    return obs_error + w0 * nobs_error



#import scipy.optimize.nnls as nnls

def singlePassWALS(R, X, Y, C, reg_lambda):
    """
    A single pass of the Weighted Alternating Least Squares algorithm.
    As presented, it solves the linear systems of the form Awithout constraints.
    If desired, `nnls` can be used to compute a non-negative solution.
    """
    M = np.shape(X)[0]
    K = np.shape(X)[1]
    N = np.shape(Y)[0]
    
    for u in range(1, M):
        Cu = np.diag(C[u, :])
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
    Performs `n_iter` passes of the WALS algorithm, printing test and
    training errors.
    """
    for j in range(1, n_iter + 1):
        singlePassWALS(R_train, X, Y, C, reg_lambda)
        predicted_ratings = predict(X, Y)
        print( "Test error: {}".format(error(predicted_ratings, R_test)) )
        print( "Train error: {}".format(error(predicted_ratings, R_train)) )
        
        
def newUserSinglePassWALS(new_user, R, C, X, Y, reg_lambda):
    
    M = np.shape(X)[0]
    K = np.shape(X)[1]    
    
    # Perform user matrix optimisation.
    for u in range(1, M):
        Cu = np.diag(C[u, :])
        A = lin.multi_dot([Y.T, Cu, Y]) + reg_lambda * np.eye(K)
        b = lin.multi_dot([Y.T, Cu, R[u, :]])
        X_u = np.linalg.solve(A, b)
        #X_u = nnls(A, b)[0]
        
        X[u,] = X_u
        