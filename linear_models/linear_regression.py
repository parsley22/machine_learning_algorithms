import numpy as np

class linear_regression:
    """
    Builds OLS models, with option to include a ridge penalty
    
    ----------------------------------------------------------
    add_bias:
        Adds a bias vector of ones to the design matrix X
        
    fit:
        calculates the model parameters; option to include a ridge penalisation
        
    predict:
        predicts a real values output vector y
        
    """
    def __init__(self):
        self.beta = None

    def add_bias(self, X):
        """
        
        Concats a vector of ones to the design matrix
        
        X : numpy array of shape [n_samples, n_features]
        
        """
        bias_vec = np.ones(X.shape[0]).reshape(-1, 1)
        return np.concatenate([X, bias_vec], axis=1)

    def fit(self, X, y, lambda_=0, bias=True):
        
        """
        Fits the model to the data
        
        X : numpy array of shape [n_samples, n_features]
        
        y : numpy array of shape [n_samples]
        
        lambda : R
            Ridge regression coefficient
        
        bias : binary
            Choose whether or not to include a bias vector to the design matrix
        """
        if bias == True:
            X = self.add_bias(X)

        penalisation = lambda_ * np.eye(X.shape[1])

        beta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X) + penalisation), X.T), y)
        self.beta = beta

    def predict(self, X):
        """
        X : numpy array of shape [n_samples, n_features]
        
        """
        assert self.beta is not None
        if len(self.beta) == X.shape[1] + 1:
            X = self.add_bias(X)
        return np.dot(self.beta, X.T)
