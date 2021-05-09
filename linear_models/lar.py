import numpy as np

class least_angle_regression:
    def __init__(self, step=0.01):
        self.step = step
        
        self.residual = None
        self.beta = None
        self.active_set = []
        self.corrcoefs = None
        
    def initialise(self, X, y):
        self.residual = y - np.mean(y)
        self.beta = np.zeros(X.shape[1])
        self.corrcoefs = np.abs(
                np.corrcoef(np.concatenate([X, y.reshape(-1, 1)], axis=1).T)[:-1, -1]
            )
        
    def move_coef_toward_ls(self, j):
        self.beta[j] 
        

    def update_corr(self):
        j = np.argmax(self.corrcoefs)
        self.corrcoefs[j] = 0
        return j
        
    def fit(self, X, y):
        self.initialise(X, y)

        while True:
            j = self.update_corr(X)
            self.active_set.append(X[:, j])
            self.beta[j] = np.dot(X[:,j], self.residual)

            # move beta towards ls coeff
            while True:
                # check for competitor
                # update beta
                self.move_coef_towar_ls(j)
            return



