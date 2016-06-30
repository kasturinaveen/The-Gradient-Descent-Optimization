import numpy as np
from sklearn.datasets import load_diabetes


class Linreg:
    """
    This class provides methods for estimating parameters of
    simple, multiple and weighted multiple linear regressions.
    It also provides various statistical measures like standard-error
    of estimates, p-values etc. for model evaluation.
    """
    def __init__(self):
        self._data_train = None  # empty two-dimensional list
        self._target_train = None
        self._data_test = None
        # self._learn_rate = None  # Learning rate for gradient descent algorithm
        self._theta = None  # regression parameters
        # self._maxiter = None  # maximum iteration

    def fit(self, x_train, y_train, learningrate=0.1, maxiter=200):
        # feed the data
        self._data_train = x_train
        self._target_train = y_train
        # self._learn_rate = learningrate
        # self._maxiter = maxiter

        # extract dimensions
        n_sample, n_feature = np.shape(self._data_train)

        # parameter initialization
        self._theta = np.ones(n_feature)

        for i in range(maxiter):
            residual = (self._target_train - np.dot(self._data_train, self._theta))
            cost_func = np.dot(residual.T, residual)/(2 * n_sample)
            gradient = (1/n_sample) * np.dot(self._target_train.T, residual)

            # update theta
            self._theta = self._theta - (learningrate * gradient)

        return self._theta

diab = load_diabetes()
data = diab.data
target = diab.target

lm = Linreg()
mlfit = lm.fit(data, target)

print(mlfit)

        


