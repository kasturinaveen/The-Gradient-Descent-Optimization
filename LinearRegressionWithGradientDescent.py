import numpy as np
import matplotlib.pyplot as plt


class Linreg:
    """
    This class provides methods for estimating parameters of
    simple, multiple and weighted multiple linear regressions.
    It also provides various statistical measures like standard-error
    of estimates, p-values etc. for model evaluation.
    """
    def __init__(self):
        self.data_train = None  # empty two-dimensional list
        self.target_train = None
        self.data_test = None
        self.learn_rate = None  # Learning rate for gradient descent algorithm
        self.theta = None  # regression parameters
        self.max_iter = None  # maximum iteration

    def fit(self, x_train, y_train, learningrate=0.05, maxiter=50):
        # feed the data
        self.data_train = x_train
        self.data_test = y_train

        # extract dimensions
        n_sample, n_feature = self.data_train.shape

        self.learn_rate = learningrate
        # self.theta = np.ones((self.data_train.shape[1] + 1))
        self.theta = np.zeros((n_feature + 1))  # parameter initialization, additional '1' for intercept
        self.max_iter = maxiter


