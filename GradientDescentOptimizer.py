# -*- coding: utf-8 -*-

"""
@author: koushik

version: 1.0
"""
import numpy as np


class GradientDescentOptimizer():
    """
    This is GradientDescentOptimizer class, providing two
    methods for batch and stochastic gradient descent
    respectively. Cost function is considered as squared
    error cost and as of now methods are only pointing to
    simple linear or multiple linear regression problem.
    Future releases may come with separate methods for
    defining several cost functions.

    Coming in future release: Mini-batch Stochastic Gradient Descent
    """

    def __init__(self):
        pass

    def batch_gradient_descent(self, init, learning_rate, data_train,
                               target_train, precision):
        """
        init: initial set of parameters
        learning_rate: learning rate for BGD to converge
        data_train: training data (predictors)
        target_train: training data (responses or the targets)
        return: minimum cost and optimized parameters
        """
        N = data_train.shape[0]  # no. of training example available
        step = 1  # no. of step required to converge

        def _gradient(param_vector):
            """
            type: private function
            param_vector: vector of parameters
            return: gradient vector evaluated at
                    the given set of parameters
            """

            # prediction error
            error = (target_train - np.dot(data_train, param_vector))

            # gradient vector
            g = (-1 / N) * np.dot(np.transpose(data_train), error)

            return g

        def _cost_function(param_vector):
            """
            type: private function
            param: vector of parameters
            return: cost function evaluated at
                    the parameter vector supplied
            """

            # predicted values of the targets for given parameters
            target_pred = np.dot(
                data_train, param_vector)
            cost = np.sum(np.square(target_train - target_pred)) / \
                (2 * N)  # cost due to prediction error
            return cost

        converged = False
        while not converged:
            cost_current = _cost_function(init)

            # learning rule
            param_new = init - (learning_rate * _gradient(init))

            # updated cost
            cost_new = _cost_function(param_new)

            # absolute value of the cost difference
            cost_diff = abs(cost_new - cost_current)

            # set to 0.0000000001
            converged = bool(cost_diff < precision)
            if step % 20 == 0:
                # prints the summary after each 50th step
                print("Step: {}\tUpdated cost: {: 0.4f}\t \
                Cost difference: {}\tUpdated parameters: {}".format(
                    step, _cost_function(param_new), cost_diff, param_new))
            init = param_new
            step = step + 1

        print("\Batch Gradient Descent Optimizer Converged!\n")
        print("Minimum cost:{:0.4f}\tFinal Parameter Estimates:{}".format(
            _cost_function(init), init))
        return _cost_function(init), init

    def stochastic_gradient_descent(self, init, learning_rate,
                                    data_train, target_train, precision):
        """
        init: initial set of parameters
        learning_rate: learning rate for BGD to converge
        data_train: training data (predictors)
        target_train: training data (responses or the targets)
        return: minimum cost and optimized parameters
        """
        N = data_train.shape[0]  # no. of training example available
        step = 1  # no. of step required to converge

        def _gradient(param_vector, index):
            """
            type: private function
            param_vector: vector of parameters
            index: position of training example, e.g. 100th example
             means index=100
            return: gradient vector evaluated at
                    the given set of parameters
            """

            # prediction error
            error = (target_train[index] - np.dot(data_train[index, :],
                                                  param_vector))

            # gradient vector
            g = (-1) * np.transpose(data_train[index, :] * error)
            return g

        def _cost_function(param_vector):
            """
            type: private function
            param: vector of parameters
            return: cost function evaluated at
                    the parameter vector supplied
            """

            # predicted values of the targets for given parameters
            target_pred = np.dot(
                data_train, param_vector)
            cost = np.sum(np.square(target_train - target_pred)) / \
                (2 * N)  # cost due to prediction error
            return cost

        converged = False
        while not converged:
            cost_current = _cost_function(init)

            # steps for stochastic gradient descent
            for i in range(N):
                param_upd = init - (learning_rate * _gradient(init, i)
                                    )  # updated vector of parameter
                if i < (N - 1):
                    init = param_upd
                else:
                    param_new = param_upd

            # updated cost
            cost_new = _cost_function(param_new)

            # absolute value of the cost difference
            cost_diff = abs(cost_new - cost_current)
            converged = bool(cost_diff < precision)
            print("Step:{}\tUpdated cost:{:0.4f}\t \
            Cost difference:{}\tUpdated parameters:{}".format(
                step, _cost_function(param_new), cost_diff, param_new))
            init = param_new
            step = step + 1

        print("\nStochastic Gradient Descent Converged!\n")
        print("Minimum cost:{:0.4f}\tFinal Parameter Estimates:{}".format(
            _cost_function(init), init))
        return _cost_function(init), init


def main():
    # Simulation of toy data
    # np.random.seed(0) # set the seed
    X = np.column_stack((np.ones((1000000, 1)), np.random.rand(1000000, 3)))
    error = np.random.randn(1000000)
    y = np.dot(X, np.array([5, 2, 0.85, 3.5]), out=None) + error

    p = np.random.randint(1, 8, size=4)  # initial values of the parameters
    data_train = X
    target_train = y
    learning_rate = 0.00003
    precision = 0.000001

    # initialize the GradientDescentOptimizer class
    gd = GradientDescentOptimizer()

    # evaluate stochastic_gradient_descent()
    try:
        cost, param = gd.stochastic_gradient_descent(
            p, learning_rate, data_train, target_train, precision)
    except TypeError:
        print("Please specify the parameters carefully!")


if __name__ == "__main__":
    main()
