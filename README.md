# Description of the files

+ **batch_gradient_descent.py**: Separate module for running batch gradient descent.
+ **stochastic_gradient_descent.py**: Separate module for running stochastic gradient descent.
+ **GradientDescentOptimizer.py**: A Python class providing methods for both *batch* and *stochastic* gradient descent algorithms.

# When to consider what ?

If the dataset is small or moderate one should consider batch gradient descent as it gives more accurate approximations to the parameters. But in case of large dataset, running batch gradient descent is computationally too expensive as it scans through all the training examples or records in the dataset for performing a single update to the parameters. On the otherhand, stochastic gradient descent is designed to perform an update to the parameters based on a single traing example or record. More explicitly, if the dataset consists of 10000 rows or records, stochastic gradient descent updates the parameters 10000 times and the result is the need of less number of step for convergence.

For example, **batch_gradient_descent.py** as well as **stochastic_gradient_descent.py** as set to perform estimation for the same set of parameters. One can run the two modules separately to judge the performances.

# References: 

+ [Stochastic Gradient Descent - by Andrew NG](https://www.youtube.com/watch?v=UfNU3Vhv5CA)
