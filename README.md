# Description of the files

+ **batch_gradient_descent.py**: Separate module for running batch gradient descent.
+ **stochastic_gradient_descent.py**: Separate module for running stochastic gradient descent.
+ **GradientDescentOptimizer.py**: A Python class providing methods for both *batch* and *stochastic* gradient descent algorithms.

# When to consider what ?

If the dataset is small or moderate one should consider batch gradient descent as it gives more accurate approximations to the parameters. But in case of large dataset, running batch gradient descent is computationally too expensive as it scans through all the training examples or records in the dataset for performing a single update to the parameters. On the otherhand, stochastic gradient descent is designed to perform an update to the parameters based on a single traing example or record. More explicitly, if the dataset consists of 10000 rows or records, stochastic gradient descent updates the parameters 10000 times for each step to the minimization of the cost function and the result is the need of less number of step for convergence.

# Data Simulation

If the modules are called directly, they will simulate random dataset with the specified set of parameters based on a given linear model. The model is in the initial description in each module. Then the gradient descent (batch or stochastic) is applied on this dummy dataset to evaluate how the algorithm is performing. 

If one wants to use it for their own dataset, please append the following in the start of the script:
```
from GradientDescentOptimizer import batch_gradient_descent
#or,
from GradientDescentOptimizer import stochastic_gradient_descent

```
But, note that these codes are created only for learning and these are designed to behave only for linear/multiple linear regression model.  

**batch_gradient_descent.py** as well as **stochastic_gradient_descent.py** as set to perform estimation for the same set of parameters. One can run the two modules separately to judge the performances.

# References: 

+ [Stochastic Gradient Descent - by Andrew NG](https://www.youtube.com/watch?v=UfNU3Vhv5CA)

# Future plan
+ a jupyter notebook explaining the theory of gradient descent (almost ready).
+ a separate method for mini-batch gradient descent algorithm.
+ a separate method for cross-entropy cost function, which is very popular for Neural Network models.

these changes will only be reflected in **GradientDescentOptimizer.py**.
