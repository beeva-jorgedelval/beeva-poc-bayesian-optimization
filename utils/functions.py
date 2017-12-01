"""Different test functions"""
import numpy as np


class Alpine1(object):
    """Alpine1 n-dimensional function is a non-convex, non-differentiable multidimensional function with minimum at
    x_i = 0 for all i in [1, ndim]. We define it for x_i in (-5,5) for all i.

    Args:
        ndim: int. Number of dimensions

    Returns:
        Numpy 1x1 array with function value.
    """

    def __init__(self, ndim=2):
        self.ndim = ndim
        self.domain = [{'name': i, 'type': 'continuous', 'domain': (-5, 5)} for i in range(ndim)]

    def f(self, x):
        X = np.array(x).reshape(self.ndim)
        fval = np.abs(X*np.sin(X)-0.1*X).sum()
        return fval.reshape([1, 1])
