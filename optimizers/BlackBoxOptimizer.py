import numpy as np


class BlackBoxOptimizer(object):
    """Abstract class for generic black box (sequential) optimizer

    Args
    ----
    domain: list of dictionaries, mandatory
        List of dictionaries defining the domain of each variable. Each dictionary should have at least the fields
        "name" and "type" (with values either "continuous", "discrete" or "categorical").

    initial_x: numpy array, optional
        Array of points of size NxM, where N is the number of points and M is the number of parameters.

    initial_y: numpy array, mandatory if initial_X is provided
        Array of function evaluation at the N elements of initial_X

    Attributes
    ----------
    domain: list of dictionaries.
    X: numpy array. Points evaluated.
    Y: numpy array. Vector of objective values at points.
    x_best: Best point found in all iterations.
    y_best: Function value at best objective.
    times: List of times of bayesian optimization.
    """

    def __init__(self, domain, initial_x=None, initial_y=None):
        self.domain = domain
        self.X = initial_x
        self.Y = initial_y
        self.y_best = np.inf
        self.x_best = None
        self.times = []

        if self.Y is not None:
            i_best = np.argmax(self.Y)
            self.y_best = self.Y[i_best]
            self.x_best = self.X[i_best]

    def suggest(self):
        """Calculates new evaluation point.

        To be defined for each inheritor optimizer type.
        """
        raise NotImplementedError

    def report(self, x, y):
        """Reports new point and updates values of X, Y, x_best and y_best.

        Args:
            x : list or numpy array. Point evaluated.
            y : float or numpy array of one element. Function value.
        """
        if self.X is None:
            self.X = np.array(x).reshape([1, len(x)])
            self.Y = np.array(y).reshape([1, 1])
            self.x_best = x
            self.y_best = y
        else:
            self.X = np.vstack((self.X, np.array(x)))
            self.Y = np.vstack((self.Y, y))
            if y < self.y_best:
                self.x_best = x
                self.y_best = y

    def minimize(self, func, n_iter, extra_params={}, initial_x=None, input_as_list=False, verbose=True):
        """Minimizes an external function.

        Args:
            func: function handle. Function to be minimized.
            n_iter: int. Total number of function evaluations.
            extra_params: dict. Dictionary with any extra argument of func for each evaluation.
            initial_x: numpy array. Optional list of (presumably good) initial evaluation points.
            input_as_list: bool. Whether if each variable of the function is provided as an argument or on a single list.
            verbose: boolean. Verbosity.

        Returns:
            x_best: numpy array. Best point.
            y_best: float. Best function value
        """
        if not input_as_list:
            param_names = [var['name'] for var in self.domain]

        if initial_x is not None:
            param_names = [d['name'] for d in self.domain]
            for i, x in enumerate(initial_x):
                if verbose:
                    print('Initial point {}:'.format(i))
                    print('Variables: {}'.format(x))
                if input_as_list:
                    y = func(x, **extra_params)
                else:
                    y = func(**dict(zip(param_names, x) + extra_params.items()))
                self.report(x, y)
                if verbose:
                    print('Objective: {}'.format(y))
                    print('..................')

        for it in range(n_iter):
            x_new = self.suggest()
            if x_new is None:
                break
            if verbose:
                print('Iteration {}:'.format(it))
                print('Variables: {}'.format(x_new))
            if input_as_list:
                y_new = func(x_new, **extra_params)
            else:
                y_new = func(**dict(zip(param_names, x_new) + extra_params.items()))
            self.report(x_new, y_new)
            if verbose:
                print('Objective: {}'.format(y_new))
                print('..................')
        if verbose:
            print('Best value: {} \n'.format(self.y_best))

        return self.x_best, self.y_best

    def reset(self):
        """Resets attributes of the optimizer"""
        self.X = None
        self.Y = None
        self.x_best = None
        self.y_best = np.inf






