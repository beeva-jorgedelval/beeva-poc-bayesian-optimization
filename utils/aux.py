"""Auxiliary functions"""
import numpy as np
import random


def adapt_to_domain(x, domain):
    """Adapt the types of the variables to the domain.

    Args:
        x: list or numpy array. Input point.
        domain: list of dicts. As in BlackBoxOptimizer.

    Returns:
        List with adapted point.
    """
    x2 = list(x)
    for i, xi in enumerate(x2):
        if domain[i]['type'] == 'discrete' and isinstance(domain[i]['domain'][0], int):
            x2[i] = int(xi)
    return x2


def sample(domain):
    """Sample a random point of the domain with uniform distribution.

    Args:
        domain: list of dicts. As in BlackBoxOptimizer.

    Returns:
        List with sampled point.
    """
    out = []
    for var in domain:
        if var['type'] == 'continuous':
            out.append(np.random.uniform(low=var['domain'][0], high=var['domain'][1]))
        elif var['type'] == 'discrete' or var['type'] == 'categorical':
            out.append(random.choice(var['domain']))
    return out


def random_intervals(nit, ndim):
    """Define a random set of grid point numbers for each variable whose product (total number of grid points)
     is not much higher than the total number of iterations.

    Args:
        nit: int. Total number of iterations of grid search algorith. We want more total gridpoints than iterations.
        ndim: int. Dimensionality of the domain.

    Returns:
        List with number of grid points per variable, i.e., [2,3,3]. It is not important if len of list is less than
        ndim; the rest of grid points are assumed to be 1s.
    """
    res = []
    s_max = nit  # The number of grid points of next variable that would make the product approximately equal to nit.
    for d in range(ndim-1):
        s = np.random.randint(s_max)+1  # Sample new number of grid points for that variable.
        res.append(s)
        s_max = int(nit/np.prod(res)+1)  # Update new upper bound.
        if s_max == 1:
            break
    if s_max != 1:
        res.append(s_max)  # If there is still margin for more grid points, add all to last variable.
    return res


def random_grid(domain, niter):
    """Randomly assigns number of grid points of interval to each variable.

    Args:
        domain: list of dicts. As in BlackBoxOptimizer.
        niter: int. Total number of iterations (approximately the number of desired grid points).
    """
    intervals = random_intervals(niter, len(domain))
    grid = []
    choices = intervals + [1]*(len(domain)-len(intervals))
    for var in domain:
        var_new = var.copy()
        range = (var['domain'][0], var['domain'][-1])
        choice = random.choice(choices)

        if choice == 1 and var['type'] == 'continuous':
            var_new['domain'] = [(range[0] + range[1]) / 2.]
        else:
            var_new['domain'] = np.linspace(range[0], range[1], choice).tolist()

        choices.remove(choice)
        grid.append(var_new)
    return grid
