from BlackBoxOptimizer import BlackBoxOptimizer
from itertools import product
from utils.aux import random_grid, adapt_to_domain

class GridSearchOptimizer(BlackBoxOptimizer):
    """Class for grid search optimizer.

    Args
    ----
    domain : List of dicts. Domain of the problem (as in BlackBoxOptimizer). If the user wants to use a custom grid,
    all variables should be of type "discrete" and the "domain" attribute is the list of grid points for that variable.

    random_grid_elements: bool. Whether to use a random grid or not.
    gridpoints: int. (Approximate) number of desired grid points if random_grid_elements is True.
    """

    def __init__(self, domain, random_grid_elements=True, gridpoints=72, initial_x=None, initial_y=None):
        BlackBoxOptimizer.__init__(self, domain, initial_x=initial_x, initial_y=initial_y)
        if random_grid_elements is not None:
            self.grid_generator = product(*[var['domain'] for var in random_grid(domain, gridpoints)])
        else:
            self.grid_generator = product(*[var['domain'] for var in domain])

    def suggest(self):
        """Suggests new point by yielding the next grid point until there are no points left.

        Returns:
            Point suggested for evaluation (list).
        """
        try:
            return adapt_to_domain(next(self.grid_generator), self.domain)
        except:
            print('Grid search is over')
            return None
