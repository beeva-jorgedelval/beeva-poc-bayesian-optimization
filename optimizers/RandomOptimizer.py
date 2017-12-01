from BlackBoxOptimizer import BlackBoxOptimizer
import utils.aux as aux


class RandomOptimizer(BlackBoxOptimizer):
    """Class for random search.

    Args:
        domain: List of dicts. Domain of the problem (as in BlackBoxOptimizer).
    """

    def __init__(self, domain, initial_x=None, initial_y=None):
        BlackBoxOptimizer.__init__(self, domain, initial_x, initial_y)

    def suggest(self):
        """Suggest new point by sampling randomly the domain.

        Returns:
            Point suggested for evaluation (list).
        """
        return aux.sample(self.domain)
