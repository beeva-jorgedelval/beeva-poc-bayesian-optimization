from BlackBoxOptimizer import BlackBoxOptimizer
from utils.aux import adapt_to_domain, sample
import time
import GPyOpt


class BayesianOptimizer(BlackBoxOptimizer):
    """Class for bayesian optimizer.

    Args:
        domain: list of dicts. Domain of the problem (as in BlackBoxOptimizer).
        initial_x: numpy array. Initial points.
        initial_y: numpy array. Initial function evaluation at points.

    """

    def __init__(self, domain, initial_x=None, initial_y=None):
        BlackBoxOptimizer.__init__(self, domain, initial_x, initial_y)

    def suggest(self):
        """Suggest new evaluation point using GPyOpt as Bayesian Optimization module.

        Returns:
            Point suggested for evaluation (list).
        """
        if self.X is None:
            return sample(self.domain)  # Samples first point randomly.
        else:
            model = GPyOpt.methods.BayesianOptimization(f=None, domain=self.domain, initial_design_numdata=0,
                                                        X=self.X, Y=self.Y, de_duplication=True)
            start = time.time()
            x_new = model.suggest_next_locations()
            end = time.time()
            self.times.append(end-start)
            return adapt_to_domain(x_new.flatten().tolist(), self.domain)
