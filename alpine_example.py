"""Example to test with alpine1 function"""
import numpy as np
import pickle
from optimizers.BayesianOptimizer import BayesianOptimizer
from optimizers.RandomOptimizer import RandomOptimizer
from optimizers.GridSearchOptimizer import GridSearchOptimizer
import argparse
from utils import aux, functions


def main(optimizer, niter, nruns, ndim):
    """Main function. Optimizes alpine1 function.

    Args:
        optimizer: string. Type of optimizer ('bayesian', 'random' or 'gridsearch'.
        niter: int. Number of iterations of each run.
        nruns: int. Number of independent runs of the optimization.
        ndim: int. Dimensionality of alpine function.
    """

    func = functions.Alpine1(ndim)

    bounds = func.domain

    if optimizer == 'bayesian':
        black_box_optimizer = BayesianOptimizer
    elif optimizer == 'random':
        black_box_optimizer = RandomOptimizer
    elif optimizer == 'gridsearch':
        black_box_optimizer = GridSearchOptimizer

    Y = np.zeros([niter, nruns])
    X = []
    times = []

    for run in range(nruns):
        print('Run {}:'.format(run))
        print('-----------------------------------------')
        if optimizer != 'gridsearch':
            BO = black_box_optimizer(bounds)
        else:
            BO = black_box_optimizer(bounds, random_grid_elements=True)
        BO.minimize(func.f, niter, verbose=True, input_as_list=True)
        Y[:BO.Y.shape[0], run] = BO.Y[:, 0]
        X.append(BO.X)
        times.append(BO.times)

    try:
        np.savetxt("result_{}_alpine_{}.csv".format(optimizer, ndim), Y, delimiter=",")
    except:
        print('Error saving Y matrix')

    try:
        pickle.dump(X, open('Xdump_{}_alpine_{}.pkl'.format(optimizer, ndim), 'wb'))
    except:
        print('Error dumping X')

    if optimizer == 'bayesian':
        try:
            np.savetxt("times_{}_alpine_{}.csv".format(optimizer, ndim), np.array(times).T, delimiter=",")
        except:
            print('Error saving times matrix')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", help="Optimizer ('bayesian','random' or 'gridsearch')", default='bayesian')
    parser.add_argument("--niter", help="Number of optimizer iterations on each experiment", default=72)
    parser.add_argument("--nruns", help="Number of independent black box optimizations to run", default=5)
    parser.add_argument("--ndim", help="Dimensions of alpine function", default=2)

    args = parser.parse_args()

    print('Launching {} independent runs on {}-dimensional alpine function with {} iterations each over {} optimizer \n'
          .format(args.nruns, args.ndim, args.niter, args.optimizer))

    main(args.optimizer, int(args.niter), int(args.nruns), int(args.ndim))
