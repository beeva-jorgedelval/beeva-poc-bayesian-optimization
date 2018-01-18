"""Trains a simple CNN on CIFAR-10 dataset and outputs (minus) the test accuracy"""
import argparse
import pickle
import numpy as np
from optimizers.BayesianOptimizer import BayesianOptimizer
from optimizers.GridSearchOptimizer import GridSearchOptimizer
from optimizers.RandomOptimizer import RandomOptimizer
from problems import cifar_cnn as cnn


def main(optimizer, niter, nruns, epochs):
    """Main function. Optimizes hyperparameters of CNN on CIFAR.

    Args:
        optimizer: string. Type of optimizer ('bayesian', 'random' or 'gridsearch'.
        niter: int. Number of iterations of each run.
        nruns: int. Number of independent runs of the optimization.
        epochs: int. Number of epoch to train the CNN.
    """
    domain = [{'name': 'conv_1_size', 'type': 'discrete', 'domain': range(3,4)},
              {'name': 'conv_1_filters', 'type': 'discrete', 'domain': range(32, 64)},
              {'name': 'conv_2_size', 'type': 'discrete', 'domain': range(3, 4)},
              {'name': 'conv_2_filters', 'type': 'discrete', 'domain': range(32, 64)},
              {'name': 'dropout_p', 'type': 'continuous', 'domain': (0.0, 0.5)},
              {'name': 'batch_size', 'type': 'discrete', 'domain': range(32, 64)},
              {'name': 'log_learning_rate', 'type': 'continuous', 'domain': (-6.9, -3.0)},
              {'name': 'log_decay', 'type': 'continuous', 'domain': (-20, -18)},
              {'name': 'log_rho', 'type': 'continuous', 'domain': (-0.28, -0.001)}]

    domain_grid = [{'name': 'conv_1_size', 'type': 'discrete', 'domain': [3]},
                   {'name': 'conv_1_filters', 'type': 'discrete', 'domain': [32, 64]},
                   {'name': 'conv_2_size', 'type': 'discrete', 'domain': [3]},
                   {'name': 'conv_2_filters', 'type': 'discrete', 'domain': [32, 64]},
                   {'name': 'dropout_p', 'type': 'continuous', 'domain': [0.1,0.3]},
                   {'name': 'batch_size', 'type': 'discrete', 'domain': [32, 64]},
                   {'name': 'log_learning_rate', 'type': 'continuous', 'domain': [-6.9, -4.605, -3.0]},
                   {'name': 'log_decay', 'type': 'continuous', 'domain': [-20.0]},
                   {'name': 'log_rho', 'type': 'continuous', 'domain': [-0.16, -0.01]}]



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
            BO = black_box_optimizer(domain)
        else:
            BO = black_box_optimizer(domain_grid, random_grid_elements=False, gridpoints=96)
        BO.minimize(cnn.train_cifar, niter, extra_params={'epochs': epochs}, verbose=True)
        Y[:BO.Y.shape[0], run] = BO.Y[:, 0]
        X.append(BO.X)
        times.append(BO.times)

    try:
        np.savetxt("result_{}_cifar_small.csv".format(optimizer), Y, delimiter=",")
    except:
        print('Error saving Y matrix')

    try:
        pickle.dump(X, open('Xdump_{}_cifar_small.pkl'.format(optimizer), 'wb'))
    except:
        print('Error dumping X')

    if optimizer == 'bayesian':
        try:
            np.savetxt("times_{}_cifar_small.csv".format(optimizer), np.array(times).T, delimiter=",")
        except:
            print('Error saving times matrix')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", help="Optimizer ('bayesian','random' or 'gridsearch')", default='bayesian')
    parser.add_argument("--niter", help="Number of optimizer iterations on each experiment", default=96)
    parser.add_argument("--nruns", help="Number of independent black box optimizations to run", default=10)
    parser.add_argument("--epochs", help="Number of epochs on each CIFAR-10 training", default=1)

    args = parser.parse_args()
    print('Launching {} independent runs on {}-epochs CIFAR-10 with {} total experiments each over {} optimizer \n'
          .format(args.nruns, args.epochs, args.niter, args.optimizer))

    main(args.optimizer, int(args.niter), int(args.nruns), int(args.epochs))

