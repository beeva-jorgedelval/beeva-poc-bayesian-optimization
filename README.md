# PoC Bayesian Optimization for Hyperparameter Selection in Machine Learning

This PoC explores the use of Bayesian Optimization in the context of hyperparameter selection.

To run the default alpine example just type

`python alpine_example.py --optimizer bayesian --ndim 10`

The results will appear as `result_alpine_bayesian_10.csv` and `times_alpine_bayesian_10.csv` with each run on an independent column.

The other examples in default configuration are plug and play.

A short tutorial on the module is available as `tutorial_blackboxoptimizer_module.ipynb`.

## Results on alpine

### 10-dimensional

![Alt text](results_alpine/results_alpine_10.png?raw=true "Performance on 10-dimensional alpine")

### 50-dimensional

![Alt text](results_alpine/results_alpine_50.png?raw=true "Performance on 50-dimensional alpine")

### 100-dimensional

![Alt text](results_alpine/results_alpine_100.png?raw=true "Performance on 100-dimensional alpine")

### 200-dimensional

![Alt text](results_alpine/results_alpine_200.png?raw=true "Performance on 200-dimensional alpine")

### Time of bayesian optimization with number of points

![Alt text](results_alpine/times_alpine.png?raw=true "Complexity of bayesian optimization")

## Results of CNN on CIFAR-10

![Alt text](results_cifar/results_cifar.png?raw=true "CIFAR-10 results")

# Results of logistic regression on MNIST

![Alt text](results_mnist/results_mnist.png?raw=true "MNIST results")
