"""Trains a simple logistic regression on MNIST with keras.

Returns the negative test accuracy score due to the minimization standard.
"""
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l1_l2
import numpy as np


def train_mnist(log_learning_rate=-9, l1=0.1, l2=0.1, batch_size=32, epochs=1):

    learning_rate = np.exp(log_learning_rate)
    reg = l1_l2(l1, l2)

    num_classes = 10

    # The data, shuffled and split between train and test sets:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    x_train = x_train.reshape([x_train.shape[0], x_train.shape[1]*x_train.shape[2]])
    x_test = x_test.reshape([x_test.shape[0], x_test.shape[1]*x_test.shape[2]])

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()

    model.add(Dense(num_classes, activation='softmax', W_regularizer=reg, input_shape=(x_train.shape[1], )))

    # initiate sgd optimizer
    opt = keras.optimizers.sgd(lr=learning_rate)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              shuffle=True)

    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test error:', 1.-scores[1])
    return -1.*scores[1]

if __name__ == "__main__":
    train_mnist()
