'''Train a simple CNN on CIFAR-10.

Returns the negative test accuracy score due to the minimization standard.
'''

from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np


def train_cifar(conv_1_size=3, conv_1_filters=32, conv_2_size=3, conv_2_filters=64, dropout_p=0.5, batch_size=32,
                log_learning_rate=-9.21, log_decay=-13.81, log_rho=-0.105, epochs=1, dense_1_neurons=1024):

    learning_rate = np.exp(log_learning_rate)
    decay = np.exp(log_decay)
    rho = np.exp(log_rho)

    num_classes = 10

    # The data, shuffled and split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()

    model.add(Conv2D(conv_1_filters, (conv_1_size, conv_1_size), padding='valid',
                     input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(conv_2_filters, (conv_2_size, conv_2_size)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(dense_1_neurons))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_p))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # initiate rmsprop optimizer
    opt = keras.optimizers.rmsprop(lr=learning_rate, rho=rho, decay=decay)

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
    train_cifar()
