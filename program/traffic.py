import cv2
import numpy
import os
import sklearn
import sys
import sklearn.model_selection
import tensorflow


def load_data():

    raise NotImplementedError


def get_model():

    raise NotImplementedError


def main():

    # check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit('Usage: python traffic.py directory [model.h5]')

    # get images arrays and labels from all image files
    images, labels = load_data(sys.argv[1])

    # split data into tranning and testing sets
    labels = tensorflow.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(numpy.array(images), numpy.array(labels), test_size = 0.4)

    # get the compiled neural network
    model = get_model()

    # fit model on training data
    model.fit(x_train, y_train, epochs = 10)

    # evaluate neural network performance
    model.evaluate(x_test, y_test, verbose = 2)

    # save model to file
    if len(sys.argv) == 3:
        model.save(sys.argv[2])
        print(f"Model saved to {sys.argv[2]}")