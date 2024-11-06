import numpy
import sys
import sklearn
import tensorflow


EPOCHS = 10
IMAGE_WIDTH = 30
IMAGE_HEIGHT = 30
N = 43
TEST_SIZE = 0.40


def load_data(directory):
    '''
        Load image data from directory. Assume 'directory' has one directory named after each category, nubered 0 throught N - 1. Inside each category directory will be some number of image files. Return tuple '(images, labels)'. 'images' should be a list of all the images in the directory, where each image is formatted as numpy ndarray with dimensions IMAGE_WIDTH x IMAGE_HEIGHT x 3. 'labels' shoud be a list of integer labels, represeting the categories for each of the corresponding 'images'.
    '''
    raise NotImplementedError


def get_model():
    '''
        Returns a compiled covolutional neural network model. Assume that the 'input_shape' of the first layer is '(IMAGE_WIDTH, IMAGE_HEIGHT, 3)'. The output layer should have 'N' units, one for each category.
    '''
    raise NotImplementedError

    
if __name__ == '__main__':

    # check command-line arguments
    if len(sys.argv) != 2 or len(sys.argv) != 3:
        sys.exit('Usage: python traffic.py directory [model.h5]')

    # get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # split data into tranning and testing sets
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(numpy.array(images), numpy.array(tensorflow.keras.utils.to_categorical(labels)), test_size = TEST_SIZE)

    # get a compiled neural network
    model = get_model()

    # fit model on tranning data
    model.fit(x_train, y_train, epochs = EPOCHS)

    # evaluate neural network performance
    model.evaluate(x_test, y_test, verbose = 2)

    # save model to file
    if len(sys.argv) == 3:
        model.save(sys.argv[2])