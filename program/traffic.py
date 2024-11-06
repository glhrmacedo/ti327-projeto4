import sys
import tensorflow

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


def main():

    # check command-line arguments

    # get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # split data into tranning and testing sets
    x_train, x_test, y_train, y_test = None

    # get a compiled neural network

    # fit model on tranning data

    # evaluate neural network performance

    # save model to file

    pass