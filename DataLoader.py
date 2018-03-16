from mnist import MNIST
import numpy as np

mndata = MNIST('samples', return_type='numpy')

def get_training():
    images, labels = mndata.load_training()
    print images.shape
    return images, vectorize_labels(labels)

def get_testing():
    images, labels = mndata.load_testing()
    return images, vectorize_labels(labels)

def vectorize_labels(labels):
    vectorized_labels = np.zeros((len(labels),10))
    for i in range(vectorized_labels.shape[0]):
        vectorized_labels[i][labels[i]] = 1
    return vectorized_labels
