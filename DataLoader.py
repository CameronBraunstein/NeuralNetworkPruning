from mnist import MNIST
import numpy as np

mndata = MNIST('samples', return_type='numpy')

def get_training():
    images, labels = mndata.load_training()
    return images.T, vectorize_labels(labels)

def get_testing():
    images, labels = mndata.load_testing()
    return images.T, vectorize_labels(labels)

def vectorize_labels(labels):
    vectorized_labels = np.zeros((10,len(labels)))
    for i in range(vectorized_labels.shape[1]):
        vectorized_labels[labels[i]][i] = 1
    return vectorized_labels
