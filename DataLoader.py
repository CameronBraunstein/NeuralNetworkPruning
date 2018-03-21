from mnist import MNIST
import numpy as np


class DataLoader:
    def __init__(self):
        self.mndata = MNIST('samples', return_type='numpy')

    def get_training(self):
        images, labels = self.mndata.load_training()

        return images/float(256), self.vectorize_labels(labels)

    def get_testing(self):
        images, labels = self.mndata.load_testing()
        return images/float(256), self.vectorize_labels(labels)

    def vectorize_labels(self,labels):
        vectorized_labels = np.zeros((len(labels),10))
        for i in range(vectorized_labels.shape[0]):
            vectorized_labels[i][labels[i]] = 1
        return vectorized_labels
