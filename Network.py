#Our Neural Network Class
import numpy as np
import DataLoader as dl
import Layer as l

def calculate_error(difference):
    error = 0
    for i in range(difference.shape[0]):
        error += np.dot(difference[i],difference[i])
    return error

class Network:
    def __init__(self,layer_sizes,learning_rate=1e-6):
        self.learning_rate = learning_rate
        self.layers = []
        for i in range(len(layer_sizes)-1):
            self.layers.append(l.Layer(layer_sizes[i],layer_sizes[i+1]))

        self.train_images, self.train_labels = dl.get_training()
        print self.train_images.shape, self.train_labels.shape

    def train(self,iterations=200):
        for i in range(iterations):
            outputs = self.forward(self.train_images)

            delta_outputs = outputs-self.train_labels
            print calculate_error(delta_outputs)

            self.backward(delta_outputs)

    def forward(self,inputs):
        neuron_layer = inputs
        for i in range(len(self.layers)):
            neuron_layer = self.layers[i].forward(neuron_layer)
        return neuron_layer

    def backward(self,delta_outputs):
        delta = delta_outputs
        for i in range(1,len(self.layers)):
            delta = self.layers[-i].backward(delta,self.learning_rate)


n = Network([784,20,10])
n.train()
