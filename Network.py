#Our Neural Network Class
import numpy as np
import DataLoader as dl
import Layer as l

dl.get_testing()

class Network:
    def __init__(self,layer_sizes):
        self.layers = []
        for i in range(len(layers)-1):
            self.layers.add(l.Layer(layer_sizes[i],layer_sizes[i+1]))

        self.train_images, self.train_labels = dl.get_training()

    def forward(self):
        
