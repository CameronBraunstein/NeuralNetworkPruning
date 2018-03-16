import numpy as np

class Layer:
    def __init__(self,num_inputs,num_outputs):
        self.W = numpy.random.rand(num_inputs,num_outputs)
        self.b = numpy.random.rand(num_outputs,1)

    def forward(self, X):
        print 'hey'
