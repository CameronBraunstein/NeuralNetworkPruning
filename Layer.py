import numpy as np
import math

def sigmoid(x):
    return float(1)/(1+math.e**(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))

class Layer:
    def __init__(self,num_inputs,num_outputs):
        self.W = np.random.rand(num_inputs,num_outputs)
        self.b = np.random.rand(1,num_outputs)

    def forward(self, X):
        self.X = X
        self.Z = np.matmul(self.X,self.W)+self.b
        self.output = sigmoid(self.Z)
        return self.output

    def backward(self,delta,learning_rate):

        partial_derivative_matrix = self.output*(1-self.output)*delta
        gradient_W = np.matmul(partial_derivative_matrix.T,self.X).T
        gradient_b = np.matmul(partial_derivative_matrix.T, np.ones((partial_derivative_matrix.shape[0],1))).T

        new_delta=np.matmul(partial_derivative_matrix,self.W.T)

        #UPDATE WEIGHTS
        self.W -= learning_rate*gradient_W
        self.b -= learning_rate*gradient_b


        return new_delta
