import numpy as np
import math

def sigmoid(x):
    return float(1)/(1+math.e**(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))

class Layer:
    def __init__(self,num_inputs=0,num_outputs=0,W=None,b=None):
        if W is not None and b is not None:
            self.W = W
            self.b = np.reshape(b,(-1,len(b)))
        else:
            self.W = np.random.rand(num_inputs,num_outputs)-0.5
            self.b = np.random.rand(1,num_outputs)-0.5

    def forward(self, X):
        self.X = X
        self.Z = np.matmul(self.X,self.W)+self.b
        self.output = sigmoid(self.Z)
        return self.output

    def backward(self,delta,learning_rate,l2=1e-4):

        partial_derivative_matrix = self.output*(1-self.output)*delta

        gradient_W = (float(1)/self.X.shape[0])*np.matmul(partial_derivative_matrix.T,self.X).T + l2*self.W #Include l2
        gradient_b = (float(1)/self.X.shape[0])*np.matmul(partial_derivative_matrix.T, np.ones((partial_derivative_matrix.shape[0],1))).T

        new_delta=np.matmul(partial_derivative_matrix,self.W.T)

        #UPDATE WEIGHTS
        self.W -= learning_rate*gradient_W
        self.b -= learning_rate*gradient_b


        return new_delta
