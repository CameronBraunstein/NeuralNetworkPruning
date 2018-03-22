import numpy as np
import math

from time import time
from Hessian import generate_inverse_hessian

def sigmoid(x):
    return float(1)/(1+math.e**(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))

class Layer:
    def __init__(self,num_inputs=0,num_outputs=0,W=None,b=None,l_obs_threshold=0):
        if W is not None and b is not None:
            self.W = W
            self.b = np.reshape(b,(-1,len(b)))
        else:
            self.W = np.random.rand(num_inputs,num_outputs)-0.5
            self.b = np.random.rand(1,num_outputs)-0.5

        self.gradient_W = np.zeros(self.W.shape)
        self.gradient_b = np.zeros(self.b.shape)
        self.l_obs_threshold=0

    def forward(self, X):
        self.X = X
        self.Z = np.matmul(self.X,self.W)+self.b
        self.output = sigmoid(self.Z)
        return self.output

    def backward(self,delta,learning_rate,l2=1e-5,new_delta_scalar=1):

        partial_derivative_matrix = self.output*(1-self.output)*delta

        gradient_W = (float(1)/self.X.shape[0])*np.matmul(partial_derivative_matrix.T,self.X).T + l2*self.W #Include l2
        gradient_b = (float(1)/self.X.shape[0])*np.matmul(partial_derivative_matrix.T, np.ones((partial_derivative_matrix.shape[0],1))).T

        new_delta=new_delta_scalar*np.matmul(partial_derivative_matrix,self.W.T)

        #UPDATE WEIGHTS
        self.W -= learning_rate*gradient_W
        self.b -= learning_rate*gradient_b


        return new_delta

    # def build_gradient(self,delta):
    #     print 'building_gradient'
    #     t = time()
    #     partial_derivative_matrix = self.output*(1-self.output)*delta
    #     print 'pdm built', time() -t
    #
    #     t = time()
    #     self.gradient_W += (float(1)/self.X.shape[0])*np.matmul(partial_derivative_matrix.T,self.X).T
    #     print 'g_W built', time() -t
    #
    #     t = time()
    #     self.gradient_b += (float(1)/self.X.shape[0])*np.matmul(partial_derivative_matrix.T, np.ones((partial_derivative_matrix.shape[0],1))).T
    #     print 'g_b built', time() -t
    #
    #
    #     t = time()
    #     new_delta=new_delta_scalar*np.matmul(partial_derivative_matrix,self.W.T)
    #     print 'nd built', time() -t
    #
    #     return new_delta


    # def update(self,learning_rate, l2=1e-5):
    #     #Add regularizer
    #     self.gradient_W *= learning_rate
    #     self.gradient_b *= learning_rate
    #
    #     self.gradient_W += l2*self.W
    #
    #     #UPDATE WEIGHTS
    #     self.W -= self.gradient_W
    #     self.b -= self.gradient_b
    #
    #     self.gradient_W.fill(0)
    #     self.gradient_b.fill(0)


    def l_obs(self):
        self.delta_W = 0
        sub_hessian = generate_inverse_hessian(self.X)
        #
        value = 0

        return value




    # def backward_divided(self,delta,learning_rate,l2=1e-5,divides=1):
    #     divide_indices = range(0,self.X.shape[0]+1,(self.X.shape[0])/divides)
    #     divide_indices[-1] = self.X.shape[0]
    #
    #     partial_derivative_matrix = self.output*(1-self.output)*delta
    #     gradient_W = np.zeros((self.W.shape[0],self.W.shape[1]))
    #     gradient_b = np.zeros((self.b.shape[0],self.b.shape[1]))
    #
    #     for i in range(len(divide_indices)-1):
    #         gradient_W += np.matmul(partial_derivative_matrix[divide_indices[i]:divide_indices[i+1]].T,self.X[divide_indices[i]:divide_indices[i+1]]).T
    #         gradient_W += np.matmul(partial_derivative_matrix[divide_indices[i]:divide_indices[i+1]].T, np.ones((divide_indices[i+1]-divide_indices[i],1))).T
    #
    #     gradient_W *= (float(1)/self.X.shape[0])
    #     gradient_b *= (float(1)/self.X.shape[0])
    #
    #     gradient_W += l2*self.W
    #
    #     new_delta=np.matmul(partial_derivative_matrix,self.W.T)
    #
    #     self.W -= learning_rate*gradient_W
    #     self.b -= learning_rate*gradient_b
    #
    #
    #     return new_delta
