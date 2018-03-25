import numpy as np
import math

from time import time
from Hessian import gen_inverse

def sigmoid(x):
    return float(1)/(1+math.e**(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))

class Layer:
    def __init__(self,num_inputs=0,num_outputs=0,W=None,b=None,l_obs_threshold=0,retain_mask=False):
        if W is not None and b is not None:
            self.W = W
            self.b = np.reshape(b,(-1,len(b)))
        else:
            self.W = np.random.rand(num_inputs,num_outputs)-0.5
            self.b = np.random.rand(1,num_outputs)-0.5

        self.gradient_W = np.zeros(self.W.shape)
        self.gradient_b = np.zeros(self.b.shape)
        self.threshold=l_obs_threshold**2
        self.losses = np.zeros(self.W.shape[0]*self.W.shape[1])
        self.delta_W=np.zeros((self.W.shape[0],1))

        if retain_mask:
            self.unpruned_W = np.copy(self.W)
            self.unpruned_W[self.unpruned_W!=0] = 1
        else:
            self.unpruned_W = np.ones(self.W.shape)

    def forward(self, X):
        self.X = X
        self.Z = np.matmul(self.X,self.W)+self.b
        self.output = sigmoid(self.Z)
        return self.output

    def backward(self,delta,learning_rate,l2=1e-5,new_delta_scalar=22000):

        partial_derivative_matrix = self.output*(1-self.output)*delta

        gradient_W = (float(1)/self.X.shape[0])*np.matmul(partial_derivative_matrix.T,self.X).T + l2*self.W #Include l2
        #gradient_W = np.matmul(partial_derivative_matrix.T,self.X).T + l2*self.W #Include l2
        gradient_b = (float(1)/self.X.shape[0])*np.matmul(partial_derivative_matrix.T, np.ones((partial_derivative_matrix.shape[0],1))).T
        #gradient_b = np.matmul(partial_derivative_matrix.T, np.ones((partial_derivative_matrix.shape[0],1))).T

        new_delta=new_delta_scalar*np.matmul(partial_derivative_matrix,self.W.T)

        #UPDATE WEIGHTS
        self.W -= learning_rate*self.unpruned_W*gradient_W
        self.b -= learning_rate*gradient_b


        return new_delta

    def calculate_sub_inverse_hessian(self):
        self.sub_inverse_hessian = gen_inverse(self.X)
        self.loss_matrix = self.W*self.W / np.diag(self.sub_inverse_hessian).reshape(len(self.sub_inverse_hessian),-1)

        #Ensure nothing goes above the threshold
        self.loss_matrix[np.where(self.loss_matrix>self.threshold)] = float("inf")

        #Set already pruned weights to inf
        for i in range(self.unpruned_W.shape[0]):
            for j in range(self.unpruned_W.shape[1]):
                if self.unpruned_W[i][j] == 0:
                    self.loss_matrix[i][j]= float("inf")
        loss_list = self.loss_matrix.reshape(-1).argsort()
        self.loss_indices = iter([ [i//self.loss_matrix.shape[1],i%self.loss_matrix.shape[1]] for i in loss_list])


    def calculate_loss(self):
        self.i_j = next(self.loss_indices)
        print (self.i_j)
        i,j = self.i_j[0],self.i_j[1]
        self.delta_W = (-float(self.W[i][j]) / self.sub_inverse_hessian[i][i])*self.sub_inverse_hessian[:,i]
        return self.loss_matrix[i,j]

        #Calculate delta

    def prune(self):
        i,j = self.i_j[0],self.i_j[1]
        self.W[:,j] += self.unpruned_W[:,j]*self.delta_W

        #Ensure that i,j is changed to zero (Sometimes a rounding error will effect this)
        self.W[i][j] = 0
        self.unpruned_W[i][j] = 0

    # def find_smallest_weight(self):
    #     try:
    #         result = min(filter(lambda x: x > 0, abs(self.W).flat))
    #     except ValueError:
    #         return float('inf')
    #     indices= (np.nonzero(result == abs(self.W)))
    #     self.i_j = [indices[0][0],indices[1][0]]
    #     return result

    def rank_weights(self):
        argsort = np.argsort(abs(self.W).flat)
        self.indices = [[i//self.W.shape[1],i%self.W.shape[1]] for i in argsort]
        self.counter = 0

    def prune_smallest_weight(self):
        i,j = self.indices[self.counter][0],self.indices[self.counter][1]
        self.W[i,j] = 0
        self.unpruned_W[i,j] = 0
        self.counter +=1

    def return_next_smallest(self):
        if self.counter >= len(self.indices):
            return float("inf")
        i,j = self.indices[self.counter][0],self.indices[self.counter][1]
        return abs(self.W[i,j])
    # def prune_and_return_next_smallest_weight(self):
    #     self.W[self.i_j[0],self.i_j[1]] = 0
    #     self.unpruned_W[self.i_j[0],self.i_j[1]] = 0
    #     return self.find_smallest_weight()
