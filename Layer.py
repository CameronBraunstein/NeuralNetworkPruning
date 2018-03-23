import numpy as np
import math

from time import time
from Hessian import gen_inverse

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
        self.l_obs_threshold=l_obs_threshold
        self.losses = np.zeros(self.W.shape[0]*self.W.shape[1])
        self.delta_W=np.zeros((self.W.shape[0],1))
        self.unpruned_W = np.ones(self.W.shape)

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
        self.W -= learning_rate*self.unpruned_W*gradient_W
        self.b -= learning_rate*gradient_b


        return new_delta

    def calculate_sub_inverse_hessian(self):
        self.sub_inverse_hessian = gen_inverse(self.X)
        self.loss_matrix = self.W*self.W / np.diag(self.sub_inverse_hessian).reshape(len(self.sub_inverse_hessian),-1)

        #Ensure nothing goes above the threshold
        self.loss_matrix[np.where(self.loss_matrix>self.l_obs_threshold)] = float("inf")

        #Set already pruned weights to inf
        for i in range(self.unpruned_W.shape[0]):
            for j in range(self.unpruned_W.shape[1]):
                if self.unpruned_W[i][j] == 0:
                    self.loss_matrix[i][j]= float("inf")
        loss_list = self.loss_matrix.reshape(-1).argsort()
        print self.loss_matrix.reshape(-1)


        self.loss_indices = iter([ [i/self.loss_matrix.shape[1],i%self.loss_matrix.shape[1]] for i in loss_list])

    # def calculate_loss(self):
    #     loss= float("inf")
    #     self.i_j = [-1,-1]
    #
    #     pre = time()
    #     for i in range(self.sub_inverse_hessian.shape[0]):
    #         for j in range(self.W.shape[1]):
    #             if self.unpruned_W[i][j]==1:
    #                 value=0.5*(self.W[i][j])**2/self.sub_inverse_hessian[i][i]
    #                 if value<loss and math.sqrt(value)<self.l_obs_threshold:
    #                     loss = value
    #                     self.i_j = [i,j]
    #
    #     #Calculate delta
    #     print 'time:', time()-pre
    #     if loss == float("inf"):
    #         return loss
    #
    #     self.delta_W = (-float(self.W[self.i_j[0]][self.i_j[1]]) /self.sub_inverse_hessian[self.i_j[0]][self.i_j[0]])*self.sub_inverse_hessian[:,self.i_j[0]]
    #
    #     return loss

    def calculate_loss(self):
        self.i_j = next(self.loss_indices)
        i,j = self.i_j[0],self.i_j[1]
        self.delta_W = (-float(self.W[i][j]) /self.sub_inverse_hessian[i][i])*self.sub_inverse_hessian[:,i]
        return self.loss_matrix[i,j]

        #Calculate delta

    def prune(self):
        #self.W[:,self.i_j[1]] += self.unpruned_W[:,self.i_j[1]]*self.delta_W[:,self.i_j[1]]
        self.W[:,self.i_j[1]] += self.unpruned_W[:,self.i_j[1]]*self.delta_W
        #Ensure that i,j is changed to zero (Sometimes a rounding error will effect this)
        self.W[self.i_j[0]][self.i_j[1]] = 0
        self.unpruned_W[self.i_j[0]][self.i_j[1]] = 0
