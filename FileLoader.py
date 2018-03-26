import numpy as np
import os

def store_layers(save_filename,layers):
    if not os.path.exists(save_filename):
        os.makedirs(save_filename)
    counter = 0
    for layer in layers:
        np.savetxt('{}/W{}.txt'.format(save_filename,counter),layer.W)
        np.savetxt('{}/b{}.txt'.format(save_filename,counter),layer.b)
        counter +=1

def retrieve_layer(save_filename,layer_number):
    try:
        W = np.loadtxt('{}/W{}.txt'.format(save_filename,layer_number))
        b = np.loadtxt('{}/b{}.txt'.format(save_filename,layer_number))
    except IOError:
        return None, None
    return W,b
