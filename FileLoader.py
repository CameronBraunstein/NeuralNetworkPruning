import numpy as np
import os

# def tokenizer(filename):
#     with open(filename) as f:
#         chunk = []
#         for line in f:
#             if 'HEAD'in line:
#                 continue
#             if 'END' in line:
#                 yield chunk
#                 chunk = []
#                 continue
#             chunk.append(line)
#
# def return_arrays(filename):
#     arrays = [np.loadtxt(A) for A in tokenizer(filename)]
#     return arrays
#
# def store_arrays(save_filename,layers):
#     f = open(save_filename,"w")
#     for i in range(len(layers)):
#         f.write("HEAD\n")
#         np.savetxt(f,layers[i].W)
#         f.write("END\n")
#         f.write("HEAD\n")
#         np.savetxt(f,layers[i].b)
#         f.write("END\n")
#     f.close()

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
    except FileNotFoundError:
        return None, None
    return W,b
