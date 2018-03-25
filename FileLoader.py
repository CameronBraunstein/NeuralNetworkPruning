import numpy as np
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
    counter = 0
    for layer in layers:
        print ('{}/layers/W{}.txt'.format(save_filename,counter))
        np.savetxt('{}/layers/W{}.txt'.format(save_filename,counter),layer.W)
        np.savetxt('{}/layers/b{}.txt'.format(save_filename,counter),layer.b)
        counter +=1

def retrieve_layer(save_filename,layer_number):
    try:
        W = np.loadtxt('{}/layers/W{}.txt'.format(save_filename,layer_number))
        b = np.loadtxt('{}/layers/b{}.txt'.format(save_filename,layer_number))
    except FileNotFoundError:
        return None, None
    return W,b
