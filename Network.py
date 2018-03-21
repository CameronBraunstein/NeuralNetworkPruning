#Our Neural Network Class
import numpy as np
import DataLoader as dl
import FileLoader as fl
import Layer as l

def calculate_error(difference):
    error = 0
    for i in range(difference.shape[0]):
        error += np.dot(difference[i],difference[i])
    print error/difference.shape[0]

def calculate_accuracy(outputs, labels):
    correct = 0
    for i in range(outputs.shape[0]):
        if labels[i][np.argmax(outputs[i])] == 1:
            correct +=1
            #print labels[i], outputs[i]
    print float(correct)/ outputs.shape[0]

class Network:
    def __init__(self,layer_sizes=[784,300,100,10],learning_rate=1e-2,from_file=None):
        self.learning_rate = learning_rate
        self.layers = []

        if from_file!=None:
            arrays = fl.return_arrays(from_file)
            for i in range(len(arrays)/2):
                self.layers.append(l.Layer(W = arrays[2*i], b=arrays[2*i+1]))
        else:
            for i in range(len(layer_sizes)-1):
                self.layers.append(l.Layer(num_inputs=layer_sizes[i],num_outputs=layer_sizes[i+1]))

        self.train_images, self.train_labels = dl.get_training()

    def train(self,iterations=10000,save_filename=None):
        for i in range(iterations):
            outputs = self.forward(self.train_images)

            delta_outputs = outputs-self.train_labels

            if i %500 ==0:
                print i
                calculate_error(delta_outputs)
                calculate_accuracy(outputs,self.train_labels)

            self.backward(delta_outputs)

        if save_filename is not None:
            fl.store_arrays(save_filename,self.layers)

    def train_stochastic(self,batches=1, iterations=10000,save_filename=None):
        #Determine size of batches. For default, set batches=1
        batch_indices = range(0,self.train_images.shape[0]+1,(self.train_images.shape[0])/batches)
        batch_indices[-1] = self.train_images.shape[0]
        
        for i in range(iterations):
            for j in range(len(batch_indices)-1):
                outputs = self.forward(self.train_images[batch_indices[j:j+1]])
                delta_outputs = outputs-self.train_labels[batch_indices[j:j+1]]
                self.backward(delta_outputs)

            if i %500 ==0:
                print i
                calculate_error(delta_outputs)
                calculate_accuracy(outputs,self.train_labels)



        if save_filename is not None:
            fl.store_arrays(save_filename,self.layers)


    def forward(self,inputs):
        neuron_layer = inputs
        for i in range(len(self.layers)):
            neuron_layer = self.layers[i].forward(neuron_layer)
        return neuron_layer

    def backward(self,delta_outputs):
        delta = delta_outputs
        for i in range(1,len(self.layers)):
            delta = self.layers[-i].backward(delta,self.learning_rate)



n = Network()
n.train_stochastic()
#n = Network(from_file = 'save.txt')
#n.train(save_filename = 'save.txt')
