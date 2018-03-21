#Our Neural Network Class
import numpy as np
import DataLoader as dl
import FileLoader as fl
import Layer as l

from time import time

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

def calculate_accuracy_and_error(outputs,labels):
    difference = outputs-labels
    error = 0
    for i in range(difference.shape[0]):
        error += np.dot(difference[i],difference[i])
    print 'Error:', error/difference.shape[0]

    correct = 0
    for i in range(outputs.shape[0]):
        if labels[i][np.argmax(outputs[i])] == 1:
            correct +=1
            #print labels[i], outputs[i]
    print 'Accuracy:', float(correct)/ outputs.shape[0]


class Network:
    def __init__(self,layer_sizes=[784,300,100,10],learning_rate=1e-4,from_file=None):
        self.learning_rate = learning_rate
        self.layers = []
        self.dl = dl.DataLoader()

        if from_file!=None:
            arrays = fl.return_arrays(from_file)
            for i in range(len(arrays)/2):
                self.layers.append(l.Layer(W = arrays[2*i], b=arrays[2*i+1]))
        else:
            for i in range(len(layer_sizes)-1):
                self.layers.append(l.Layer(num_inputs=layer_sizes[i],num_outputs=layer_sizes[i+1]))

        self.train_images, self.train_labels = self.dl.get_training()

    def test(self):
        test_images,test_labels = self.dl.get_testing()
        outputs = self.forward(test_images)
        calculate_accuracy_and_error(outputs,test_labels)


    def train(self,iterations=2000,save_filename=None):
        t = time()
        for i in range(iterations):
            outputs = self.forward(self.train_images)

            delta_outputs = outputs-self.train_labels

            if i %1 ==0:
                print i

                calculate_error(delta_outputs)
                calculate_accuracy(outputs,self.train_labels)


            self.backward(delta_outputs)
        print 'time', time()- t

        if save_filename is not None:
            fl.store_arrays(save_filename,self.layers)

    # def train_full(self,batches=10, iterations=1,save_filename=None):
    #     #Determine size of batches. For default, set batches=1
    #     batch_indices = range(0,self.train_images.shape[0]+1,(self.train_images.shape[0])/batches)
    #     batch_indices[-1] = self.train_images.shape[0]
    #
    #     t = time()
    #     for i in range(iterations):
    #         for j in range(len(batch_indices)-1):
    #             outputs = self.forward(self.train_images[batch_indices[j:j+1]])
    #             delta = outputs-self.train_labels[batch_indices[j:j+1]]
    #             for k in range(1,len(self.layers)):
    #                 delta = self.layers[-k].build_gradient(delta)
    #
    #         if i %1 ==0:
    #             print i
    #             outputs = self.forward(self.train_images)
    #             calculate_accuracy_and_error(outputs,self.train_labels)
    #
    #         for k in range(len(self.layers)):
    #             self.layers[k].update(self.learning_rate)


                #calculate_error(delta_outputs)
                #calculate_accuracy(outputs,self.train_labels)
        print 'time', time()- t

        if save_filename is not None:
            fl.store_arrays(save_filename,self.layers)


    # def train_stochastic(self,batches=10, iterations=1000,save_filename=None):
    #     #Determine size of batches. For default, set batches=1
    #     batch_indices = range(0,self.train_images.shape[0]+1,(self.train_images.shape[0])/batches)
    #     batch_indices[-1] = self.train_images.shape[0]
    #
    #     t = time()
    #     for i in range(iterations):
    #         for j in range(len(batch_indices)-1):
    #             outputs = self.forward(self.train_images[batch_indices[j:j+1]])
    #             delta_outputs = outputs-self.train_labels[batch_indices[j:j+1]]
    #             self.backward(delta_outputs)
    #
    #         if i %50 ==0:
    #             print i
    #             outputs = self.forward(self.train_images)
    #             calculate_accuracy_and_error(outputs,self.train_labels)
    #
    #             #calculate_error(delta_outputs)
    #             #calculate_accuracy(outputs,self.train_labels)
    #     print 'time', time()- t
    #
    #
    #
    #     if save_filename is not None:
    #         fl.store_arrays(save_filename,self.layers)
    #
    # def train_stochastic_random(self,batches=10, iterations=1000,save_filename=None):
    #     #Determine size of batches. For default, set batches=1
    #     batch_indices = range(0,self.train_images.shape[0]+1,(self.train_images.shape[0])/batches)
    #     batch_indices[-1] = self.train_images.shape[0]
    #
    #     t = time()
    #     for i in range(iterations):
    #         p = np.random.permutation(self.train_images.shape[0])
    #         for j in range(len(batch_indices)-1):
    #             outputs = self.forward((self.train_images[p])[batch_indices[j:j+1]])
    #             delta_outputs = outputs-(self.train_labels[p])[batch_indices[j:j+1]]
    #             self.backward(delta_outputs)
    #
    #         if i %50 ==0:
    #             print i
    #             outputs = self.forward(self.train_images)
    #             calculate_accuracy_and_error(outputs,self.train_labels)
    #
    #             #calculate_error(delta_outputs)
    #             #calculate_accuracy(outputs,self.train_labels)
    #     print 'time', time()- t
    #
    #
    #
    #     if save_filename is not None:
    #         fl.store_arrays(save_filename,self.layers)


    def forward(self,inputs):
        neuron_layer = inputs
        for i in range(len(self.layers)):
            neuron_layer = self.layers[i].forward(neuron_layer)
        return neuron_layer

    def backward(self,delta_outputs):
        delta = delta_outputs
        for i in range(1,len(self.layers)):
            delta = self.layers[-i].backward(delta,self.learning_rate,new_delta_scalar=170000)


#n = Network()

n = Network(from_file = 'save.txt',learning_rate=1e-4)
n.train(save_filename='save1.txt')
#n.train()
n.test()
