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
    print 'Accuracy:', float(correct)/ outputs.shape[0]


class Network:
    def __init__(self,layer_sizes=[784,300,100,10],learning_rate=1e-4,from_file=None,epsilons=[0,0,0]):
        self.learning_rate = learning_rate
        self.epsilons = epsilons
        self.layers = []
        self.dl = dl.DataLoader()

        if from_file!=None:
            arrays = fl.return_arrays(from_file)
            for i in range(len(arrays)/2):
                self.layers.append(l.Layer(W = arrays[2*i], b=arrays[2*i+1],l_obs_threshold=epsilons[i]))
        else:
            for i in range(len(layer_sizes)-1):
                self.layers.append(l.Layer(num_inputs=layer_sizes[i],num_outputs=layer_sizes[i+1],l_obs_threshold=epsilons[i]))

        self.train_images, self.train_labels = self.dl.get_training()

    def test(self):
        test_images,test_labels = self.dl.get_testing()
        outputs = self.forward(test_images)
        calculate_accuracy_and_error(outputs,test_labels)


    def train(self,iterations=10,save_filename=None):
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



    def forward(self,inputs):
        neuron_layer = inputs
        for i in range(len(self.layers)):
            neuron_layer = self.layers[i].forward(neuron_layer)
        return neuron_layer

    def backward(self,delta_outputs):
        delta = delta_outputs
        for i in range(1,len(self.layers)):
            delta = self.layers[-i].backward(delta,self.learning_rate,new_delta_scalar=220000)

    def l_obs_prune(self, save_filename=None,max_time=25,recalculate_hessian=500):

        self.forward(self.train_images)
        self.test()
        last_pruned_layer=0


        #Initially calculate inverse_hessian
        for layer in self.layers:
            layer.calculate_sub_inverse_hessian()

        # losses=[float("inf")]*len(self.layers)
        # for i in range(1,len(self.layers)):
        #     losses[i] = self.layers[i].calculate_loss_1()

        losses = [layer.calculate_loss() for layer in self.layers]

        t = time()
        iterations = 0
        while True and time()-t< max_time:
            pre = time()
            #Calculate losses
            losses[last_pruned_layer] = self.layers[last_pruned_layer].calculate_loss()

            #Find smallest loss
            min_loss = min(losses)
            if min_loss == float("inf"):
                print 'no more prunable weights', iterations
                return

            last_pruned_layer = losses.index(min_loss)
            self.layers[last_pruned_layer].prune()

            iterations += 1
            print time() - pre

            #When the algorithm has run for long enough, recalculate the hessians
            if iterations % recalculate_hessian == 0:
                self.forward(self.train_images)
                for i in range(1, len(self.layers)):
                    self.layers[i].calculate_sub_inverse_hessian()

                for i in range(len(self.layers)):
                    losses[i] = self.layers[i].calculate_loss()

                self.test()


        if save_filename is not None:
            fl.store_arrays(save_filename,self.layers)





if __name__ =='__main__':
    n = Network(from_file = 'save.txt',learning_rate=1e-4,epsilons=[1e-0,1e-0,1e-0])
    #n.train(save_filename='save.txt')
    n.l_obs_prune(save_filename='l_obs.txt')
    #n.train()
    n.test()
