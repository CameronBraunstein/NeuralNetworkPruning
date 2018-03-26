#Our Neural Network Class
import numpy as np
import DataLoader as dl
import FileLoader as fl
import Layer as l

from time import time

def calculate_accuracy_and_error(outputs,labels):
    difference = outputs-labels
    error = 0
    for i in range(difference.shape[0]):
        error += np.dot(difference[i],difference[i])
    error = error/difference.shape[0]
    correct = 0
    for i in range(outputs.shape[0]):
        if labels[i][np.argmax(outputs[i])] == 1:
            correct +=1
    accuracy = float(correct)/ outputs.shape[0]
    return error, accuracy


class Network:
    def __init__(self,layer_sizes=[784,300,100,10],learning_rate=1e-4,from_file=None,retain_mask=True,epsilons=[1,1,1]):
        self.learning_rate = learning_rate
        self.epsilons = epsilons
        self.layers = []
        self.dl = dl.DataLoader()
        if from_file!=None:
            counter=0
            while True:
                W,b = fl.retrieve_layer(from_file,counter)
                if W is not None:
                    self.layers.append(l.Layer(W = W, b=b,l_obs_threshold=epsilons[counter],retain_mask=retain_mask))
                else:
                    break
                counter +=1
        else:
            for i in range(len(layer_sizes)-1):
                self.layers.append(l.Layer(num_inputs=layer_sizes[i],num_outputs=layer_sizes[i+1],l_obs_threshold=epsilons[i]))

        self.train_images, self.train_labels = self.dl.get_training()
        self.test_images,self.test_labels = self.dl.get_testing()

    def calculate_weights(self):
        weights=0
        for layer in self.layers:
            weights +=layer.W.shape[0]*layer.W.shape[1]
        return weights

    def save_network(self,save_filename):
        fl.store_layers(save_filename,self.layers)

    def test_training(self):
        outputs = self.forward(self.train_images)
        return calculate_accuracy_and_error(outputs,self.train_labels)

    def test_testing(self):
        outputs = self.forward(self.test_images)
        return calculate_accuracy_and_error(outputs,self.test_labels)

    def shuffle_samples(self):
        permutation = np.random.permutation(self.train_images.shape[0])
        self.train_images = self.train_images[permutation]
        self.train_labels = self.train_labels[permutation]

    def forward(self,inputs):
        neuron_layer = inputs
        for i in range(len(self.layers)):
            neuron_layer = self.layers[i].forward(neuron_layer)
        return neuron_layer

    def backward(self,delta_outputs):
        delta = delta_outputs
        for i in range(1,len(self.layers)):
            #delta = self.layers[-i].backward(delta,self.learning_rate,new_delta_scalar=220000)
            delta = self.layers[-i].backward(delta,self.learning_rate)

    def train(self,iterations=3,batches=60000,save_filename=None):
        indices = range(0,self.train_images.shape[0]+1,self.train_images.shape[0]//batches)

        for i in range(iterations):
            if i % 20 == 0:
                self.shuffle_samples()

            for j in range(len(indices)-1):
                outputs = self.forward(self.train_images[j:j+1])
                delta_outputs = outputs-self.train_labels[j:j+1]

                self.backward(delta_outputs)
            print (self.test_testing())


    def l_obs_prune_simple(self,report_file='report_simple_l_obs.txt',measure=500,save_filename=None):
        weights = self.calculate_weights()
        errors_and_accuracies = -np.ones((weights//measure+1,2))
        iterations = 0

        #Initially calculate inverse_hessian
        for layer in self.layers:
            self.forward(self.train_images)
            layer.calculate_sub_inverse_hessian()
            loss = layer.calculate_loss()
            while loss < layer.threshold:
                if iterations % measure ==0:
                    errors_and_accuracies[iterations//measure,:] = self.test_testing()
                    print (loss, errors_and_accuracies[iterations//measure,:], iterations)
                layer.prune()
                loss = layer.calculate_loss()
                iterations +=1

        np.savetxt(report_file,errors_and_accuracies)

    def l_obs_prune_continuous(self,report_file='report_continuous_l_obs.txt',measure=500,recalculate_hessian=2000,layer_bias=10,retrain=False):
        weights = self.calculate_weights()
        errors_and_accuracies = -np.ones((weights//measure+1,2))
        self.forward(self.train_images)
        for layer in self.layers:
            layer.calculate_sub_inverse_hessian()
        losses = [layer.calculate_loss() for layer in self.layers]
        for i in range(len(losses)):
            losses[i] *= layer_bias**i
        last_pruned_layer = np.argmin(losses)

        for iterations in range(weights):
            self.layers[last_pruned_layer].prune()
            losses[last_pruned_layer] = (layer_bias**last_pruned_layer)*self.layers[last_pruned_layer].calculate_loss()
            last_pruned_layer = np.argmin(losses)

            if iterations % measure ==0:
                errors_and_accuracies[iterations//measure,:] = self.test_testing()
                print (errors_and_accuracies[iterations//measure,:], iterations)

            #Recalculate Hessians Periodically
            if iterations % recalculate_hessian ==0 and iterations!=0:
                if retrain:
                    self.train()
                self.forward(self.train_images)
                for i in range(1,len(self.layers)):
                    self.layers[i].calculate_sub_inverse_hessian()
                    losses[i]=(layer_bias**last_pruned_layer)*self.layers[i].calculate_loss()
                last_pruned_layer = np.argmin(losses)
        np.savetxt(report_file,errors_and_accuracies)

    def remove_by_magnitude(self,measure=500,report_file='report_control.txt'):
        weights = self.calculate_weights()
        errors_and_accuracies = -np.ones((weights//measure+1,2))
        print(weights, len(errors_and_accuracies))

        weights = [float("inf")]*len(self.layers)
        for layer in self.layers:
            layer.rank_weights()
        for i in range(len(weights)):
            weights[i] = self.layers[i].return_next_smallest()
        last_pruned_layer = np.argmin(weights)
        iterations = 0
        while last_pruned_layer != -1:
            self.layers[last_pruned_layer].prune_smallest_weight()
            weights[last_pruned_layer]=self.layers[last_pruned_layer].return_next_smallest()
            if min(weights)<float("inf"):
                last_pruned_layer=np.argmin(weights)
            else:
                last_pruned_layer=-1
            if iterations % measure == 0:
                errors_and_accuracies[iterations//measure,:] = self.test_testing()
                print(errors_and_accuracies[iterations//measure,:])
            iterations +=1
        np.savetxt(report_file,errors_and_accuracies)


if __name__ =='__main__':
    n = Network(from_file = 'unpruned',learning_rate=1e-7,epsilons=[3.16e+2,3.16e+2,2.2e+2],retain_mask=True) #3e-5 #1e-5 is good for fine tuning, 5e-5 good for approx
    n.l_obs_prune_continuous(report_file='report_l_obs_continuous_retrain.txt',retrain=True)

    #n.save_network(save_filename='l_obs')
