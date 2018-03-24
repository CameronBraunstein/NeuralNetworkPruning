#Our Neural Network Class
import numpy as np
import DataLoader as dl
import FileLoader as fl
import Layer as l
import n2ps_algo as n2ps

from time import time

def calculate_error(difference):
    error = 0
    for i in range(difference.shape[0]):
        error += np.dot(difference[i],difference[i])
    print (error/difference.shape[0])

def calculate_accuracy(outputs, labels):
    correct = 0
    for i in range(outputs.shape[0]):
        if labels[i][np.argmax(outputs[i])] == 1:
            correct +=1
            #print labels[i], outputs[i]
    print (correct/ outputs.shape[0])

def calculate_accuracy_and_error(outputs,labels):
    difference = outputs-labels
    error = 0
    for i in range(difference.shape[0]):
        error += np.dot(difference[i],difference[i])
    print ('Error:', error/difference.shape[0])

    correct = 0
    for i in range(outputs.shape[0]):
        if labels[i][np.argmax(outputs[i])] == 1:
            correct +=1
    print ('Accuracy:', float(correct)/ outputs.shape[0])


class Network:
    def __init__(self,layer_sizes=[784,300,100,10],learning_rate=1e-4,from_file=None,epsilons=[0,0,0]):
        self.learning_rate = learning_rate
        self.epsilons = epsilons
        self.layers = []
        self.dl = dl.DataLoader()

        if from_file!=None:
            arrays = fl.return_arrays(from_file)
            for i in range(int(len(arrays)/2)):
                self.layers.append(l.Layer(W = arrays[2*i], b=arrays[2*i+1],l_obs_threshold=epsilons[i]))
        else:
            for i in range(len(layer_sizes)-1):
                self.layers.append(l.Layer(num_inputs=layer_sizes[i],num_outputs=layer_sizes[i+1],l_obs_threshold=epsilons[i]))

        self.train_images, self.train_labels = self.dl.get_training()

    def test(self):
        test_images,test_labels = self.dl.get_testing()
        outputs = self.forward(test_images)
        calculate_accuracy_and_error(outputs,test_labels)


    def train(self,iterations=300,save_filename=None):
        t = time()
        for i in range(iterations):
            outputs = self.forward(self.train_images)

            delta_outputs = outputs-self.train_labels

            if i %1 ==0:
                print(i)
                calculate_error(delta_outputs)
                calculate_accuracy(outputs,self.train_labels)


            self.backward(delta_outputs)

        print ('time', time()- t)

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

    def l_obs_prune(self, save_filename=None,max_time=50,recalculate_hessian=500):

        self.forward(self.train_images)
        self.test()
        last_pruned_layer=0


        #Initially calculate inverse_hessian
        for layer in self.layers:
            layer.calculate_sub_inverse_hessian()

        losses=[float("inf")]*len(self.layers)
        for i in range(1,len(self.layers)):
            losses[i] = self.layers[i].calculate_loss()

        t = time()
        iterations = 0
        while True and time()-t< max_time:

            #Calculate losses
            losses[last_pruned_layer] = self.layers[last_pruned_layer].calculate_loss()

            #Find smallest loss
            min_loss = min(losses)
            if min_loss == float("inf"):
                print ('no more prunable weights')
                return
            index = losses.index(min_loss)
            print ('Layer to Prune:', index)
            self.layers[index].prune()
            last_pruned_layer=index

            self.test()
            iterations += 1

            #When the algorithm has run for long enough, recalculate the hessians
            if iterations % recalculate_hessian == 0:
                self.forward(self.train_images)
                for i in range(1, len(self.layers)):
                    self.layers[i].calculate_sub_inverse_hessian()

                for i in range(len(self.layers)):
                    losses[i] = self.layers[i].calculate_loss()


        if save_filename is not None:
            fl.store_arrays(save_filename,self.layers)

    def N2PS_prune_input_neurons(self):
        X = self.train_images
        W = self.layers[0].W
        num_inp_neurons = X.shape[1]
        txip = X.sum(axis=0)
        ftxip = l.sigmoid(abs(txip))
        si = np.zeros(num_inp_neurons)
        alpha = 0
        for i in range(num_inp_neurons):
            for j in range(W.shape[1]):
                si[i] += abs(ftxip[i] + W[i][j])
            alpha += si[i]    
        alpha = alpha/num_inp_neurons
        for i in range(X.shape[1]):
            if si[i] <= alpha:
                for j in range(W.shape[1]):
                    W[i][j] = 0
        self.layers[0].W =  W

    def N2PS_prune_hidden_neurons(self,ftnet,layer=0):
        
        inp_Wt = self.layers[layer].W
        out_Wt = self.layers[layer+1].W
        num_hid_neurons = inp_Wt.shape[1]
        tnetjl = np.zeros(num_hid_neurons)
    
        if layer == 0: 
            X_inp = self.train_images
            temp_tnet = np.matmul(X_inp,inp_Wt)
            tnetjl = temp_tnet.sum(axis=0)
        else:
            X_inp = ftnet
            for i in range(X_inp.shape[0]):
                tnetjl += X_inp[i]* inp_Wt[i]       
      
        si = np.zeros(num_hid_neurons)
        ftnet = np.zeros(num_hid_neurons)
    
        for s in range(num_hid_neurons):
            ftnetjl = l.sigmoid(abs(tnetjl[s]))
            for m in range(out_Wt.shape[1]):
                si[s] += abs(ftnetjl + out_Wt[s][m])
            ftnet[s] = ftnetjl
    
        beta = si.sum(axis=0)
        beta = beta/num_hid_neurons    
        for i in range(num_hid_neurons):
            if si[i] <= beta:
                for k in range(inp_Wt.shape[0]):
                    inp_Wt[k][i] = 0
                for j in range(out_Wt.shape[1]):
                    out_Wt[i][j] = 0
        self.layers[layer+1].W = out_Wt 
        return ftnet



#n = Network()

n = Network(from_file = 'save.txt',learning_rate=1e-4,epsilons=[1e-0,1e-0,1e-0])
#n.train(save_filename='save.txt')
#n.l_obs_prune(save_filename='l_obs.txt')
n.test()
n.N2PS_prune_input_neurons() #Pruning of input layer
ftnet = 0
for layer in range(len(n.layers)-1):
    ftnet = n.N2PS_prune_hidden_neurons(ftnet,layer=layer) #Pruning of hidden layer
n.train()
n.test()
