import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from Network import Network
from collections import Counter

rc('font',**{'family':'serif','serif':['Times']})
rc('font', size=14)
rc('text', usetex=True)

def weight_histogram(from_file):
    n = Network(from_file=from_file)
    frequencies = {}
    for layer in n.layers:
        magnatudes = np.floor(np.log10(np.abs(layer.W)))
        print (magnatudes)
        for magnatude in magnatudes.flat:
            if magnatude in frequencies:
                frequencies[magnatude] +=1
            else:
                frequencies[magnatude]=1
    del frequencies[-float('inf')]
    plt.bar(list(frequencies.keys()), frequencies.values(), width=1, color='g')
    plt.xlabel('Order of Magnatude')
    plt.ylabel('Frequency')
    plt.title('Orders of Magnatude of Weights')
    plt.show()

def view_report(from_file):
    data=np.loadtxt(from_file)
    x = np.arange(0,1,float(1)/data.shape[0])[::-1]
    plt.plot(x,data[:,1],'r--')
    plt.xlim(1, 0)
    plt.xlabel('Compression Ratio')
    plt.title('Comparison of Pruning Algorithms')
    plt.ylabel('Accuracy')
    plt.show()



def compare(file1,file2,file3):
    hfont = {'fontname':'Helvetica'}
    data1=np.loadtxt(file1)
    data2=np.loadtxt(file2)
    data3=np.loadtxt(file3)
    x = np.arange(0,1,float(1)/data1.shape[0])[::-1]
    plt.plot(x,data1[:,1],'g--',label='L-OBS Pruning with Retraining')
    plt.plot(x,data2[:,1],'b--',label='L-OBS Pruning')
    plt.plot(x,data3[:,1],'r--',label='Magnitude Based Pruning')
    plt.legend(framealpha=1, frameon=True, loc='lower left')
    plt.xlim(1, 0)
    plt.xlabel('Compression Ratio')
    plt.title('Comparison of Pruning Algorithms')
    plt.ylabel('Accuracy')
    plt.show()

if __name__ == '__main__':
    #weight_histogram('l_obs')
    compare('report_l_obs_continuous_retrain.txt','report_continuous_l_obs.txt','report_control.txt')
    #view_report('report_simple_l_obs.txt')
