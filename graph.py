import matplotlib.pyplot as plt
import numpy as np
from Network import Network
from collections import Counter


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
    plt.title('Comparing Compressions')
    plt.ylabel('Accuracy')
    plt.show()

def compare(file1,file2):
    data1=np.loadtxt(file1)
    data2=np.loadtxt(file2)
    x = np.arange(0,1,float(1)/data1.shape[0])[::-1]
    plt.plot(x,data1[:,1],'r--',x,data2[:,1],'b--',)
    plt.xlim(1, 0)
    plt.xlabel('Compression Ratio')
    plt.title('Comparing Compressions')
    plt.ylabel('Accuracy')
    plt.show()

if __name__ == '__main__':
    #weight_histogram('l_obs')
    compare('report_control.txt','report_simple_l_obs.txt')
    #view_report('report_simple_l_obs.txt')
