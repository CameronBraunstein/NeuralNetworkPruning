import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    data=np.loadtxt('report.txt')
    print data
    x = np.arange(0,1,float(1)/data.shape[0])[::-1]
    plt.plot(x,data[:,1],'r--', x,data[:,0], 'b^')
    plt.xlim(1, 0)
    plt.xlabel('Compression Ratio')
    plt.show()
