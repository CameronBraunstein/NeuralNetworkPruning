import numpy as np
import multiprocessing as mp
from time import time

# def gen_inverse(X, process_number=4,alpha=1e-6):
#     t = time()
#     print('gen processes')
#
#     pool = mp.Pool()
#     print('here')
#     #Divide X
#     X_array = np.split(X,process_number)
#     print(X_array)
#     #Calculate hessian using all 4 processors
#
#     #pool.map(gen_hessian,X_array[0])
#     pool = mp.Pool()
#     #pool.terminate()
#     results = [pool.apply_async(gen_hessian, X ) for X in X_array]
#
#
#     H = (float(1)/alpha)*np.eye(X.shape[1],X.shape[1])
#
#     print (results)
#     for result in results:
#         H +=result
#
#     return np.linalg.inv(H)
#
#
# def gen_hessian(X):
#     H =np.zeros((X.shape[1],X.shape[1]))
#     print('h')
#     for i in range(X.shape[0]):
#         X_row = X[i].reshape((1,X.shape[1]))
#         H += np.matmul(X_row.T,X_row)
#     return H

def gen_inverse(X,alpha=1e-6):
    H = (float(1)/alpha)*np.eye(X.shape[1],X.shape[1])
    for i in range(X.shape[0]):
        X_row = X[i].reshape((1,X.shape[1]))
        H += np.matmul(X_row.T,X_row)
    return np.linalg.inv(H)
