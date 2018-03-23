import numpy as np
import multiprocessing as mp
from time import time

def generate_inverse_hessian(X,alpha=1e-6):
    #Employ Woodbury Identity to quicken process:
    H_i_inverse = alpha*np.eye(X.shape[1],X.shape[1])
    P_inverse = X.shape[0]*np.eye(X.shape[1],X.shape[1])

    for i in range(X.shape[0]):
        X_row = X[i].reshape((1,X.shape[1]))


        print H_i_inverse.shape
        print X_row.T.shape



        H_i_inverse = H_i_inverse- np.linalg.multi_dot( (H_i_inverse,X_row,np.linalg.inv(P_inverse + np.linalg.multi_dot( (X_row,H_i_inverse, X_row.T) )),X_row.T,H_i_inverse) )

    return H_i_inverse

# def gen_inverse(X,alpha=1e-6):
#     H= (float(1)/alpha)*np.eye(X.shape[1],X.shape[1])
#     for i in range(X.shape[0]):
#         X_row = X[i].reshape((1,X.shape[1]))
#         H += np.matmul(X_row.T,X_row)
#     return np.linalg.inv(H)



def gen_inverse(X, process_number=4,alpha=1e-6):
    t = time()

    pool = mp.Pool(processes=process_number)
    #Divide X
    X_array = np.split(X,process_number)
    #Calculate hessian using all 4 processors
    results = pool.map(gen_hessian,X_array)

    H = (float(1)/alpha)*np.eye(X.shape[1],X.shape[1])
    for result in results:
        H +=result

    print 'Done adding', time() -t
    return np.linalg.inv(H)


def gen_hessian(X):
    H =np.zeros((X.shape[1],X.shape[1]))
    for i in range(X.shape[0]):
        X_row = X[i].reshape((1,X.shape[1]))
        H += np.matmul(X_row.T,X_row)
    return H
