import numpy as np

def generate_inverse_hessian(X,alpha=1e-6):
    #Employ Woodbury Identity to quicken process:
    H_i_inverse = alpha*np.eye((X.shape[1],X.shape[1]))
    P_inverse = layer.X.shape[0]*np.eye((X.shape[1],X.shape[1]))

    for i in range(layer.X.shape[0]):
        X_row = X[i]
        H_i_inverse = H_i_inverse- np.linalg.multi_dot(H_i_inverse,X_row.T,np.linalg.inv(P_inverse + np.linalg.multi_dot(X_row,H_i_inverse, X_row.T) ),X_row,H_i_inverse)

    return H_i_inverse
