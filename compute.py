'''
    fichier qui contient toutes les fonctions pour calculer les covariances
'''

import numpy as np

def center(X):
    X_centred = np.zeros(shape=X.shape, dtype=X.dtype)
    mean = np.mean(X, axis=1)
    for i in range(0, X.shape[0]):
        X_centred[i] = X[i] - mean[i]
    return X_centred

def cov(X, lamda=10e-2):
    bias = (1 / (X.shape[0]-1)) if X.shape[0] != 1 else 1
    cov_matrix = bias * (np.dot(X.T, X))
    return cov_matrix + lamda * np.eye(X.shape[0])

def compute_temporel_cov(X):
    '''
        retourne des matrices de covariance temporelle
        calcul des C_T = somme(cov(X_Td))
    '''
    X_cov = []
    for x in X:
        for sub_x in x:
            x_centred = center(sub_x)
            x_cov = cov(np.array([x_centred[0]]))
            for x_ in x_centred[1:]:
                x_cov = x_cov + cov(np.array([x_]))
        X_cov.append(x_cov)
    return np.array(X_cov, dtype="object")

def compute_merge_sum_cov(X):
    '''
        on fusionne d'abord le x_i puis 
        on calcul la somme des covariance pour chaque x_i
    '''
    X_cov = []
    for x in X:
        x_merged = x[0].T
        for x_ in x[1:]:
            x_merged = np.concatenate((x_merged, x_.T), axis=1)
        x_merged = center(x_merged)
        #cacule de la somme des cov...
        x_cov = cov(x_merged[:, 0])
        for i in range(1, x_merged.shape[1]):
            x_cov = x_cov + cov(x_merged[:, i])
        X_cov.append(x_cov)
    return np.array(X_cov, dtype="object")

def compute_sum_cov(X):
    '''
        on calcule la somme des cov(x_i) pour chaque x
    '''
    X_cov = []
    for x in X:
        x_cov = cov(center(x[0].T))
        for x_ in x[1:]:
            x_cov = x_cov + cov(center(x_.T))
        X_cov.append(x_cov)
    return np.array(X_cov, dtype="object")