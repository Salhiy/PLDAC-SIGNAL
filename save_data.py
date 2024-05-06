'''
    sauvgarde toute les donnees (donnees, matrice de covariance...)
    ainsi pas besoin de re-calculer a chaque fois
'''

from reader import read_h5py
from compute import *
from data_augment import data_augment
import numpy as np

def save_data(data, suffix):
    np.save(f"./saved_data/X{suffix}.npy", data[0])
    np.save(f"./saved_data/Y{suffix}.npy", data[1])
    np.save(f"./saved_data/S{suffix}.npy", data[2])

def save_cov(X, suffix):
    x_cov_temp = compute_temporel_cov(X)
    np.save(f"./saved_data/X_cov_temp{suffix}.npy", x_cov_temp)
    x_merge_sum_cov = compute_merge_sum_cov(X)
    np.save(f"./saved_data/X_compute_merge_sum_cov{suffix}.npy", x_merge_sum_cov)
    x_sum_cov = compute_sum_cov(X)
    np.save(f"./saved_data/X_sum_cov{suffix}.npy", x_sum_cov)

def save_agumented(data):
    X, Y, S = data
    data_augmented = data_augment(X, Y, S, pourcentage=1)
    save_data(data_augmented, suffix="_augmented")
    return data_augmented

def save():
    #reshape imgs to 32*32 
    data = read_h5py("./SoliData.zip", reshaped=True)
    save_data(data, suffix="")
    #compute covs...
    save_cov(data[0], suffix="")
    #compute augmented data and then save it
    data_augmented = save_agumented(data)
    save_cov(data_augmented[0], "_augmented")

save()