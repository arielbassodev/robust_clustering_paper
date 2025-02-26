"""
Author : 

"""
import random
import numpy as np
from Robust_kmeans_methods import stof_detection as stof
from sklearn.cluster import kmeans_plusplus

def stof_initialisation(data,k):
    centroids = []
    centroid_number = 0
    while centroid_number != k:
        centroid, _ = kmeans_plusplus(data, n_clusters=1, random_state=0)
        if not stof.has_stof(data, centroid, 2):
            centroid_number = centroid_number + 1
         
    return centroids

