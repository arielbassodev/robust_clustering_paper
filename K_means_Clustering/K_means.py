"""
Author : Ariel Basso

"""
from K_means_Clustering import clustering_methods as cm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from Initialisation_Method import stof_init
from Robust_kmeans_methods import outlier_detection as od
import Robust_kmeans_methods.stof_detection as stof

class KMeans():
    
    def __init__(self, k, max_iterations):
        self.k = k
        self.max_iterations = max_iterations

    def fit(self, data):
        centroids = stof_init.Stof_initialisation(data, self.k)
        delete_obser = 0
        indices_to_remove = []
        for i in range(self.max_iterations):
            print("iteration : ", i)
            old_centroids = centroids
            clusters = cm.assign_clusters(data, centroids)
            centroids = cm.compute_centroid(clusters)
            stof_list = stof.stof_detection(clusters, p=2)
            if cm.has_converged(old_centroids, centroids):
                break

        self.centroids = centroids
        self.clusters = clusters
        self.labels = self._get_labels(data, clusters)  # Calcul des labels
        inertia = cm.compute_intra_inertia(clusters)
        
        print("Inertia: ", inertia)
        cm.plot_clusters(clusters)
        return self.labels
    
    def _get_labels(self, data, clusters):
        """
        Retourne les labels pour chaque observation.
        """
        labels = np.zeros(len(data), dtype=int)
        for cluster_idx, cluster in enumerate(clusters):
            for observation in cluster:
                observation_idx = np.where((data == observation).all(axis=1))[0]
                labels[observation_idx] = cluster_idx
        return labels

    def get_labels(self):
        """
        Retourne les labels des données après clustering.
        """
        return self.labels
