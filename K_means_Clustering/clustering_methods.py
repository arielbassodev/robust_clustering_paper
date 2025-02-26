import math
import numpy as np
import pandas as pd
import random
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


def compute_distance(array_1, array_2):
    return np.linalg.norm(array_1 - array_2)


def compute_centroid(clusters):
 centroids = []
 for cluster in clusters:
   centroid = np.mean(cluster, axis=0)
   centroids.append(centroid)
 return centroids

def has_converged(old_centroids, new_centroids):
    if np.equal(old_centroids, new_centroids).all():
        return True
    else:
        return False
        
def assign_clusters(data, centroids):
  clusters = [[] for i in range(len(centroids))]
  for observation in data:
    distances = [compute_distance(observation, centroid) for centroid in centroids]
    cluster_index = distances.index(min(distances))
    clusters[cluster_index].append(observation)
  return clusters

def plot_clusters(clusters):
    color = ['r','g','b','y']
    for cluster in clusters:
      for elem in cluster:
        plt.scatter(elem[2], elem[0],c=color[clusters.index(cluster)])     
    plt.show()

def compute_intra_inertia(clusters):
  inertia = 0
  centroids = compute_centroid(clusters)
  for cluster in clusters:
    for elem in cluster:
      inertia += compute_distance(elem, centroids[clusters.index(cluster)])**2
  return inertia

def get_centroid_of_cluster(cluster):
  centroid = np.mean(cluster, axis=0)
  return centroid

