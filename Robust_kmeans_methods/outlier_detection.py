import K_means_Clustering.clustering_methods as km
import numpy as np

"""
 Author: Basso Madjoukeng Ariel
 Date: 2022/11/07
 Description: Master study - Implementation of K-means clustering algorithm with outlier detection and centroid optimization.

"""

def outlier_detection(clusters, p):
  outlier = []
  for cluster in clusters:
    for observation in cluster:
       mean_contribution = compute_residual_contribution(observation, cluster)
       observation_contribution = km.compute_distance(observation, km.get_centroid_of_cluster(cluster))
       if observation_contribution > p * mean_contribution:
           outlier.append(observation)
  return outlier


def compute_residual_contribution(observation, cluster):
    centroid = km.get_centroid_of_cluster(cluster)
    distance_points_to_centroids = 0
    dist_table = []
    for point in cluster:
        if (point == observation).all():
         continue
        else:
            distance_points_to_centroids = km.compute_distance(point, centroid) 
            dist_table.append(distance_points_to_centroids)
    dist_table.sort(reverse=True)

    len_dist_table = (len(cluster)-1)

    dist_table = dist_table[:len_dist_table]

    mean = sum(dist_table)/len_dist_table
    return mean


def compute_observation_contribution(observation, cluster):
    centroid = km.get_centroid_of_cluster(cluster)
    distance_to_centroid = km.compute_distance(observation, centroid)
    distance_points_to_centroids = compute_residual_contribution(observation, cluster)
    return distance_to_centroid / distance_points_to_centroids
