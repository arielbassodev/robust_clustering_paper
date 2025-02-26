import numpy as np
from Initialisation_Method.stof_init import Stof_initialisation as sti
import K_means_Clustering.clustering_methods as cm
from  sklearn.datasets import load_iris
import math

def stof_detection(clusters, p):
    stof_observations = []
    stof_number = 0
    for cluster in clusters:
        for observation in cluster:
            centroid = cm.get_centroid_of_cluster(cluster)
            for colum, index in enumerate(observation):
                distance = math.pow((observation[colum]-centroid[colum]),2)
                residual_contribution = error_in_feature_contribution(cluster, observation, observation[colum], centroid[colum])
                error_observation_contribution = distance / residual_contribution 
                mean_residual_contribution = np.mean(residual_contribution)
                if error_observation_contribution > p * mean_residual_contribution:
                    stof_number += 1
                    pairs = (observation, colum)
                    stof_observations.append(pairs)
    print("Number of STOFs: ", stof_number)
    return stof_observations


def error_in_feature_contribution(cluster, observation, observation_value, centroid_colum):
    all_distance_except_observation = 0
    for obs in cluster:
        if (observation == obs).all():
            continue
        else:
            all_distance_except_observation = all_distance_except_observation + math.pow((observation_value-centroid_colum),2)
    return all_distance_except_observation


def has_stof(dataset,observation,treshold):
    for feature in dataset.columns:
        error_contribution = error_in_feature_contribution(dataset, observation, observation[feature], dataset[feature].mean())
        if error_contribution > treshold * dataset[feature].mean():
            return False
    return True
        
# stof_detection(centroids, k)