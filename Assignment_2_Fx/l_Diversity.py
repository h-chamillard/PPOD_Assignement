import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from diffprivlib.models import KMeans as DP_KMeans
from sklearn.cluster import KMeans
from collections import Counter
import random


def load_data_frame(path):
    return pd.read_csv(path)


def add_noise(row, percentage=0.08):
    for col in row.index:
        random_factor = 1 + random.uniform(-percentage, percentage)
        row[col] = max(0, row[col] * random_factor)  # S'assurer que les valeurs ne deviennent pas négatives
    return row


def find_clusters_with_false(l_diversity_results):
    clusters_with_false = []
    for index, row in l_diversity_results.iterrows():
        if not row.all():
            clusters_with_false.append(index)

    print(clusters_with_false)

    return clusters_with_false


def cluster_distance(cluster1_stats, cluster2_stats):
    distance = np.sqrt(((cluster1_stats - cluster2_stats) ** 2).sum().sum())
    return distance


def find_closest_clusters(clusters_to_merge, cluster_means_stds):
    closest_clusters = {}

    for cluster in clusters_to_merge:
        cluster_stats = cluster_means_stds.loc[cluster]
        distances = cluster_means_stds.apply(lambda x: cluster_distance(cluster_stats, x), axis=1)
        distances = distances.drop(cluster)
        closest_cluster = distances.idxmin()
        closest_clusters[cluster] = closest_cluster


    return closest_clusters


def process_clusters(data, cluster_means_stds, numeric_columns, l_diversity_value):
    l_diversity_results = check_l_diversity_all(data, l_diversity_value)
    clusters_to_merge = find_clusters_with_false(l_diversity_results)

    if not clusters_to_merge:
        return data

    closest_clusters = find_closest_clusters(clusters_to_merge, cluster_means_stds)
    closter_to_change = []

    for cluster_to_merge, target_cluster in closest_clusters.items():
        closter_to_change.append(target_cluster)
        data.loc[data['cluster_labels'] == cluster_to_merge, 'cluster_labels'] = target_cluster
    cluster_means_stds = data.groupby('cluster_labels')[numeric_columns].agg(['mean', 'std']).fillna(0)
    non_compliant_individuals = data['cluster_labels'].isin(closter_to_change)
    data.loc[non_compliant_individuals, numeric_columns] = data.loc[non_compliant_individuals, numeric_columns].apply(
        add_noise, axis=1)
    return process_clusters(data, cluster_means_stds, numeric_columns, l_diversity_value)


def check_l_diversity_all(data, l_diversity_value):
    l_diversity_age = check_l_diversity(data, 'age', l_diversity_value)
    l_diversity_height = check_l_diversity(data, 'height', l_diversity_value)
    l_diversity_weight = check_l_diversity(data, 'weight', l_diversity_value)

    return pd.DataFrame({
        'age': l_diversity_age,
        'height': l_diversity_height,
        'weight': l_diversity_weight
    })


def check_l_diversity(data, attribute, l_value):
    return data.groupby('cluster_labels')[attribute].nunique() >= l_value


def k_means_l_diversity():
    path = "Assignment_2_Fx/Assignment_2_Data_sets/df_clustered.csv"
    data = load_data_frame(path)
    data = data.fillna(0)
    numeric_columns = data.select_dtypes(include=['number']).drop(columns='cluster_labels').columns
    cluster_stats = data.groupby('cluster_labels')[numeric_columns].agg(['mean', 'std'])
    cluster_stats = cluster_stats.fillna(0)

    def replace_nan_with_random(row, stats):
        cluster_label = row['cluster_labels']
        for col in numeric_columns:
            if col != 'cluster_labels' and pd.isna(row[col]):
                mean = stats.loc[cluster_label, (col, 'mean')]
                std = stats.loc[cluster_label, (col, 'std')]
                row[col] = np.random.normal(mean, std)
        return row

    data = data.apply(lambda row: replace_nan_with_random(row, cluster_stats), axis=1)

    cluster_means_stds = data.groupby('cluster_labels')[['age', 'height', 'weight']].agg(['mean', 'std']).fillna(0)

    data = process_clusters(data, cluster_means_stds, numeric_columns, 7)

    data = data.drop(columns=['cluster_labels'])
    data.to_csv("Assignment_2_Fx/Assignment_2_Data_sets/l_diversity.csv", index=False)

    return data
