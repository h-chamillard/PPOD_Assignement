import pandas as pd
import numpy as np
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from collections import Counter

def load_data_frame(path):
    return pd.read_csv(path)

def hot_encode_and_fillna(df, column):
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column].astype(str))
    mean_value = df[column].mean()
    df[column] = df[column].fillna(int(mean_value))
    return df

def remove_outliers(df, col, min_val, max_val):
    df[col] = df[col].mask((df[col] < min_val) | (df[col] > max_val), np.nan)
    return df


def handle_outliers(df):
    columns_with_outliers = {
        'height': (0, 96),
        'weight': (50, 400),
        'candj': (45, 500),
        'snatch': (45, 400),
        'deadlift': (100, 1000),
        'backsq': (50, 800),
    }

    for col, (min_val, max_val) in columns_with_outliers.items():
        df = remove_outliers(df, col, min_val, max_val)
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
    return df


def k_means_k_anonymity():
    path = "Assignment_2_Fx/Assignment_2_Data_sets/selected_data.csv"
    df = load_data_frame(path)

    region_frequencies = df['region'].value_counts(normalize=True)
    label_encoders = {}
    columns_to_process = df.columns
    for column in columns_to_process:
        if df[column].dtype == 'float64':
            mean_value = df[column].mean()
            df[column] = df[column].apply(lambda x: float(mean_value) if pd.isnull(x) else float(x))
        elif df[column].dtype == 'object':
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column].astype(str))
            label_encoders[column] = le

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[columns_to_process])

    kmeans_model = KMeans(n_clusters=100,random_state=42)
    kmeans_model.fit(scaled_data)
    df['cluster_labels'] = kmeans_model.labels_


    def calculate_penalty(cluster_stats1, cluster_stats2):
        weight_mean_difference = 1.0
        weight_variance_difference = 0.5
        mean_difference = np.abs(cluster_stats1.loc['mean'] - cluster_stats2.loc['mean'])
        variance_difference = np.abs(cluster_stats1.loc['std'] - cluster_stats2.loc['std']) ** 2
        penalty = (weight_mean_difference * mean_difference + weight_variance_difference * variance_difference).sum()

        return penalty

    seuil = 15
    small_clusters = [i for i, count in enumerate(np.bincount(kmeans_model.labels_)) if count < seuil]
    cluster_centers = kmeans_model.cluster_centers_
    distances = euclidean_distances(cluster_centers)
    print(small_clusters)
    label_changes = {i: i for i in range(len(cluster_centers))}
    scaled_df = pd.DataFrame(scaled_data, columns=columns_to_process)

    for cluster in small_clusters:
        current_cluster_data = scaled_df[df['cluster_labels'] == cluster]
        current_cluster_stats = current_cluster_data.describe()
        similar_clusters = np.argsort(distances[cluster])
        min_penalty = np.inf
        best_cluster_to_merge = None
        for potential_merge in similar_clusters:
            if potential_merge not in small_clusters:
                potential_cluster_data = df[df['cluster_labels'] == potential_merge]
                potential_cluster_stats = potential_cluster_data.describe()
                penalty = calculate_penalty(current_cluster_stats, potential_cluster_stats)
                if penalty < min_penalty:
                    min_penalty = penalty
                    best_cluster_to_merge = potential_merge

        if best_cluster_to_merge is not None:
            indices_to_merge = np.where(kmeans_model.labels_ == cluster)[0]
            kmeans_model.labels_[indices_to_merge] = best_cluster_to_merge
            distances[cluster, :] = np.inf
            distances[:, cluster] = np.inf

    for original_label, new_label in label_changes.items():
        df['cluster_labels'].replace(original_label, new_label, inplace=True)

    df['cluster_labels'] = kmeans_model.labels_
    centroids = df.groupby('cluster_labels').agg(lambda x: x.median() if np.issubdtype(x.dtype, np.number) else x.value_counts().idxmax())

    for col in columns_to_process:
        if col in centroids.columns:
            df[col] = df['cluster_labels'].map(centroids[col])

    for column, le in label_encoders.items():
        if column in df.columns:
            df[column] = le.inverse_transform(df[column].astype(int))



    def calculate_k_anonymity(df):
        combined = df.astype(str).agg('|'.join, axis=1)

        frequency_counts = Counter(combined)
        k_anonymity_level = min(frequency_counts.values())

        return k_anonymity_level

    print("Level of K-Anonymity reached : ", calculate_k_anonymity(df))

    df.to_csv("Assignment_2_Fx/Assignment_2_Data_sets/k_anonymity.csv", index=False)

    return df

