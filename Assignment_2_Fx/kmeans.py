import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

def load_data_frame(path):
    return pd.read_csv(path)

def hot_encode_and_fillna(df, column):
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column].astype(str))
    mean_value = df[column].mean()
    df[column] = df[column].fillna(int(mean_value))
    return df

def k_means_simple():
    path = "Assignment_2_Fx/Assignment_2_Data_sets/selected_data.csv"
    df = load_data_frame(path)

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


    centroids = df.groupby('cluster_labels').agg(lambda x: x.value_counts().idxmax() if x.dtype == 'int64' else x.mean())
    for col in columns_to_process:
        if col in centroids.columns:
            df[col] = df['cluster_labels'].map(centroids[col])
    for column, le in label_encoders.items():
        if column in df.columns:
            df[column] = le.inverse_transform(df[column].astype(int))


    path = "Assignment_2_Fx/Assignment_2_Data_sets/selected_data.csv"
    df_new = load_data_frame(path)
    df_new['cluster_labels'] = df['cluster_labels']
    df_new.to_csv("Assignment_2_Fx/Assignment_2_Data_sets/df_clustered.csv", index=False)

    df.to_csv("Assignment_2_Fx/Assignment_2_Data_sets/kmeans.csv", index=False)

    return df

