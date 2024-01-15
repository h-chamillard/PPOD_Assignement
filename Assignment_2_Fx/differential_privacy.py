import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from diffprivlib.models import KMeans as DP_KMeans
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




def remove_outliers(data, feature_names):
    for feature in feature_names:
        Q1 = data[feature].quantile(0.25)
        Q3 = data[feature].quantile(0.75)
        IQR = Q3 - Q1

        data[feature] = np.where(
            (data[feature] < (Q1 - 1.5 * IQR)) | (data[feature] > (Q3 + 1.5 * IQR)),
            np.nan,
            data[feature]
        )

    print(len(data['deadlift']))
    return data


def differential_privacy():
    path = "Assignment_2_Fx/Assignment_2_Data_sets/selected_data.csv"
    df = load_data_frame(path)

    df = remove_outliers(df, ['age', 'height', 'weight', 'candj', 'snatch', 'backsq', 'deadlift'])



    region_frequencies = df['region'].value_counts(normalize=True)
    gender_frequencies = df['gender'].value_counts(normalize=True)

    # perturbation_percentage = 0.30
    #
    # num_records_to_perturb = int(len(df) * perturbation_percentage)
    # indices_to_perturb = np.random.choice(df.index, size=num_records_to_perturb, replace=False)
    # df.loc[indices_to_perturb, 'region'] = np.random.choice(region_frequencies.index, size=len(indices_to_perturb), p=region_frequencies.values)
    # df.loc[indices_to_perturb, 'gender'] = np.random.choice(gender_frequencies.index, size=len(indices_to_perturb), p=gender_frequencies.values)


    numeric_columns = ['age', 'height', 'weight', 'candj', 'snatch', 'deadlift', 'backsq']
    df_numeric = df[numeric_columns]

    stds = df_numeric.std()

    noise_scale = 0.3 * stds
    noise = np.random.laplace(0, noise_scale, df_numeric.shape)
    df_noisy_numeric = df_numeric + noise
    df_noisy = df.copy()
    df_noisy[numeric_columns] = df_noisy_numeric

    print(len(df_noisy['deadlift']))

    df_noisy.to_csv("Assignment_2_Fx/Assignment_2_Data_sets/diff_privacy.csv", index=False)

    return df_noisy
