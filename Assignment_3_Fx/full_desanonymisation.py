import pandas as pd
from scipy.spatial.distance import chebyshev


def full_desanonymisation():

    df_anonymized = pd.read_csv('Last_Assignment_Dataset/adult_anonymized.csv')

    age_medians = {
        '17-25': 17,
        '26-40': 33,
        '41-65': 53,
        '+66': 85
    }
    df_anonymized['age'] = df_anonymized['age'].map(age_medians)
    df_anonymized['age'] = df_anonymized['age'].fillna(85)

    df_anonymized = df_anonymized.drop(['sex', 'race'], axis=1)
    df_anonymized['education'] = df_anonymized['education'].apply(lambda x: 1 if x == 'Higher education' else 0)
    df_anonymized['workclass'] = df_anonymized['workclass'].apply(lambda x: 1 if x == 'Government' else 0)
    df_anonymized['salary-class'] = df_anonymized['salary-class'].apply(lambda x: 1 if x == '>50K' else 0)
    df_anonymized['occupation'] = df_anonymized['occupation'].apply(
        lambda x: 1 if x == 'Technical' else (0.5 if x == 'Nontechnical' else 0))

    salary_weight_factor = 1.1
    occupation_weight_factor = 1
    workclass_weight_factor = 0.8
    education_weight_factor = 0.2

    df_anonymized['salary-class'] = df_anonymized['salary-class'] * salary_weight_factor
    df_anonymized['occupation'] = df_anonymized['occupation'] * occupation_weight_factor
    df_anonymized['workclass'] = df_anonymized['workclass'] * workclass_weight_factor
    df_anonymized['education'] = df_anonymized['education'] * education_weight_factor

    def assign_age_cluster_corrected(age):
        if 17 <= age <= 25:
            return 0
        elif 26 <= age <= 40:
            return 1
        elif 41 <= age <= 65:
            return 2
        else:
            return 3


    df_anonymized['age_cluster'] = df_anonymized['age'].apply(assign_age_cluster_corrected)
    centroids = df_anonymized.groupby('age_cluster')[['education', 'workclass', 'occupation', 'salary-class']].mean()


    def calculate_distance_to_centroids(row, centroids):
        distances = {}
        attributes = row[['education', 'workclass', 'occupation', 'salary-class']]
        for cluster_id in centroids.index:
            centroid = centroids.loc[cluster_id]
            distances[cluster_id] = chebyshev(attributes, centroid)
        return distances


    def calculate_relative_distance(row, centroids):
        distances = calculate_distance_to_centroids(row, centroids)
        cluster = row['age_cluster']
        if cluster == 0:
            relative_distance = distances[1]
        elif cluster == 3:
            relative_distance = - distances[2]
        else:
            lower_cluster = cluster - 1
            upper_cluster = cluster + 1
            relative_distance = distances[upper_cluster] - distances[lower_cluster]
        return relative_distance

    df_anonymized['relative_distance'] = df_anonymized.apply(lambda row: calculate_relative_distance(row, centroids),
                                                             axis=1)

    def apply_adjusted_distribution(dataframe, distribution_targets, cluster):
        cluster_data = dataframe[dataframe['age_cluster'] == cluster].copy()
        sorted_cluster_data = cluster_data.sort_values(by='relative_distance')
        total_observations = len(sorted_cluster_data)
        surplus_observations = total_observations - sum(distribution_targets.values())
        adjusted_distribution_targets = distribution_targets.copy()

        for value in sorted(adjusted_distribution_targets.keys(), reverse=True):
            if surplus_observations > 0:
                additional = min(surplus_observations, total_observations - adjusted_distribution_targets[value])
                adjusted_distribution_targets[value] += additional
                surplus_observations -= additional
            if surplus_observations <= 0:
                break

        adjusted_normalized_values = []
        current_index = 0
        for value, count in adjusted_distribution_targets.items():
            end_index = current_index + count
            adjusted_normalized_values.extend([value] * (end_index - current_index))
            current_index = end_index

        sorted_cluster_data['age_predicted'] = adjusted_normalized_values[:total_observations]
        dataframe.loc[sorted_cluster_data.index, 'age_predicted'] = sorted_cluster_data['age_predicted']
        return dataframe

    distribution_targets = {
        17: 400,
        18: 500,
        19: 600,
        20: 600,
        21: 600,
        22: 700,
        23: 700,
        24: 700,
        25: 800
    }

    df_anonymized = apply_adjusted_distribution(df_anonymized, distribution_targets, 0)

    distribution_targets = {
        26: 800,
        27: 800,
        28: 800,
        29: 800,
        30: 800,
        31: 800,
        32: 800,
        33: 800,
        34: 800,
        35: 800,
        36: 800,
        37: 800,
        38: 800,
        39: 800,
        40: 800,
    }

    df_anonymized = apply_adjusted_distribution(df_anonymized, distribution_targets, 1)

    distribution_targets = {
        41: 750,
        42: 750,
        43: 750,
        44: 750,
        45: 700,
        46: 600,
        47: 600,
        48: 550,
        49: 550,
        50: 500,
        51: 500,
        52: 450,
        53: 450,
        54: 400,
        55: 400,
        56: 400,
        57: 350,
        58: 350,
        59: 350,
        60: 300,
        61: 300,
        62: 300,
        63: 200,
        64: 200,
        65: 100,
    }

    df_anonymized = apply_adjusted_distribution(df_anonymized, distribution_targets, 2)

    distribution_targets = {
        66: 100,
        67: 100,
        68: 100,
        69: 90,
        70: 80,
        71: 70,
        72: 60,
        73: 50,
        74: 50,
        75: 40,
        76: 30,
        77: 20,
        78: 20,
        79: 15,
        80: 15,
        81: 12,
        82: 10,
        83: 5,
        84: 4,
        85: 3,
        86: 2,
        87: 1
    }

    df_anonymized = apply_adjusted_distribution(df_anonymized, distribution_targets, 3)

    df_anonymized.to_csv('Last_Assignment_Dataset/adult_predicted.csv', index=False)





