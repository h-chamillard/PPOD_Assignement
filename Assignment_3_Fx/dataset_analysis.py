import numpy as np
import pandas as pd
import matplotlib
from scipy.stats import chi2_contingency, zscore

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations


def dataset_analysis():

    df_anonymized = pd.read_csv('Last_Assignment_Dataset/adult_anonymized.csv')

    data_description = df_anonymized.describe()

    missing_values = df_anonymized.isnull().sum()

    print(data_description)
    print(missing_values)

    unique_ages = df_anonymized['age'].value_counts()

    categorical_columns = ['age', 'race', 'education', 'workclass', 'occupation', 'salary-class']
    unique_values_per_categorical_column = {column: df_anonymized[column].value_counts() for column in
                                            categorical_columns}

    plt.figure(figsize=(6, 4))
    sns.countplot(x='age', data=df_anonymized)
    plt.title('Distribution of Age')
    plt.xlabel('Age')
    plt.ylabel('Count')
    sex_distribution_fig = plt.gcf()
    plt.close()

    print(unique_ages)
    print(unique_values_per_categorical_column)


    def compute_chi_square_test(df, col1, col2):
        contingency_table = pd.crosstab(df[col1], df[col2])
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        return chi2, p


    chi_square_results = {}
    for i in range(len(categorical_columns)):
        for j in range(i + 1, len(categorical_columns)):
            col1, col2 = categorical_columns[i], categorical_columns[j]
            chi2, p = compute_chi_square_test(df_anonymized, col1, col2)
            chi_square_results[(col1, col2)] = {'chi2': chi2, 'p-value': p}

    print(chi_square_results)


    def identify_outliers_iqr(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        outlier_condition = ((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR)))
        return df.loc[outlier_condition]


    def identify_outliers_zscore(df, column):
        df['Z_score'] = zscore(df[column])
        return df.loc[np.abs(df['Z_score']) > 3]


    numerical_columns = df_anonymized.select_dtypes(include=[np.number]).columns
    outliers_zscore = {column: identify_outliers_zscore(df_anonymized, column) for column in numerical_columns}

    for column in numerical_columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df_anonymized[column])
        plt.title(f'Boxplot of {column}')
        plt.xlabel(column)
        plt.ylabel('Values')
        plt.show()

    outliers_iqr = {column: identify_outliers_iqr(df_anonymized, column) for column in numerical_columns}

    print("Outliers identified by Z-score:")
    for col, out_df in outliers_zscore.items():
        print(f"{col} has {len(out_df)} outliers")
        print(out_df)

    print("\nOutliers identified by IQR:")
    for col, out_df in outliers_iqr.items():
        print(f"{col} has {len(out_df)} outliers")
        print(out_df)

    for column in categorical_columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(y=column, data=df_anonymized)
        plt.title(f'Countplot of {column}')
        plt.xlabel('Counts')
        plt.ylabel(column.capitalize())
        plt.show()

    rare_combinations = {}
    for cols in combinations(categorical_columns, 2):
        counts = df_anonymized.groupby(list(cols)).size()
        rare_combinations[cols] = counts[counts <= 1]

    print("Rare combinations of categorical values:")
    for cols, counts in rare_combinations.items():
        print(f"{cols} has {len(counts)} rare combinations")
        for index, count in counts.items():
            print(f"{index}: {count} occurrence(s)")
