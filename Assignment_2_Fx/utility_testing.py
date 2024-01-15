import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_data_frame(path):
    return pd.read_csv(path)

def descriptive_stats(original, anonymized):
    stats_original = original.describe().transpose()
    stats_anonymized = anonymized.describe().transpose()
    stats_original.columns = ['original_' + col for col in stats_original.columns]
    stats_anonymized.columns = ['anonymized_' + col for col in stats_anonymized.columns]
    combined_stats = pd.merge(stats_original, stats_anonymized, left_index=True, right_index=True)
    return combined_stats


def completeness(original, anonymized):
    missing_original = original.isnull().sum()
    missing_anonymized = anonymized.isnull().sum()

    completeness_comparison = pd.DataFrame({
        'original_missing': missing_original,
        'anonymized_missing': missing_anonymized,
        'original_missing_perc': (missing_original / len(original)) * 100,
        'anonymized_missing_perc': (missing_anonymized / len(anonymized)) * 100
    })
    return completeness_comparison


def correlation_analysis(original, anonymized):
    correlations = {}
    for column in original.columns:
        if column in anonymized.columns and original[column].dtype in [np.float64, np.int64]:
            combined = pd.concat([original[column], anonymized[column]], axis=1, keys=['original', 'anonymized']).dropna()
            correlation = pearsonr(combined['original'], combined['anonymized'])[0]
            correlations[column] = correlation
    return correlations


def similarity_measures(original, anonymized):
    similarity_results = {}

    correlations = correlation_analysis(original, anonymized)
    similarity_results['pearson_correlation'] = correlations
    mae_results = {}
    rmse_results = {}
    for column in original.columns:
        if column in anonymized.columns and original[column].dtype in [np.float64, np.int64]:
            original_col_data = original[column].dropna()
            anonymized_col_data = anonymized.loc[original_col_data.index, column]
            mae = np.mean(np.abs(original_col_data - anonymized_col_data))
            rmse = np.sqrt(np.mean((original_col_data - anonymized_col_data) ** 2))
            mae_results[column] = mae
            rmse_results[column] = rmse

    return {'MAE': mae_results, 'RMSE': rmse_results}


def plot_mean_comparison(stats_comparison, filename):
    means = stats_comparison.loc[:, ['original_mean', 'anonymized_mean']]
    means.plot(kind='bar', figsize=(10, 6))
    plt.title('Comparison of Mean Values')
    plt.ylabel('Mean')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filename)


def plot_std_comparison(stats_comparison, filename):
    stds = stats_comparison.loc[:, ['original_std', 'anonymized_std']]
    stds.plot(kind='bar', figsize=(10, 6), color=['skyblue', 'orange'])
    plt.title('Comparison of Standard Deviations')
    plt.ylabel('Standard Deviation')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filename)

def plot_missing_comparison(completeness_comparison, filename):
    missing = completeness_comparison.loc[:, ['original_missing_perc', 'anonymized_missing_perc']]
    missing.plot(kind='bar', figsize=(10, 6), color=['lightgreen', 'red'])
    plt.title('Comparison of Missing Values Percentage')
    plt.ylabel('Percentage of Missing Values')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filename)



def plot_RMSE(rmse_results, filename):
    rmse_values = list(rmse_results.values())
    columns = list(rmse_results.keys())
    plt.figure(figsize=(10, 6))
    plt.bar(columns, rmse_values, color='lightcoral')
    plt.title('Root Mean Squared Error (RMSE) Comparison')
    plt.ylabel('RMSE')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(filename)


def plot_MAE(mae_results, filename):
    mae_values = list(mae_results.values())
    columns = list(mae_results.keys())
    plt.figure(figsize=(10, 6))
    plt.bar(columns, mae_values, color='lightcoral')
    plt.title('Mean Absolute Error (MAE) Comparison')
    plt.ylabel('MAE')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(filename)


def plot_correlations(original, anonymized, filename):
    correlations = correlation_analysis(original, anonymized)

    plt.figure(figsize=(10, 6))
    plt.bar(correlations.keys(), correlations.values())
    plt.xlabel('Columns')
    plt.ylabel('Pearson Correlation')
    plt.title('Correlation between Original and Anonymized Data')
    plt.xticks(rotation=45)
    plt.ylim(-1, 1)
    plt.savefig(filename)


def utility_testing():

    path = "Assignment_2_Fx/Assignment_2_Data_sets/selected_data.csv"
    data = load_data_frame(path)
    path = "Assignment_2_Fx/Assignment_2_Data_sets/diff_privacy.csv"
    anon_data = load_data_frame(path)

    stats_comparison = descriptive_stats(data, anon_data)
    completeness_comparison = completeness(data, anon_data)
    similarity_results = similarity_measures(data, anon_data)
    similarity_results_df = pd.DataFrame(similarity_results)
    all_results = pd.concat([stats_comparison, completeness_comparison, similarity_results_df], axis=1)
    all_results.to_csv('Assignment_2_Fx/Assignment_2_Data_sets/data_utility/diff_privacy_utility.csv', index=True)


    path = "Assignment_2_Fx/Assignment_2_Data_sets/k_anonymity.csv"
    anon_data = load_data_frame(path)

    stats_comparison = descriptive_stats(data, anon_data)
    completeness_comparison = completeness(data, anon_data)
    similarity_results = similarity_measures(data, anon_data)
    similarity_results_df = pd.DataFrame(similarity_results)
    all_results = pd.concat([stats_comparison, completeness_comparison, similarity_results_df], axis=1)
    all_results.to_csv('Assignment_2_Fx/Assignment_2_Data_sets/data_utility/k_anonymity_utility.csv', index=True)


    path = "Assignment_2_Fx/Assignment_2_Data_sets/kmeans.csv"
    anon_data = load_data_frame(path)

    stats_comparison = descriptive_stats(data, anon_data)
    completeness_comparison = completeness(data, anon_data)
    similarity_results = similarity_measures(data, anon_data)
    similarity_results_df = pd.DataFrame(similarity_results)
    all_results = pd.concat([stats_comparison, completeness_comparison, similarity_results_df], axis=1)
    all_results.to_csv('Assignment_2_Fx/Assignment_2_Data_sets/data_utility/k_means_simple_utility.csv', index=True)

    path = "Assignment_2_Fx/Assignment_2_Data_sets/l_diversity.csv"
    anon_data = load_data_frame(path)

    stats_comparison = descriptive_stats(data, anon_data)
    completeness_comparison = completeness(data, anon_data)
    similarity_results = similarity_measures(data, anon_data)
    similarity_results_df = pd.DataFrame(similarity_results)
    all_results = pd.concat([stats_comparison, completeness_comparison, similarity_results_df], axis=1)
    all_results.to_csv('Assignment_2_Fx/Assignment_2_Data_sets/data_utility/l_diversity_utility.csv', index=True)

    path = "Assignment_2_Fx/Assignment_2_Data_sets/ucc_simplified.csv"
    anon_data = load_data_frame(path)

    stats_comparison = descriptive_stats(data, anon_data)
    completeness_comparison = completeness(data, anon_data)
    similarity_results = similarity_measures(data, anon_data)
    similarity_results_df = pd.DataFrame(similarity_results)
    all_results = pd.concat([stats_comparison, completeness_comparison, similarity_results_df], axis=1)
    all_results.to_csv('Assignment_2_Fx/Assignment_2_Data_sets/data_utility/ucc_simplified_utility.csv', index=True)
