import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_path_diff_privacy = 'Assignment_2_Fx/Assignment_2_Data_sets/data_utility/diff_privacy_utility.csv'
file_path_k_anonymity = 'Assignment_2_Fx/Assignment_2_Data_sets/data_utility/k_anonymity_utility.csv'
file_path_k_means_simple = 'Assignment_2_Fx/Assignment_2_Data_sets/data_utility/k_means_simple_utility.csv'
file_path_l_diversity = 'Assignment_2_Fx/Assignment_2_Data_sets/data_utility/l_diversity_utility.csv'
file_path_ucc_simplified = 'Assignment_2_Fx/Assignment_2_Data_sets/data_utility/ucc_simplified_utility.csv'

datasets = {
    "Differential Privacy": pd.read_csv(file_path_diff_privacy),
    "K-Anonymity": pd.read_csv(file_path_k_anonymity),
    "K-Means Simple": pd.read_csv(file_path_k_means_simple),
    "L-Diversity": pd.read_csv(file_path_l_diversity),
    "UCC Simplified": pd.read_csv(file_path_ucc_simplified)
}

min_length = min([len(data) for data in datasets.values()])

def check_metric_consistency(metric):
    for label, data in datasets.items():
        if f'original_{metric}' not in data.columns or f'anonymized_{metric}' not in data.columns:
            return False, label
    return True, None


def plot_combined_mean_original(metric, title, ylabel):
    plt.figure(figsize=(15, 8))
    bar_width = 0.15
    colors = ['b', 'g', 'r', 'c', 'm']

    original_data = list(datasets.values())[0]
    index = np.arange(min_length)
    original_metric = original_data[f'original_{metric}'][:min_length]
    plt.bar(index, original_metric, bar_width, label='Original', color='black')

    for i, (label, data) in enumerate(datasets.items()):
        anonymized_metric = data[f'anonymized_{metric}'][:min_length]
        plt.bar(index + (i + 1) * bar_width, anonymized_metric, bar_width, label=f'{label} Anonymized', color=colors[i])

    plt.xlabel('Variables')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(index + bar_width * 3, original_data['Unnamed: 0'][:min_length])
    plt.legend()
    plt.tight_layout()
    plt.savefig("Assignment_2_Fx/Assignment_2_Data_sets/data_utility/means_comparison.png")


def plot_combined_std_original(metric, title, ylabel):
    plt.figure(figsize=(15, 8))
    bar_width = 0.15
    colors = ['b', 'g', 'r', 'c', 'm']

    original_data = list(datasets.values())[0]
    index = np.arange(min_length)
    original_metric = original_data[f'original_{metric}'][:min_length]
    plt.bar(index, original_metric, bar_width, label='Original', color='black')

    for i, (label, data) in enumerate(datasets.items()):
        anonymized_metric = data[f'anonymized_{metric}'][:min_length]
        plt.bar(index + (i + 1) * bar_width, anonymized_metric, bar_width, label=f'{label} Anonymized', color=colors[i])

    plt.xlabel('Variables')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(index + bar_width * 3, original_data['Unnamed: 0'][:min_length])
    plt.legend()
    plt.tight_layout()
    plt.savefig("Assignment_2_Fx/Assignment_2_Data_sets/data_utility/std_comparison.png")




def plot_combined_errors_single_original():
    file_paths = {
        "Differential Privacy": 'Assignment_2_Fx/Assignment_2_Data_sets/data_utility/diff_privacy_utility.csv',
        "K-Anonymity": 'Assignment_2_Fx/Assignment_2_Data_sets/data_utility/k_anonymity_utility.csv',
        "K-Means Simple": 'Assignment_2_Fx/Assignment_2_Data_sets/data_utility/k_means_simple_utility.csv',
        "L-Diversity": 'Assignment_2_Fx/Assignment_2_Data_sets/data_utility/l_diversity_utility.csv',
        "UCC Simplified": 'Assignment_2_Fx/Assignment_2_Data_sets/data_utility/ucc_simplified_utility.csv'
    }

    datasets = {label: pd.read_csv(file_path) for label, file_path in file_paths.items()}

    datasets["K-Anonymity"] = datasets["K-Anonymity"][datasets["K-Anonymity"]['Unnamed: 0'] != 'cluster_labels']
    datasets["K-Means Simple"] = datasets["K-Means Simple"][datasets["K-Means Simple"]['Unnamed: 0'] != 'cluster_labels']

    plt.figure(figsize=(15, 8))
    bar_width = 0.15
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    first_dataset = next(iter(datasets.values()))
    n_vars = first_dataset.shape[0]
    index = np.arange(n_vars) * (len(datasets) + 1) * bar_width

    for i, (label, data) in enumerate(datasets.items()):
        if 'MAE' in data and len(data['MAE']) == n_vars :
            mae = data['MAE']
            corrected_index = index + i * bar_width
            plt.bar(corrected_index, mae, bar_width, label=f'{label} MAE', color=colors[i % len(colors)])
        else :
            print(f"Skipping dataset {label} due to mismatched length.")

    plt.xlabel('Variables')
    plt.ylabel('Error Values')
    plt.title('Comparison of Anonymized MAE Across Datasets')
    variable_names = first_dataset[
        first_dataset.columns[0]].tolist()
    plt.xticks(index + bar_width / 2, variable_names, rotation=90)
    plt.legend()

    plt.tight_layout()
    plt.savefig("Assignment_2_Fx/Assignment_2_Data_sets/data_utility/combined_error_comparison.png")

def plot_data_utility():
    plot_combined_mean_original('mean', 'Comparison of Original and Anonymized Means Across Datasets', 'Means')
    plot_combined_std_original('std', 'Comparison of Original and Anonymized Standard Deviations Across Datasets', 'Standard Deviations')
    plot_combined_errors_single_original()
