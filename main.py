from Assignment_2_Fx.machine_learning import *
from Assignment_2_Fx.differential_privacy import *
from Assignment_2_Fx.k_anonymity import *
from Assignment_2_Fx.kmeans import *
from Assignment_2_Fx.l_Diversity import *
from Assignment_2_Fx.compartmentation_and_clustering import *
from Assignment_2_Fx.handling_NaN_values import *
from Assignment_2_Fx.plot_data_utility import *
from Assignment_2_Fx.plot_data_analytics import *
from Assignment_2_Fx.utility_testing import *
import time
from sklearn.metrics import mutual_info_score
from scipy.stats import wasserstein_distance

if __name__ == '__main__':

    # 0. Data selection
    handling_NaN()

    # 1. Anonymisation : Bare Bones
    k_means_simple()

    # 2. Anonymising your dataset
    k_means_k_anonymity()
    k_means_l_diversity()
    differential_privacy()

    # 3. Compartmentation and Clustering
    ucc_handler()

    # 4. Testing Data Utility
    # 4.1 Statistical indicators
    utility_testing()
    plot_data_utility()

    # 4.2
    machine_learning()

    # 4.3
    def normalized_wasserstein_distance(u_values, v_values, max_distance):
        u_values, v_values = u_values.dropna(), v_values.dropna()
        if len(u_values) == 0 or len(v_values) == 0:
            return 1.0
        return wasserstein_distance(u_values, v_values) / max_distance

    def normalized_mutual_information(u_values, v_values):
        u_values = u_values.astype(str).fillna('NaN_category')
        v_values = v_values.astype(str).fillna('NaN_category')

        mi = mutual_info_score(u_values, v_values)
        normalization = np.log(len(u_values)) or 1

        return mi / normalization

    def calculate_query_accuracy(df_original, df_anonymized):
        similarity_scores = []

        for column in df_original.columns:
            if np.issubdtype(df_original[column].dtype, np.number):
                max_distance = np.nanmax(df_original[column]) - np.nanmin(df_original[column])
                if max_distance > 0:
                    distance = normalized_wasserstein_distance(df_original[column], df_anonymized[column], max_distance)
                    similarity_scores.append(1 - distance)
            else:
                nmi = normalized_mutual_information(df_original[column], df_anonymized[column])
                similarity_scores.append(nmi)

        if similarity_scores:
            indicator = np.mean(similarity_scores)
        else:
            indicator = np.nan

        return indicator

    def calculate_information_loss(df_original, df_anonymized):
        if not (isinstance(df_original, pd.DataFrame) and isinstance(df_anonymized, pd.DataFrame)):
            raise ValueError('Both inputs must be pandas DataFrames.')
        p = df_original.shape[1]
        n = df_original.shape[0]
        il_sum = 0
        count = 0
        for j in range(p):
            column = df_original.iloc[:, j]
            if column.dtype.kind in 'bifc':
                S_j = np.nanstd(column)
                if S_j == 0:
                    continue
                for i in range(n):
                    x_ij = df_original.iloc[i, j]
                    y_ij = df_anonymized.iloc[i, j]
                    if pd.isna(x_ij) or pd.isna(y_ij):
                        continue
                    il_sum += np.abs(x_ij - y_ij) / (np.sqrt(2) * S_j)
                    count += 1
            else:
                for i in range(n):
                    x_ij = df_original.iloc[i, j]
                    y_ij = df_anonymized.iloc[i, j]
                    if pd.isna(x_ij) or pd.isna(y_ij):
                        continue
                    il_sum += 0 if x_ij == y_ij else 1
                    count += 1
        il = il_sum / count if count else np.nan
        return il


    def measure_anonymization(df_original, anonymize_func):
        print(anonymize_func)
        start_time = time.time()
        df_anonymized = anonymize_func()
        end_time = time.time()


        time_to_anonymize = end_time - start_time
        info_loss = calculate_information_loss(df_original, df_anonymized)
        # info_loss = info_loss_calculator(df_original, df_anonymized)
        query_accuracy = calculate_query_accuracy(df_original, df_anonymized)

        return time_to_anonymize, info_loss, query_accuracy


    size_of_data = [1000, 10000, 50000, 100000, 200000, 300000, 420003]
    # size_of_data = [100000]
    anonymization_methods = {
        'differential_privacy': differential_privacy,
        'ucc_handler': ucc_handler,
        'k_means_simple': k_means_simple,
        'k_means_l_diversity': k_means_l_diversity,
        'k_means_k_anonymity': k_means_k_anonymity
    }

    data_for_dfs = {method: [] for method in anonymization_methods}

    for size in size_of_data:
        df_original = handling_NaN(size)
        for method_name, method_func in anonymization_methods.items():
            time_taken, info_loss, q_accuracy = measure_anonymization(df_original, method_func)
            data_for_dfs[method_name].append(
                {'Size': size, 'Time': time_taken, 'Info_Loss': info_loss, 'Q_Accuracy': q_accuracy})

    dfs = {method: pd.DataFrame(data) for method, data in data_for_dfs.items()}

    for method_name, df in dfs.items():
        df.to_csv(f"Assignment_2_Fx/Assignment_2_Data_sets/stats/{method_name}_results.csv", index=False)

    plot_data_analytics()
