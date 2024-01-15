import math
from Assignment_1_Fx.Aggregation.aggregation_estimation import *


def analysis():
    list_df = get_data_frames()

    for i in list_df:
        i.dropna(subset=['name'])

    # aggr_mean_2(list_df[0], list_df[4])

    il_pseudo, u_pseudo = analysis_string_simple(list_df[0], list_df[1])
    il_rand_simple, u_rand_simple = analysis_string_simple(list_df[0], list_df[2])
    il_rand_meaningful, u_rand_meaningful = analysis_string_meaningful(list_df[0], list_df[3])
    il_aggr, u_aggr = analysis_float(list_df[0], list_df[4])
    print(il_aggr)
    il_perturbation, u_perturbation = analysis_float(list_df[0], list_df[5])

    print("Pseudonymisation         => Information Loss Indicator : ", il_pseudo, " / Uniqueness Score : ", u_pseudo)
    print("Simple Randomization     => Information Loss Indicator : ", il_rand_simple, " / Uniqueness Score : ",
          u_rand_simple)
    print("Meaningful Randomization => Information Loss Indicator : ", il_rand_meaningful, " / Uniqueness Score : ",
          u_rand_meaningful)
    print("Aggregation              => Information Loss Indicator : ", il_aggr, " / Uniqueness Score : ", u_aggr)
    print("Perturbation             => Information Loss Indicator : ", il_perturbation, " / Uniqueness Score : ",
          u_perturbation)


def get_data_frames():
    list_df = [0] * 6
    list_df[0] = df_original = load_data_frame("Data_Fitness/athletes.csv")
    list_df[1] = df_pseudo = load_data_frame("Data_Fitness/athletes_pseudonyzed.csv")
    list_df[2] = df_random_simple = load_data_frame("Data_Fitness/randomized_simple.csv")
    list_df[3] = df_random_meaningful = load_data_frame("Data_Fitness/randomized_meaningful.csv")
    list_df[4] = df_aggregation = load_data_frame("Data_Fitness/aggregation_mean.csv")
    list_df[5] = df_perturbation = load_data_frame("Data_Fitness/perturbation.csv")
    return list_df


def analysis_float(df_original, df_transformed):
    std = df_original['age'].std()
    sum_diff = sum(
        (x - y)/std for x, y in zip(df_original['age'].values.flatten(), df_transformed['age'].values.flatten()))
    IL = round((1 / len(df_original['age'])) * sum_diff , 2)

    uniqueness = unique_ratio(df_original, df_transformed, ['age'])

    return IL, uniqueness


def analysis_string_simple(df_original, df_transformed):
    sum_diff = sum(
        0 if x == y else 1 for x, y in
        zip(df_original['name'].values.flatten(), df_transformed['name'].values.flatten()))
    variance = sum_diff / (2 * len(df_original['name']))
    std_dev = math.sqrt(variance)
    IL = round((1 / len(df_original['name'])) * sum_diff / (math.sqrt(2 * std_dev)), 2)

    uniqueness = unique_ratio(df_original, df_transformed, ['name'])

    return IL, uniqueness


def analysis_string_meaningful(df_original, df_transformed):
    def modified_difference(x, y):
        if x == y:
            return 0
        elif x[0] == y[0]:
            return 0.6
        else:
            return 1

    sum_diff = sum(
        modified_difference(x, y) for x, y in
        zip(df_original['name'].values.flatten(), df_transformed['name'].values.flatten()))
    variance = sum_diff / (2 * len(df_original['name']))
    std_dev = math.sqrt(variance)
    IL = round((1 / len(df_original['name'])) * sum_diff / (math.sqrt(2 * std_dev)), 2)

    uniqueness = unique_ratio(df_original, df_transformed, ['name'])

    return IL, uniqueness


def aggr_mean_2(df_original, df_aggr):

    list_col = ['age']

    def custom_function(row):
        bounds = row[col][1:-1].split(', ')
        left_bound = float(bounds[0])
        right_bound = float(bounds[1])
        return round((left_bound + right_bound) / 2, 2)

    for col in list_col:
        df_aggr[col] = df_aggr.apply(custom_function, axis=1)
        # df_aggr = replace_Nan_with_mean(df_original, col)
        print(df_aggr[col].head())

    df_aggr.to_csv("aggregation_mean.csv", index=False)


def unique_ratio(df_original, df_transformed, columns):
    unique_ratios = []
    for i in columns:
        unique_percentage = (df_transformed[i].nunique() - df_original[i].nunique()) * 100 / len(df_original)
        unique_ratios.append(unique_percentage)
    unique = round(np.mean(unique_ratios), 3)

    return unique


def analysis_aggr_failed(df_original, df_transformed):
    list_col = identify_quasi_identifiers(df_original)
    reduced_df_original = df_original.loc[:, list_col]
    print("A")
    reduced_df_transformed = df_transformed.loc[:, list_col]
    print("B")
    num_attributes = len(reduced_df_original.columns)
    total_sum = 0

    for j in range(num_attributes):
        reduced_df_original = replace_Nan_with_mean(reduced_df_original, reduced_df_original.columns[j])
        reduced_df_transformed = replace_Nan_with_mean(reduced_df_transformed, reduced_df_transformed.columns[j])
        print(j)
        observation_sum = sum((reduced_df_original.iloc[i, j] - reduced_df_transformed.iloc[i, j]) for i in
                              range(len(reduced_df_original)))
        total_sum += observation_sum / math.sqrt(2 * np.std(reduced_df_original.iloc[:, j]))
        print(total_sum)

    IL = round((1 / (num_attributes * len(reduced_df_original['age']))) * total_sum, 2)
    uniqueness = unique_ratio(df_original, df_transformed, list_col)

    return IL, uniqueness
