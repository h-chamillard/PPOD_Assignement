import pandas as pd
from itertools import combinations
from collections import Counter
import numpy as np

def generalize_float(df, col, bins, right_inclusive=True):
    labels = [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(bins)-1)]
    df[col] = pd.cut(df[col], bins=bins, labels=labels, right=right_inclusive, include_lowest=True)


def load_data_frame(path):
    return pd.read_csv(path)


def find_unique_column_combinations(df, max_combination_degree=5):
    unique_combinations = []
    for i in range(2, max_combination_degree + 1):
        for combo in combinations(df.columns, i):
            if df[list(combo)].dropna().duplicated().sum() < 10000 :
                unique_combinations.append(combo)

    return unique_combinations

def find_most_frequent_columns(unique_combinations):
    frequency_dict = {}
    for combination in unique_combinations:
        for column in combination:
            if column in frequency_dict:
                frequency_dict[column] += 1
            else:
                frequency_dict[column] = 1

    return frequency_dict


def interval_to_midpoint(df, col):
    def midpoint(interval):
        if isinstance(interval, str):
            bounds = interval.split('-')
            if len(bounds) == 2:
                low, high = float(bounds[0]), float(bounds[1])
                return float(round((low + high) / 2))
        return float(0)

    df[col] = df[col].apply(midpoint)


def ucc_handler():
    age_bins = [13, 20, 30, 40, 50, 60, 70, 80, 90]
    height_bins = [40, 50, 60, 70, 80, 90]
    weight_bins = [50, 100, 150, 200, 250, 300, 400]
    fran_bins = [60, 120, 150, 180, 210, 240, 270, 300, 330, 360, 390, 420, 480, 600]
    helen_bins = [450, 600, 800, 1000, 1800]
    grace_bins = [60, 240, 300, 360, 600]
    filthy50_bins = [900, 1500, 1800, 2400, 3600]
    fgonebad_bins = [60, 180, 220, 260, 340, 900]
    run400_bins = [45, 75, 105, 130, 170, 300]
    run5k_bins = [900, 1200, 1500, 1800, 2100, 2400, 3600]
    candj_bins = [45, 100, 150, 200, 250, 300, 500]
    snatch_bins = [45, 80, 115, 150, 185, 220, 400]
    deadlift_bins = [100, 200, 300, 400, 500, 600, 700, 1000]
    backsq_bins = [50, 150, 225, 300, 375, 450, 800]
    pullups_bins = [1, 10, 20, 30, 40, 50, 60, 100]

    path = "Assignment_2_Fx/Assignment_2_Data_sets/selected_data.csv"
    df = load_data_frame(path)
    unique_combination = find_unique_column_combinations(df, 5)

    count = 0
    # for i in find_most_frequent_columns(unique_combination):
    #     # print(i, " : ", find_most_frequent_columns(unique_combination)[i])
    #     count = count +1
    # if count > 0 :
    #     print("UCC discovery identified some column to anonymize")
    # if count == 0 :
    #     print("there isn't any more UCC using these criteria")

    generalize_float(df, 'age', age_bins)
    generalize_float(df, 'weight', weight_bins)
    generalize_float(df, 'backsq', backsq_bins)
    generalize_float(df, 'deadlift', deadlift_bins)


    unique_combination = find_unique_column_combinations(df, 5)
    count = 0
    for i in find_most_frequent_columns(unique_combination):
        print(i, " : ", find_most_frequent_columns(unique_combination)[i])
        count = count +1
    if count == 0 :
        print("there isn't any more UCC using these criteria")
    if count > 0 :
        print("UCC discovery identified some column to anonymize")


    df.to_csv("Assignment_2_Fx/Assignment_2_Data_sets/ucc_discovered.csv", index=False)

    interval_to_midpoint(df, 'age')
    interval_to_midpoint(df, 'weight')
    interval_to_midpoint(df, 'backsq')
    interval_to_midpoint(df, 'deadlift')

    df['deadlift'] = df['deadlift'].astype(float)

    df.to_csv("Assignment_2_Fx/Assignment_2_Data_sets/ucc_simplified.csv", index=False)

    return df


