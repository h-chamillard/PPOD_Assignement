import pandas as pd
from sklearn.impute import SimpleImputer
import pdb


def load_data_frame(path):
    data_frame = pd.read_csv(path)
    data_frame = data_frame.dropna(subset=['name'])
    return data_frame


def what_column(index_col, data_frame):
    column_to_print = data_frame.iloc[:, index_col - 1]
    return column_to_print


def least_present_value_count(df, col):
    counts = df[col].value_counts()
    sorted_counts = counts.sort_values()
    least_value = sorted_counts.index[0]
    nombre_occurrences = sorted_counts.iloc[0]
    print("The least present value is : ", least_value, " with ", nombre_occurrences, " occurrences")


def replace_Nan_with_mean(df, col):
    df[col].fillna(df[col].mean(), inplace=True)
    return df

def score_unique(df):
    unique_rows = df.drop_duplicates().shape[0]
    print(unique_rows)
    score_uniqueness = (unique_rows / df.shape[0]) * 100
    print(score_uniqueness)


