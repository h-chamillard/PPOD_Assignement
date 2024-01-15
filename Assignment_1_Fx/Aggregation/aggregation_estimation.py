import pandas as pd
import numpy as np
from Assignment_1_Fx.Pseudonymisation.attributes_identification import *


def aggregation_all(df):
    list_quasi_identifier = identify_quasi_identifiers(df)
    print(list_quasi_identifier)
    for i in list_quasi_identifier:
        new_df = aggregation_df(df, i)
        df.update(new_df[i])
    df.to_csv("aggregation.csv", index=False)


def aggregation_df(df, col):
    df_new = df[['athlete_id', col]]
    df_new = df_new.sort_values(by=col)
    print(df_new.columns)
    df_new[col] = pd.qcut(df_new[col], q=np.arange(0, 1.1, 0.1))

    def round_interval(interval):
        return pd.Interval(round(interval.left, 1), round(interval.right, 1), closed=interval.closed)

    df_new[col] = df_new[col].apply(round_interval)
    df_new = df_new.sort_index()
    return df_new


def identify_quasi_identifiers(df):
    float64_columns = df.select_dtypes(include='float64').columns.tolist()
    quasi_identifiers = []
    quasi_identifiers.append('age')
    quasi_identifiers.append('weight')
    quasi_identifiers.append('height')
    unique_ratio = attributes_identification(df)
    unique_ratio = unique_ratio[2:-1]

    result_dict = {}
    for sublist in unique_ratio:
        key = sublist[0]
        value = sublist[-1]
        result_dict[key] = value

    for column in float64_columns:
        if column in result_dict :
            if result_dict[column]>0.005:
                quasi_identifiers.append(column)

    quasi_identifiers.remove('fgonebad')
    return quasi_identifiers
