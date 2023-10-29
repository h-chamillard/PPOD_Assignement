import pandas as pd


def load_data_frame(path):
    data_frame = pd.read_csv(path)
    return data_frame


def what_column(index_col, data_frame):
    column_to_print = data_frame.iloc[:, index_col - 1]
    return column_to_print
