import pandas as pd
from Fx.Basic_Functions.basics import *
from anonymizedf.anonymizedf import anonymize


def pseudo_(df):
    an = anonymize(df)
    create_fake_csv_name(an)
    load_fake_csv_name()
    new_df = get_new_df()
    print(new_df.head())


def create_fake_csv_name(an):
    fake_df = an.fake_names("name")
    fake_df.to_csv("athletes_pseudonyzed.csv", index=False)


def load_fake_csv_name():
    new_df = load_data_frame("athletes_pseudonyzed.csv")
    list_col = del_column_name(new_df.columns)
    list_col = refactor_order_columns(list_col)
    new_df = new_df[list_col]
    new_df.rename(columns={'Fake_name': 'name'}, inplace=True)
    new_df.to_csv("athletes_pseudonyzed.csv", index=False)


def del_column_name(list_col):
    list_col = list_col.delete(list_col.get_loc('name'))
    list_col = list(list_col)
    return list_col


def refactor_order_columns(list_col):
    last_el = list_col[-1]
    others_el = list_col[1:-1]
    list_col = [last_el] + others_el
    return list_col


def get_new_df():
    path = "athletes_pseudonyzed.csv"
    df = load_data_frame(path)
    return df


def create_fake_csv_id(an):
    fake_df = an.fake_ids("athlete_id")
    fake_df.to_csv("athletes_pseudonyzed.csv", index=False)


def load_fake_csv_id():
    new_df = load_data_frame("athletes_pseudonyzed.csv")
    list_col = del_column_id(new_df.columns)
    list_col = refactor_order_id(list_col)
    new_df = new_df[list_col]
    new_df.rename(columns={'Fake_athlete_id': 'athlete_di'}, inplace=True)
    new_df.to_csv("athletes_pseudonyzed.csv", index=False)


def del_column_id(list_col):
    list_col = list_col.delete(list_col.get_loc('athlete_id'))
    list_col = list(list_col)
    return list_col


def refactor_order_id(list_col):
    first_el = list_col[-1]
    others_el = list_col[1:-1]
    list_col = [first_el] + others_el
    return list_col
