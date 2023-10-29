import pandas as pd
from Fx.Basic_Functions.basics import *


def attributes_identification(df):
    identification = []
    for column in df.columns:
        nb_unique_values = df[column].nunique()
        identification[column][0] = nb_unique_values

    print(identification)