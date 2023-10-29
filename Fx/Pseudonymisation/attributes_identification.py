import pandas as pd
from Fx.Basic_Functions.basics import *


def attributes_identification(df):
    identification = [[0] * 4 for i in range(28)]
    max10 = [[0] * 2 for i in range(10)]
    i = 0
    for column in df.columns:
        identification[i][0] = column
        nb_unique_values = df[column].nunique()
        identification[i][1] = nb_unique_values
        nb_total_values = df[column].count()
        identification[i][2] = nb_total_values
        identification[i][3] = (nb_unique_values / nb_total_values)
        i += 1

    identification_ordered = sorted(identification, key=lambda x: x[3], reverse=True)
    print("Uniqueness Ratio of the 10 more unique attributes : ")
    for i in range(10):
        print(identification_ordered[i][0], " : ", identification_ordered[i][3])
