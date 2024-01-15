import pandas as pd
import numpy as np


def add_noise(df, attribute, noise_factor=0.1, noise_fraction=0.5):
    df_original = df.copy()
    attribute_std = df[attribute].std()
    mask = np.random.rand(len(df)) < noise_fraction
    noise = np.random.normal(loc=0, scale=attribute_std * noise_factor, size=len(df))
    df.loc[mask, attribute] = df.loc[mask, attribute] + noise[mask]
    df[attribute] = df[attribute].astype(int)

    print("Std changed percentage  : ", round((df[attribute].std() - df_original[attribute].std()) / df_original[attribute].std() * 100, 2), "%")

    df.to_csv("perturbation.csv", index=False)
    return df
