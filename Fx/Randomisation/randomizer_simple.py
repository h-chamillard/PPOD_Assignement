import pandas as pd
import numpy as np
import random
import string
from Fx.Basic_Functions.basics import *


def randomizer(df, col_to_randomize):
    length = 6
    df[col_to_randomize] = [random_string(length) for i in range(len(df))]
    df.to_csv("randomized_simple.csv", index=False)
    print(df.head())


def random_string(length):
    car = string.ascii_lowercase
    return ''.join(random.choice(car) for _ in range(length))
