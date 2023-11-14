import pandas as pd
from faker import Faker
import re
import csv
import random


def random_meaningful(df):
    df['name'] = df['name'].apply(clean)
    name_library = create_name_library(names_generator(df))
    lookup_table = {}
    for index, row in df.iterrows():
        name = row['name']
        first_letter = clean_letters(name[0])
        replaced_name = name_library[first_letter]
        df.at[index, 'name'] = replaced_name

        if first_letter not in lookup_table:
            lookup_table[first_letter] = [name]
        else:
            lookup_table[first_letter].append(name)

    df.to_csv("randomized_meaningful.csv", index=False)
    file_lookup = 'lookup_table.csv'
    with open(file_lookup, 'w', encoding='utf-8', newline='') as f:
        [f.write('{0},{1};'.format(key, value)) for key, value in lookup_table.items()]


def names_generator(df):
    fake = Faker()
    data = {
        'name': first_letter_library(df)
    }
    df_name = pd.DataFrame(data)
    random_name = []
    for i in range(len(df_name)):
        original_letter = df_name.at[i, 'name']
        new_name = fake.first_name()
        j = 0
        max_try = 5000
        while not new_name.startswith(original_letter) and j < max_try:
            new_name = fake.first_name()
            j += 1

        if j == max_try:
            new_name = original_letter
        random_name.append(new_name)
    return random_name


def first_letter_library(df):
    first_letters = df['name'].str[:1]
    unique_letters = first_letters.unique()
    unique_letters = list(unique_letters)
    unique_letters = [x.upper() for x in unique_letters]
    unique_letters = list(set(unique_letters))
    text = "".join(unique_letters)
    text = re.sub(r'[ÁÂÀÄÃÅÆА]', 'A', text)
    text = re.sub(r'[ÝỲŶ]', 'Y', text)
    text = re.sub(r'[ÉÈÊËЕ]', 'E', text)
    text = re.sub(r'[ÌİÍ]', 'E', text)
    text = re.sub(r'[ÛÜU]', 'U', text)
    text = re.sub(r'[ÒÓÔÕØÖ]', 'O', text)
    text = re.sub(r'[Ž]', 'Z', text)
    text = re.sub(r'[X]', 'X', text)
    text = re.sub(r'[Q]', 'Q', text)
    text = re.sub(r'[ĽŁ]', 'L', text)
    text = re.sub(r'[CС]', 'C', text)
    text = re.sub(r'[Т]', 'T', text)
    unique_letters = list(text)
    unique_letters = list(set(unique_letters))
    return unique_letters


def clean(name):
    name = re.sub(r'[^\w\s]', '', name)
    return name


def create_name_library(list_names):
    name_library = {}
    for name in list_names:
        first_letter = name[0]  # Prend la première lettre du mot
        name_library[first_letter] = name
    return name_library


def clean_letters(letter):
    letter = letter.upper()
    letter = re.sub(r'[ÁÂÀÄÃÅÆА]', 'A', letter)
    letter = re.sub(r'[ÝỲŶ]', 'Y', letter)
    letter = re.sub(r'[ÉÈÊËЕ]', 'E', letter)
    letter = re.sub(r'[ÌİÍ]', 'E', letter)
    letter = re.sub(r'[ÛÜU]', 'U', letter)
    letter = re.sub(r'[ÒÓÔÕØÖ]', 'O', letter)
    letter = re.sub(r'[Ž]', 'Z', letter)
    letter = re.sub(r'[X]', 'X', letter)
    letter = re.sub(r'[Q]', 'Q', letter)
    letter = re.sub(r'[ĽŁ]', 'L', letter)
    letter = re.sub(r'[CС]', 'C', letter)
    letter = re.sub(r'[Т]', 'T', letter)
    return letter
