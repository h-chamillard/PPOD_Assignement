import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')


def integrated_and_simplified_edu_work(edu, work):
    if 'HS-grad' in edu or 'Below-HS' in edu:
        edu_level = 'Secondary'
    elif 'Bachelors' in edu or 'Masters' in edu or 'Doctorate' in edu:
        edu_level = 'Graduate'
    else:
        edu_level = 'Intermediate'

    if 'Private' in work:
        work_sector = 'Non-Government'
    elif 'gov' in work:
        work_sector = 'Government'
    elif 'Self-emp' in work:
        work_sector = 'Non-Government'
    else:
        work_sector = 'Other'

    mapping = {
        'Graduate-Government': 'GradGov',
        'Graduate-Non-Government': 'GradNoGov',
        'Graduate-Other': 'GradOther',
        'Secondary-Non-Government': 'SecNoGov',
        'Secondary-Government': 'SecGov',
        'Secondary-Other': 'SecOther',
        'Intermediate-Government': 'InterGov',
        'Intermediate-Non-Government': 'InterNoGov',
        'Intermediate-Other': 'InterOther',
        'Other': 'SecNoGov'
    }
    category = f"{edu_level}-{work_sector}"
    return mapping.get(category, 'SecNoGov')


def integrated_and_simplified_occ_salary(occ, salary):
    if 'Exec-managerial' in occ or 'Prof-specialty' in occ:
        occupation_type = 'Technical'
    elif 'Sales' in occ or 'Adm-clerical' in occ:
        occupation_type = 'Non-Technical'
    else:
        occupation_type = 'Other'

    if '>50K' in salary:
        salary_level = 'HigherIncome'
    else:
        salary_level = 'LowerIncome'

    mapping = {
        'Technical-LowerIncome': 'TechModest',
        'Non-Technical-LowerIncome': 'NoTechModest',
        'Other-LowerIncome': 'OtherModest',
        'Technical-HigherIncome': 'TechHigh',
        'Non-Technical-HigherIncome': 'NoTechHigh',
        'Other-HighIncome': 'HigherIncome'
    }
    category = f"{occupation_type}-{salary_level}"
    return mapping.get(category, 'Other')


def ensure_k_anonymity(data, column, k=500):
    counts = data[column].value_counts()
    to_replace = counts[counts < k].index
    data.loc[data[column].isin(to_replace), column] = 'SecNoGov'
    return data


def shuffle_fraction(data, column, frac=0.25):
    n_shuffle = int(len(data) * frac)
    indices_to_shuffle = np.random.choice(data.index, size=n_shuffle, replace=False)
    shuffled_values = data.loc[indices_to_shuffle, column].sample(frac=1).values
    data.loc[indices_to_shuffle, column] = shuffled_values
    return data


def new_anonymization():

    data = pd.read_csv('Last_Assignment_Dataset/adults.csv')

    data['race'] = data['race'].replace(['Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'], 'Other')
    data['sex'] = data['sex'].apply(lambda x: 1 if x == 'Female' else 0)
    below_hs_levels = ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th']
    data['education'] = data['education'].replace(below_hs_levels, 'Below-HS')
    data['edu_work'] = data.apply(lambda x: integrated_and_simplified_edu_work(x['education'], x['workclass']), axis=1)
    data['occ_salary'] = data.apply(lambda x: integrated_and_simplified_occ_salary(x['occupation'], x['salary-class']),
                                    axis=1)

    data = ensure_k_anonymity(data, 'edu_work', k=500)
    data = ensure_k_anonymity(data, 'occ_salary', k=500)

    data = data.drop(['education', 'workclass', 'occupation', 'salary-class'], axis=1)
    data = shuffle_fraction(data, 'edu_work')
    data = shuffle_fraction(data, 'occ_salary')
    data = shuffle_fraction(data, 'sex')
    data = shuffle_fraction(data, 'race')

    scale = 5.0
    noise = np.random.laplace(0, scale, size=data['age'].shape)
    age_noisy = data['age'] + noise
    age_noisy_clamped = np.clip(age_noisy, 10, 100)

    age_noisy_clamped.loc[data['age'] == 10] = 10
    age_noisy_clamped.loc[data['age'] == 100] = 100

    data['age'] = round(age_noisy_clamped)
    data.to_csv('Last_Assignment_Dataset/new_adult_anonymized.csv', index=False)