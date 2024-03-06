import pandas as pd
import matplotlib

matplotlib.use('TkAgg')
from scipy.stats import entropy



def risk_assesment():

    anonymized_data_path = 'Last_Assignment_Dataset/new_adult_anonymized.csv'
    anonymized_data = pd.read_csv(anonymized_data_path)

    edu_work_dist = anonymized_data['edu_work'].value_counts()
    occ_salary_dist = anonymized_data['occ_salary'].value_counts()

    age_distribution = anonymized_data['age'].describe()

    unique_combinations = anonymized_data.groupby(['sex', 'age', 'race', 'edu_work', 'occ_salary']).size().reset_index(
        name='counts')
    rare_combinations = unique_combinations[unique_combinations['counts'] == 1]

    print(edu_work_dist, occ_salary_dist, age_distribution)
    print("Rare combination : ", rare_combinations.shape[0])

    edu_work_diversity = anonymized_data.groupby(['sex', 'race'])['edu_work'].nunique().reset_index(
        name='edu_work_diversity')

    occ_salary_diversity = anonymized_data.groupby(['sex', 'race'])['occ_salary'].nunique().reset_index(
        name='occ_salary_diversity')


    print("Edu_work Diversity : \n", edu_work_diversity)
    print("Occ_salary Diversity : \n", occ_salary_dist)

    global_occ_salary_dist = anonymized_data['occ_salary'].value_counts(normalize=True)
    grouped_occ_salary_dist = anonymized_data.groupby(['sex', 'race', 'edu_work'])['occ_salary'].value_counts(
        normalize=True).unstack(fill_value=0)

    t_closeness_values = pd.DataFrame(index=grouped_occ_salary_dist.index, columns=['t_closeness'])

    for index, row in grouped_occ_salary_dist.iterrows():
        t_closeness_values.loc[index, 't_closeness'] = entropy(row, global_occ_salary_dist)

    print("T Closeness values : ", t_closeness_values)

    unique_combinations_counts = anonymized_data.groupby(['sex', 'race', 'edu_work', 'occ_salary']).size().reset_index(
        name='counts')

    unique_records_count = unique_combinations_counts[unique_combinations_counts['counts'] == 1].shape[0]

    total_records = anonymized_data.shape[0]
    proportion_unique = unique_records_count / total_records

    print("Unique record count : ", unique_records_count)
    print("Proporiton unique : ", proportion_unique)
