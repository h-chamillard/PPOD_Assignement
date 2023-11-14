from Fx.Basic_Functions.basics import *
from Fx.Pseudonymisation.pseudo import *
from Fx.Pseudonymisation.attributes_identification import *
from Fx.Randomisation.randomizer_simple import *
from Fx.Randomisation.randomizer_complex import *
from Fx.Aggregation.aggregation_estimation import *
from Fx.Perturbation.perturbation_fx import *
from Fx.Data_Analysis.analysis import *


def hub():
    path_data_fitness = "Data_Fitness/athletes.csv"
    data_frame = load_data_frame(path_data_fitness)
    data_frame.dropna(subset=['name'])

    # 3.1.1
    # Function to estimate which columns need to be anonymize
    # attributes_identification(data_frame)

    # 3.1.2
    # Function to pseudonyms Data Frame and deleting athlete_id
    # pseudo_(data_frame)

    # 3.2.1
    # Function to randomize values without any pattern
    # randomizer(data_frame, 'name')

    # 3.2.2
    # Function to randomize values while respecting their first letter so the result becomes more meaningful
    # random_meaningful(data_frame)

    # 3.3
    # Function to estimate the quasi identifiers numbers and aggregate them per 10% each
    # aggregation_all(data_frame)

    # 3.4
    # Function to add noise and perturbed the Data Frame
    # data_frame = add_noise(data_frame,'age')

    # 3.5
    # Function to estimate the Information Loss (IL) Indicator and the Uniqueness Score of each transformed Dataframe
    # analysis()
