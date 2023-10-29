from Fx.Basic_Functions.basics import *
from Fx.Pseudonymisation.pseudo import *
from Fx.Pseudonymisation.attributes_identification import *


def hub():
    path_data_fitness = "Data_Fitness/athletes.csv"
    data_frame = load_data_frame(path_data_fitness)
    attributes_identification(data_frame)

    # Function to pseudonyms Data Frame and deleting athlete_id
    # pseudo_(data_frame)
