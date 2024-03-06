from Assignment_3_Fx.dataset_analysis import *
from Assignment_3_Fx.full_desanonymisation import *
from Assignment_3_Fx.new_anonymization import *
from Assignment_3_Fx.desanonymization_evaluation import *
from Assignment_3_Fx.PII_detection import *
from Assignment_3_Fx.risk_assesment import *
from Assignment_3_Fx.text_replacement import *
from Assignment_3_Fx.utility_testing import *

if __name__ == '__main__':

    # Assignments #4-5 Anonymising Textual Data and De-Anonymisation

    # 3.1 Textual Data Anonymisation Functions :
    # First Iteration
    pii_detection(iteration=False)
    text_replacement(clean=False)
    # Second Iteration
    pii_detection(iteration=True)
    text_replacement(clean=True)

    # 3.2 De-Anonymising a dataset
    dataset_analysis()
    full_desanonymisation()
    desanonymization_evaluation()
    new_anonymization()

    # 3.3 Expriments
    utility_testing()
    risk_assesment()
