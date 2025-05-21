'''Configs of all plots'''
EXPERIMENT_NAME = "p-variation"
EXPERIMENTS = {1: f"{EXPERIMENT_NAME}_exp1", 2: f"{EXPERIMENT_NAME}_exp2", 3: f"{EXPERIMENT_NAME}_exp3" }

#### matching experiment to its p-value
P_VALUE = {1: 3/2.0, 2: 2, 3: 3}

#### locations 
ROOT_LOCATION = "../../increment_results/"
DET_LOCATION = "/deterministic"
MEAN_LOCATION = ""
DATA_SOURCE = "/L2-inc.csv"

#### output 
OUTPUT_LOCATION = f"output/{EXPERIMENT_NAME}/increments/"

#### stochatic
NUMBER_SAMPLES = 100

### EOC
EOC_AT_TIME = 1