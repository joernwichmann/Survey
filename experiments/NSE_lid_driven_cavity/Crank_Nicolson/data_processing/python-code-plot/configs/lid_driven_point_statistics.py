'''Configs of all plots'''

EXPERIMENT_NAME = "nse-lid-driven"
EXPERIMENTS = {1: f"{EXPERIMENT_NAME}_additive", 2: f"{EXPERIMENT_NAME}_multiplicative", 3: f"{EXPERIMENT_NAME}_transport" }

#### matching experiment to its p-value
P_VALUE = {1: "additive", 2: "multiplicative", 3: "transport"}

#### locations 
POINT = "p1"
ROOT_LOCATION = "../../point_statistic_results/"
DET_LOCATION = f"/deterministic/{POINT}/mean"
MEAN_LOCATION = f"/{POINT}/mean"
IND_LOCATION = f"/{POINT}/seed"
DATA_SOURCE = "/refinement_9.csv"

#### output 
OUTPUT_LOCATION = f"output/{EXPERIMENT_NAME}/point_statistics/{POINT}/"

#### stochatic
NUMBER_SAMPLES = 100

#### stationary
#STATIONARY_TIME = {1: 0.13, 2: 0.32, 3: 0.13}
STATIONARY_TIME = {1: 19.0, 2: 19.0, 3: 19.0}

#### plotting configs
# colours
COLOURS_MEAN = {1: "#1b9e77", 2: "#d95f02", 3: "#7570b3"}
COLOURS_INDIVIDUAL  = {1: "#66c2a5", 2: "#fc8d62", 3: "#8da0cb"}
BLACK = "#000000"

# histogram plot
HIST_DPI = 300
HIST_FILEFORMAT = "pdf"

YMAX =  {1: 0.06, 2: 0.016, 3: 0.014}
LINEAR_PLOT = True
LOG_PLOT = False

# trajectory plot
TRAJ_DPI = 300
TRAJ_FILEFORMAT = "png"

LINEWIDTH_MEAN = 2
LINEOPACITY_MEAN = 1

LINEWIDTH_SD = 1.5
LINEOPACITY_SD = 1
LINESTYLE_SD = (0, (1, 1))

LINEWIDTH_INDIVIDUAL = 0.1
LINEOPACITY_INDIVIDUAL = 1

LABEL_FONTSIZE = 15
TICK_FONTSIZE = 15

LINESTYLES_DET = {1: "dotted", 2: "dashed", 3: "dashdot"}

