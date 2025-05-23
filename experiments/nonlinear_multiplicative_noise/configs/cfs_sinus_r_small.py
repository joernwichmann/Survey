"""Contains local parameter configuration."""
### Experimentname
NAME_EXPERIMENT: str = "sinus_r_small"

### Algorithm
ALGORITHM_NAME: str =  "Implicit Euler mixed FEM" #see src.algorithms.select.py for available choices

TRUNCATION_INDEX_NOISE: int = 6
R_VALUE_NOISE: float = 0.1

################               FILE/DIRECTORY NAMES               ############################
#Log
NAME_LOGFILE_GENERATE: str = f"{NAME_EXPERIMENT}.log"

#Vtk
VTK_DIRECTORY: str = f"vtk"

#Convergence
TIME_DIRECTORYNAME: str = f"convergence_results/{NAME_EXPERIMENT}"

#Stability
STABILITY_DIRECTORYNAME: str = f"stability_results/{NAME_EXPERIMENT}"

#Energy
ENERGY_DIRECTORYNAME: str = f"energy_results/{NAME_EXPERIMENT}"

#Statistics
STATISTICS_DIRECTORYNAME: str = f"statistic_results/{NAME_EXPERIMENT}"

#Point statistics
POINT_STATISTICS_DIRECTORYNAME: str = f"point_statistic_results/{NAME_EXPERIMENT}"

#Increment check
INCREMENT_DIRECTORYNAME: str = f"increment_results/{NAME_EXPERIMENT}"
