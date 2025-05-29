"""Contains global parameter configuration."""
################               GLOBAL configs               ############################
LOG_LEVEL: str = "info"  #supported levels: debug, info, warning, error, critical 

################               GENERATE configs               ############################
### Model
MODEL_NAME: str = "Stokes" #see src.algorithms.select.py for available choices
REYNOLDS_NUMBER: float = 1

GAMMA: float = 1    #Hoelder index of time-regularity of pressure

# Time
INITIAL_TIME: float = 0
END_TIME: float = 1
REFINEMENT_LEVELS: list[int] = list(range(3,10))

# Elements
VELOCITY_ELEMENT: str = "CG"    #see firedrake doc for available spaces
VELOCITY_DEGREE: int = 2       

PRESSURE_ELEMENT: str = "CG"    #see firedrake doc for available spaces
PRESSURE_DEGREE: int = 1

# Mesh
MESH_NAME: str = "unit square"  #see 'src.discretisation.mesh' for available choices

# Monte Carlo
MC_SAMPLES: int = 100
NOISE_INCREMENTS: str = "classical" # see 'src.noise' for available choices


################               ANALYSE configs               ############################
#Convergence
TIME_CONVERGENCE: bool = True
TIME_COMPARISON_TYPE: str = "absolute"       ## "absolute" and "relative" are supported

#Stability
STABILITY_CHECK: bool = True

#Mean energy
ENERGY_CHECK: bool = True

#Individual energy
IND_ENERGY_CHECK: bool = False
IND_ENERGY_NUMBER: int = 100

#Statistics
STATISTICS_CHECK: bool = True

#Point statistics
POINT_STATISTICS_CHECK: bool = False
POINT_1: list[float] = [1/2.0,3/4.0]
POINT_2: list[float] = [1/8.0,1/8.0]
POINT_3: list[float] = [7/8.0,7/8.0]
IND_POINT_STATISTICS_CHECK_NUMBER: int = 100

#Increment check
INCREMENT_CHECK: bool = False

