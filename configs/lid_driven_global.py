"""Contains global parameter configuration."""
################               GLOBAL configs               ############################
LOG_LEVEL: str = "info"  #supported levels: debug, info, warning, error, critical 

### Dump-location
DUMP_LOCATION: str = "sample_dump"

################               GENERATE configs               ############################
### Model
MODEL_NAME: str = "p-Stokes" #see src.algorithms.select.py for available choices
KAPPA_VALUE: float = 0.1

# Deterministic forcing
FORCING: str = "zero"   #see 'src.predefined_data' for available choices
FORCING_FREQUENZY_X: int = 2
FORCING_FREQUENZY_Y: int = 4
FORCING_INTENSITY: float = 100

# Time
INITIAL_TIME: float = 0
END_TIME: float = 1
REFINEMENT_LEVELS: list[int] = list(range(7,10))

# Initial data
INITIAL_CONDITION_NAME: str = "zero"    #see 'src.predefined_data' for available choices
INITIAL_FREQUENZY_X: int = 2
INITIAL_FREQUENZY_Y: int = 4
INITIAL_INTENSITY: float = 1000

# Elements
VELOCITY_ELEMENT: str = "CG"    #see firedrake doc for available spaces
VELOCITY_DEGREE: int = 2       

PRESSURE_ELEMENT: str = "CG"    #see firedrake doc for available spaces
PRESSURE_DEGREE: int = 1

# Mesh
NUMBER_SPACE_POINTS: int = 12
MESH_NAME: str = "unit square"  #see 'src.discretisation.mesh' for available choices

# Boundary conditions # WARNING: DONT USE BOTH, IMPLICIT AND EXPLICIT BCs SIMULTANEOULSY
NAME_BOUNDARY_CONDITION: str = "zero"  #see 'src.discretisation.mesh' for available choices ### imposes BC implicitly via firedrake blackbox
BOUNDARY_CONDITION_EXPLICIT_NAME: str = "lid-driven-strong"
BOUNDARY_CONDITION_EXPLICIT_INTENSITY: float = 1

# Monte Carlo
MC_SAMPLES: int = 1000
NOISE_INCREMENTS: str = "classical" # see 'src.noise' for available choices

# Noise coefficient
NOISE_INTENSITY: float = 1000
NOISE_COEFFICIENT_NAME: str = "polynomial" #see 'src.predefined_data' for available choices
NOISE_FREQUENZY_X: int = 2
NOISE_FREQUENZY_Y: int = 4

################               ANALYSE configs               ############################
#Convergence
TIME_CONVERGENCE: bool = False
TIME_COMPARISON_TYPE: str = "absolute"       ## "absolute" and "relative" are supported

#Stability
STABILITY_CHECK: bool = False

#Mean energy
ENERGY_CHECK: bool = True

#Individual energy
IND_ENERGY_CHECK: bool = True
IND_ENERGY_NUMBER: int = 100

#Statistics
STATISTICS_CHECK: bool = True

#Point statistics
POINT_STATISTICS_CHECK: bool = True
POINT: list[float] = [1/2.0,3/4.0]
IND_POINT_STATISTICS_CHECK_NUMBER: int = 100

#Increment check
INCREMENT_CHECK: bool = True

