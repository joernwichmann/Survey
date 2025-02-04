from firedrake import *
from copy import deepcopy
from tqdm import tqdm
from typing import TypeAlias, Callable, Optional
import logging

from src.discretisation.time import trajectory_to_incremets
from src.discretisation.space import SpaceDiscretisation
from src.algorithms.nonlinearities import S_tensor, S_tensor_sym, epsilon
from src.algorithms.solver_configs import enable_monitoring, direct_solve_details, direct_solve 

### abstract structure of a p-Stokes algorithm
p_NSE_Algorithm: TypeAlias = Callable[
    [SpaceDiscretisation, list[float], list[float], Function, Function, Optional[float], Optional[float], Optional[dict[float,Function]], Optional[float]],
    tuple[dict[float,Function],dict[float,Function],dict[float,Function],dict[float,Function]]
]

### converter that maps a string representation of the algorithm to its implementation   
def get_algorithm_by_name(algorithm_name: str) -> p_NSE_Algorithm:
    match algorithm_name:
        case "Crank Nicolson mixed FEM Stratonovich Transport Noise with anti-symmetrisation":
            return CrankNicolson_mixedFEM_strato_transportNoise_withAntisym
        case "lid-driven cavity solver":
            return lid_driven_cavity_solver
        case other:
            print(f"The algorithm '{algorithm_name}' is not avaiable.")
            raise NotImplementedError

### implementations of abstract structure
def CrankNicolson_mixedFEM_strato_transportNoise_withAntisym(space_disc: SpaceDiscretisation,
                           time_grid: list[float], 
                           noise_coeff_to_noise_increments: dict[Function,list[int]],
                           initial_condition: Function,
                           p_value: float = 2.0,
                           kappa_value: float = 0.1,
                           time_to_det_forcing: dict[float,Function] | None = None, 
                           Reynolds_number: float = 1) -> tuple[dict[float,Function],dict[float,Function],dict[float,Function],dict[float,Function]]:
    """Solve p-Stokes system with kappa regularisation and mixed finite elements. The viscous stress is given by S(A) = (kappa + |A|^2)^((p-2)/2)A.
    
    Return 'time -> velocity' and 'time -> pressure' dictionaries. """
    # initialise constants in variational form
    Re = Constant(Reynolds_number)
    tau = Constant(1.0)
    noise_coeff_to_dW = {noise_coeff: Constant(1.0) for noise_coeff in noise_coeff_to_noise_increments}

    # initialise function objects
    v, q = TestFunctions(space_disc.mixed_space)

    up = Function(space_disc.mixed_space)
    u, p = split(up)    # split types: <class 'ufl.tensors.ListTensor'> and <class 'ufl.indexed.Indexed'> needed for nonlinear solver
    velocity, pressure = up.subfunctions    #subfunction types: <class 'firedrake.function.Function'> and <class 'firedrake.function.Function'>

    upold = Function(space_disc.mixed_space)
    uold, pold = upold.subfunctions

    # initialise deterministic forcing by zero as default 
    det_forcing, _ = Function(space_disc.mixed_space).subfunctions

    # set initial condition to uold
    uold.assign(initial_condition)

    # build variational form
    VariationalForm = ( 
        inner(u - uold,v) 
        + tau*( 1.0/Re*inner( S_tensor_sym((grad(u) + grad(uold))/2.0,p_value,kappa_value), epsilon(grad(v))) - inner(p, div(v)) + inner(div(u), q) )
        + tau/4.0*( inner(dot(grad(u) + grad(uold), u + uold), v) - inner(dot(grad(v), u + uold),u + uold) )
        - tau*inner(det_forcing,v)
        )*dx
    
    for noise_coeff in noise_coeff_to_noise_increments:
        VariationalForm = VariationalForm - noise_coeff_to_dW[noise_coeff]/4.0*inner(dot(grad(u) + grad(uold), noise_coeff), v)*dx
        VariationalForm = VariationalForm + noise_coeff_to_dW[noise_coeff]/4.0*inner(dot(grad(v), noise_coeff), u + uold)*dx

    # setup initial time and time increments
    initial_time, time_increments = trajectory_to_incremets(time_grid)
    time = initial_time

    # initialise storage for solution output
    time_to_velocity = dict()
    time_to_pressure = dict()
    time_to_velocity_midpoints = dict()
    time_to_pressure_midpoints = dict()

    # store initialisation of time-stepping
    time_to_velocity[time] = deepcopy(uold)
    time_to_velocity_midpoints[time] = deepcopy(det_forcing)
    time_to_pressure[time] = deepcopy(pold)
    time_to_pressure_midpoints[time] = deepcopy(pold)

    for index in tqdm(range(len(time_increments))):
        # update random and deterministc time step, and nodal time
        for noise_coeff in noise_coeff_to_noise_increments:
            noise_coeff_to_dW[noise_coeff].assign(noise_coeff_to_noise_increments[noise_coeff][index])
        tau.assign(time_increments[index])
        time += time_increments[index]
        
        # if provided change deterministic forcing to provided one
        if time_to_det_forcing:
            try:
                det_forcing.assign(time_to_det_forcing[time])
            except KeyError as k:
                print(f"Deterministic forcing couldn't be set.\nRequested time:\t {time}\nAvailable times:\t {list(time_to_det_forcing.keys())}")
                raise k
        
        #try solve nonlinear problem by using firedrake blackbox. If default solve doesn't converge, restart solve with enbabled montoring to see why it fails. 
        try:   
            solve(VariationalForm == 0, up, bcs=space_disc.bcs_mixed, nullspace=space_disc.null, solver_parameters=direct_solve)
        except ConvergenceError as e:
            logging.exception(e)
            solve(VariationalForm == 0, up, bcs=space_disc.bcs_mixed, nullspace=space_disc.null, solver_parameters=direct_solve_details)

        #correct mean-value of pressure
        mean_p = Constant(assemble( inner(p,1)*dx ))
        pressure.dat.data[:] = pressure.dat.data - Function(space_disc.pressure_space).assign(mean_p).dat.data

        #store solution
        time_to_velocity[time] = deepcopy(velocity)
        velocity_mid = Function(space_disc.velocity_space)
        velocity_mid.dat.data[:] = (velocity.dat.data + uold.dat.data)/2.0
        time_to_velocity_midpoints[time] = deepcopy(velocity_mid)
        time_to_pressure[time] = deepcopy(pressure)
        pressure_mid = Function(space_disc.pressure_space)
        pressure_mid.dat.data[:] = (pressure.dat.data + pold.dat.data)/2.0
        time_to_pressure_midpoints[time] = pressure_mid

        #update uold to proceed time-steppping
        uold.assign(velocity)
        pold.assign(pressure)

    return time_to_velocity, time_to_pressure, time_to_velocity_midpoints, time_to_pressure_midpoints

### CAREFUL: lid driven solver additionally gets boundary conditions as input
def lid_driven_cavity_solver(space_disc: SpaceDiscretisation,
                           time_grid: list[float],
                           noise_coeff_to_noise_increments: dict[Function,list[int]],
                           initial_velocity: Function,
                           initial_pressure: Function,
                           boundary_condition: Function,
                           p_value: float = 2.0,
                           kappa_value: float = 0.1,
                           time_to_det_forcing: dict[float,Function] | None = None, 
                           Reynolds_number: float = 1) -> tuple[dict[float,Function],dict[float,Function],dict[float,Function],dict[float,Function]]:
    """Solve p-Stokes system with kappa regularisation and mixed finite elements. The viscous stress is given by S(A) = (kappa + |A|^2)^((p-2)/2)A.
    
    Return 'time -> velocity' and 'time -> pressure' dictionaries. """
    # initialise constants in variational form
    Re = Constant(Reynolds_number)
    tau = Constant(1.0)
    noise_coeff_to_dW = {noise_coeff: Constant(1.0) for noise_coeff in noise_coeff_to_noise_increments}

    # initialise function objects
    v, q = TestFunctions(space_disc.mixed_space)

    up = Function(space_disc.mixed_space)
    u, p = split(up)    # split types: <class 'ufl.tensors.ListTensor'> and <class 'ufl.indexed.Indexed'> needed for nonlinear solver
    velocity, pressure = up.subfunctions    #subfunction types: <class 'firedrake.function.Function'> and <class 'firedrake.function.Function'>

    upold = Function(space_disc.mixed_space)
    uold, pold = upold.subfunctions

    # initialise deterministic forcing by zero as default 
    det_forcing, _ = Function(space_disc.mixed_space).subfunctions

    # set initial conditions
    uold.assign(initial_velocity-boundary_condition)
    pold.assign(initial_pressure)

    # build variational form
    VariationalForm = ( 
        inner(u - uold,v) 
        + tau*( 1.0/Re*inner( S_tensor_sym((grad(u) + grad(uold))/2.0 + grad(boundary_condition),p_value,kappa_value), epsilon(grad(v))) )
        + tau/4.0*( inner(dot(grad(u) + grad(uold) + grad(boundary_condition), u + uold + boundary_condition), v) - inner(dot(grad(v), u + uold + boundary_condition),u + uold + boundary_condition) )
        - inner(p - pold, div(v)) + inner(div(u) - div(boundary_condition), q)
        - tau*inner(det_forcing,v)
        )*dx
    
    for noise_coeff in noise_coeff_to_noise_increments:
        VariationalForm = VariationalForm - noise_coeff_to_dW[noise_coeff]/4.0*inner(dot(grad(u) + grad(uold), noise_coeff), v)*dx
        VariationalForm = VariationalForm + noise_coeff_to_dW[noise_coeff]/4.0*inner(dot(grad(v), noise_coeff), u + uold)*dx
        VariationalForm = VariationalForm - noise_coeff_to_dW[noise_coeff]*inner(dot(grad(boundary_condition), noise_coeff), v)*dx

    # setup initial time and time increments
    initial_time, time_increments = trajectory_to_incremets(time_grid)
    time = initial_time

    # initialise storage for solution output
    time_to_velocity = dict()
    time_to_pressure = dict()
    time_to_velocity_midpoints = dict()
    time_to_pressure_midpoints = dict()

    # store initialisation of time-stepping
    time_to_velocity[time] = deepcopy(Function(space_disc.velocity_space))
    time_to_velocity_midpoints[time] = deepcopy(det_forcing)
    time_to_pressure[time] = deepcopy(pold)
    time_to_pressure_midpoints[time] = deepcopy(pold)

    for index in tqdm(range(len(time_increments))):
        # update random and deterministc time step, and nodal time
        for noise_coeff in noise_coeff_to_noise_increments:
            noise_coeff_to_dW[noise_coeff].assign(noise_coeff_to_noise_increments[noise_coeff][index])
        tau.assign(time_increments[index])
        time += time_increments[index]
        
        # if provided change deterministic forcing to provided one
        if time_to_det_forcing:
            try:
                det_forcing.assign(time_to_det_forcing[time])
            except KeyError as k:
                print(f"Deterministic forcing couldn't be set.\nRequested time:\t {time}\nAvailable times:\t {list(time_to_det_forcing.keys())}")
                raise k
        
        #try solve nonlinear problem by using firedrake blackbox. If default solve doesn't converge, restart solve with enbabled montoring to see why it fails. 
        try:   
            solve(VariationalForm == 0, up, bcs=space_disc.bcs_mixed, nullspace=space_disc.null, solver_parameters=direct_solve)
        except ConvergenceError as e:
            logging.exception(e)
            solve(VariationalForm == 0, up, bcs=space_disc.bcs_mixed, nullspace=space_disc.null, solver_parameters=direct_solve_details)

        #correct mean-value of pressure
        mean_p = Constant(assemble( inner(p,1)*dx ))
        pressure.dat.data[:] = pressure.dat.data - Function(space_disc.pressure_space).assign(mean_p).dat.data

        #store solution
        velocity_nodal = Function(space_disc.velocity_space)
        velocity_nodal.dat.data[:] = velocity.dat.data + boundary_condition.dat.data
        time_to_velocity[time] = deepcopy(velocity_nodal)

        velocity_mid = Function(space_disc.velocity_space)
        velocity_mid.dat.data[:] = (velocity.dat.data + uold.dat.data)/2.0 + boundary_condition.dat.data
        time_to_velocity_midpoints[time] = deepcopy(velocity_mid)

        time_to_pressure[time] = deepcopy(pressure)
        pressure_mid = Function(space_disc.pressure_space)
        pressure_mid.dat.data[:] = (pressure.dat.data + pold.dat.data)/2.0
        time_to_pressure_midpoints[time] = pressure_mid

        #update uold to proceed time-steppping
        uold.assign(velocity)
        pold.assign(pressure)

    return time_to_velocity, time_to_pressure, time_to_velocity_midpoints, time_to_pressure_midpoints