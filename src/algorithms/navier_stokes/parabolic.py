from firedrake import *
from copy import deepcopy
from tqdm import tqdm
from typing import TypeAlias, Callable, Optional
import logging

from src.discretisation.time import trajectory_to_incremets
from src.discretisation.space import SpaceDiscretisation
from src.math.norms.space import l2_space
from src.algorithms.solver_configs import enable_light_monitoring, direct_solve, direct_solve_details 

### abstract structure of a Stokes algorithm
NavierStokesAlgorithm: TypeAlias = Callable[
    [SpaceDiscretisation, list[float], list[float], Function, Function, Optional[dict[float,Function]], Optional[float]],
    tuple[dict[float,Function],dict[float,Function]]
]

### converter that maps a string representation of the algorithm to its implementation
def get_algorithm_by_name(algorithm_name: str) -> NavierStokesAlgorithm:
    match algorithm_name:
        case "CN mixed FEM Strato Transport with div-sym":
            return CrankNicolson_mixedFEM_strato_transportNoise_withAntisym
        case "CN mixed FEM Strato Transport with div-sym additive":
            return CrankNicolson_mixedFEM_strato_transportNoise_withAntisym_additive
        case "CN mixed FEM Strato Transport with div-sym multiplicative":
            return CrankNicolson_mixedFEM_strato_transportNoise_withAntisym_multiplicative
        case "IE mixed FEM Strato Transport with div-sym":
            return CrankNicolson_mixedFEM_strato_transportNoise_withAntisym
        case "IE mixed FEM Strato Transport with div-sym additive":
            return CrankNicolson_mixedFEM_strato_transportNoise_withAntisym_additive
        case "IE mixed FEM Strato Transport with div-sym multiplicative":
            return CrankNicolson_mixedFEM_strato_transportNoise_withAntisym_multiplicative
        case other:
            print(f"The algorithm '{algorithm_name}' is not avaiable.")
            raise NotImplementedError

#################################################################################################
####################### below IMPLEMENTATION OF ALGORITHMS ######################################





#################################################################################################
###################################### CRANK-NICOLSON ALGORITHMS ################################
#################################################################################################

def CrankNicolson_mixedFEM_strato_transportNoise_withAntisym(space_disc: SpaceDiscretisation,
                           time_grid: list[float],
                           noise_steps: list[float], 
                           noise_coefficient: Function,
                           initial_condition: Function,
                           time_to_det_forcing: dict[float,Function] | None = None, 
                           Reynolds_number: float = 1) -> tuple[dict[float,Function], dict[float,Function]]:
    """Solve p-Stokes system with kappa regularisation and mixed finite elements. The viscous stress is given by S(A) = (kappa + |A|^2)^((p-2)/2)A.
    
    Return 'time -> velocity' and 'time -> pressure' dictionaries. """
    # initialise constants in variational form
    Re = Constant(Reynolds_number)
    tau = Constant(1.0)
    dW = Constant(1.0)

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
        + tau*( 1.0/(2.0*Re)*inner(grad(u) + grad(uold), grad(v)) - inner(p, div(v)) + 1/2.0*inner(div(u) + div(uold), q) )
        - tau*inner(det_forcing,v)
        + tau/8.0*inner(dot(grad(u) + grad(uold), u + uold), v)
        - tau/8.0*inner(dot(grad(v), u + uold), u + uold)
        - dW/4.0*inner(dot(grad(u) + grad(uold), noise_coefficient), v)
        + dW/4.0*inner(dot(grad(v), noise_coefficient), u + uold)
        )*dx

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

    #check if deterministic and random increments are iterables of the same length
    if not len(time_increments) == len(noise_steps):
        msg_error = "Time grid and noise grid are not of the same length.\n"
        msg_error += f"Time grid length: \t {len(time_increments)}\n"
        msg_error += f"Noise grid length: \t {len(noise_steps)}"
        raise ValueError(msg_error)

    for index in tqdm(range(len(time_increments))):
        # update random and deterministc time step, and nodal time
        dW.assign(noise_steps[index])
        #dW.assign(time_increments[index])
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

def CrankNicolson_mixedFEM_strato_transportNoise_withAntisym_additive(space_disc: SpaceDiscretisation,
                           time_grid: list[float],
                           noise_steps: list[float], 
                           noise_coefficient: Function,
                           initial_condition: Function,
                           time_to_det_forcing: dict[float,Function] | None = None, 
                           Reynolds_number: float = 1) -> tuple[dict[float,Function], dict[float,Function]]:
    """Solve p-Stokes system with kappa regularisation and mixed finite elements. The viscous stress is given by S(A) = (kappa + |A|^2)^((p-2)/2)A.
    
    Return 'time -> velocity' and 'time -> pressure' dictionaries. """
    # initialise constants in variational form
    Re = Constant(Reynolds_number)
    tau = Constant(1.0)
    dW = Constant(1.0)

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
        + tau*( 1.0/(2.0*Re)*inner(grad(u) + grad(uold), grad(v)) - inner(p, div(v)) + 1/2.0*inner(div(u) + div(uold), q) )
        - tau*inner(det_forcing,v)
        + tau/8.0*inner(dot(grad(u) + grad(uold), u + uold), v)
        - tau/8.0*inner(dot(grad(v), u + uold), u + uold)
        - dW*inner(noise_coefficient, v)
        )*dx

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

    #check if deterministic and random increments are iterables of the same length
    if not len(time_increments) == len(noise_steps):
        msg_error = "Time grid and noise grid are not of the same length.\n"
        msg_error += f"Time grid length: \t {len(time_increments)}\n"
        msg_error += f"Noise grid length: \t {len(noise_steps)}"
        raise ValueError(msg_error)

    for index in tqdm(range(len(time_increments))):
        # update random and deterministc time step, and nodal time
        dW.assign(noise_steps[index])
        #dW.assign(time_increments[index])
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

def CrankNicolson_mixedFEM_strato_transportNoise_withAntisym_multiplicative(space_disc: SpaceDiscretisation,
                           time_grid: list[float],
                           noise_steps: list[float], 
                           noise_coefficient: Function,
                           initial_condition: Function,
                           time_to_det_forcing: dict[float,Function] | None = None, 
                           Reynolds_number: float = 1) -> tuple[dict[float,Function], dict[float,Function]]:
    """Solve p-Stokes system with kappa regularisation and mixed finite elements. The viscous stress is given by S(A) = (kappa + |A|^2)^((p-2)/2)A.
    
    Return 'time -> velocity' and 'time -> pressure' dictionaries. """
    # initialise constants in variational form
    Re = Constant(Reynolds_number)
    tau = Constant(1.0)
    dW = Constant(1.0)

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
        + tau*( 1.0/(2.0*Re)*inner(grad(u) + grad(uold), grad(v)) - inner(p, div(v)) + 1/2.0*inner(div(u) + div(uold), q) )
        - tau*inner(det_forcing,v)
        + tau/8.0*inner(dot(grad(u) + grad(uold), u + uold), v)
        - tau/8.0*inner(dot(grad(v), u + uold), u + uold)
        - l2_space(noise_coefficient)*dW/2.0*inner(u+uold, v)
        )*dx

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

    #check if deterministic and random increments are iterables of the same length
    if not len(time_increments) == len(noise_steps):
        msg_error = "Time grid and noise grid are not of the same length.\n"
        msg_error += f"Time grid length: \t {len(time_increments)}\n"
        msg_error += f"Noise grid length: \t {len(noise_steps)}"
        raise ValueError(msg_error)

    for index in tqdm(range(len(time_increments))):
        # update random and deterministc time step, and nodal time
        dW.assign(noise_steps[index])
        #dW.assign(time_increments[index])
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


#################################################################################################
###################################### IMPLICIT ALGORITHMS ######################################
#################################################################################################

def implicit_mixedFEM_strato_transportNoise_withAntisym(space_disc: SpaceDiscretisation,
                           time_grid: list[float],
                           noise_steps: list[float], 
                           noise_coefficient: Function,
                           initial_condition: Function,
                           time_to_det_forcing: dict[float,Function] | None = None, 
                           Reynolds_number: float = 1) -> tuple[dict[float,Function], dict[float,Function]]:
    """Solve p-Stokes system with kappa regularisation and mixed finite elements. The viscous stress is given by S(A) = (kappa + |A|^2)^((p-2)/2)A.
    
    Return 'time -> velocity' and 'time -> pressure' dictionaries. """
    # initialise constants in variational form
    Re = Constant(Reynolds_number)
    tau = Constant(1.0)
    dW = Constant(1.0)

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
        + tau*( 1.0/(Re)*inner(grad(u), grad(v)) - inner(p, div(v)) + 1/2.0*inner(div(u) + div(uold), q) )
        - tau*inner(det_forcing,v)
        + tau/8.0*inner(dot(grad(u) + grad(uold), u + uold), v)
        - tau/8.0*inner(dot(grad(v), u + uold), u + uold)
        - dW/4.0*inner(dot(grad(u) + grad(uold), noise_coefficient), v)
        + dW/4.0*inner(dot(grad(v), noise_coefficient), u + uold)
        )*dx

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

    #check if deterministic and random increments are iterables of the same length
    if not len(time_increments) == len(noise_steps):
        msg_error = "Time grid and noise grid are not of the same length.\n"
        msg_error += f"Time grid length: \t {len(time_increments)}\n"
        msg_error += f"Noise grid length: \t {len(noise_steps)}"
        raise ValueError(msg_error)

    for index in tqdm(range(len(time_increments))):
        # update random and deterministc time step, and nodal time
        dW.assign(noise_steps[index])
        #dW.assign(time_increments[index])
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

def implicit_mixedFEM_strato_transportNoise_withAntisym_additive(space_disc: SpaceDiscretisation,
                           time_grid: list[float],
                           noise_steps: list[float], 
                           noise_coefficient: Function,
                           initial_condition: Function,
                           time_to_det_forcing: dict[float,Function] | None = None, 
                           Reynolds_number: float = 1) -> tuple[dict[float,Function], dict[float,Function]]:
    """Solve p-Stokes system with kappa regularisation and mixed finite elements. The viscous stress is given by S(A) = (kappa + |A|^2)^((p-2)/2)A.
    
    Return 'time -> velocity' and 'time -> pressure' dictionaries. """
    # initialise constants in variational form
    Re = Constant(Reynolds_number)
    tau = Constant(1.0)
    dW = Constant(1.0)

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
        + tau*( 1.0/(Re)*inner(grad(u), grad(v)) - inner(p, div(v)) + 1/2.0*inner(div(u) + div(uold), q) )
        - tau*inner(det_forcing,v)
        + tau/8.0*inner(dot(grad(u) + grad(uold), u + uold), v)
        - tau/8.0*inner(dot(grad(v), u + uold), u + uold)
        - dW*inner(noise_coefficient, v)
        )*dx

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

    #check if deterministic and random increments are iterables of the same length
    if not len(time_increments) == len(noise_steps):
        msg_error = "Time grid and noise grid are not of the same length.\n"
        msg_error += f"Time grid length: \t {len(time_increments)}\n"
        msg_error += f"Noise grid length: \t {len(noise_steps)}"
        raise ValueError(msg_error)

    for index in tqdm(range(len(time_increments))):
        # update random and deterministc time step, and nodal time
        dW.assign(noise_steps[index])
        #dW.assign(time_increments[index])
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

def implicit_mixedFEM_strato_transportNoise_withAntisym_multiplicative(space_disc: SpaceDiscretisation,
                           time_grid: list[float],
                           noise_steps: list[float], 
                           noise_coefficient: Function,
                           initial_condition: Function,
                           time_to_det_forcing: dict[float,Function] | None = None, 
                           Reynolds_number: float = 1) -> tuple[dict[float,Function], dict[float,Function]]:
    """Solve p-Stokes system with kappa regularisation and mixed finite elements. The viscous stress is given by S(A) = (kappa + |A|^2)^((p-2)/2)A.
    
    Return 'time -> velocity' and 'time -> pressure' dictionaries. """
    # initialise constants in variational form
    Re = Constant(Reynolds_number)
    tau = Constant(1.0)
    dW = Constant(1.0)

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
        + tau*( 1.0/(Re)*inner(grad(u) + grad(uold), grad(v)) - inner(p, div(v)) + 1/2.0*inner(div(u) + div(uold), q) )
        - tau*inner(det_forcing,v)
        + tau/8.0*inner(dot(grad(u) + grad(uold), u + uold), v)
        - tau/8.0*inner(dot(grad(v), u + uold), u + uold)
        - l2_space(noise_coefficient)*dW/2.0*inner(u+uold, v)
        )*dx

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

    #check if deterministic and random increments are iterables of the same length
    if not len(time_increments) == len(noise_steps):
        msg_error = "Time grid and noise grid are not of the same length.\n"
        msg_error += f"Time grid length: \t {len(time_increments)}\n"
        msg_error += f"Noise grid length: \t {len(noise_steps)}"
        raise ValueError(msg_error)

    for index in tqdm(range(len(time_increments))):
        # update random and deterministc time step, and nodal time
        dW.assign(noise_steps[index])
        #dW.assign(time_increments[index])
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