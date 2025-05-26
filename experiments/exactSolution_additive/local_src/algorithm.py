from firedrake import *
from copy import deepcopy
from tqdm import tqdm

from src.discretisation.time import trajectory_to_incremets
from src.discretisation.space import SpaceDiscretisation

def implicitEuler_mixedFEM(space_disc: SpaceDiscretisation,
                           time_grid: list[float],
                           noise_increments: list[int],
                           noise_coefficient: Function,
                           initial_condition: Function,
                           bodyforce1: Function,
                           bodyforce2: Function,
                           Reynolds_number: float = 1) -> tuple[dict[float,Function], dict[float,Function]]:
    """Solve Stokes system with mixed finite elements for multiplicative noise. 
    
    Return 'time -> velocity' and 'time -> pressure' dictionaries. """

    u, p = TrialFunctions(space_disc.mixed_space)
    v, q = TestFunctions(space_disc.mixed_space)

    Re = Constant(Reynolds_number)
    tau = Constant(1.0)
    dW = Constant(1.0)
    W = Constant(1.0)
    t = Constant(1.0)

    upold = Function(space_disc.mixed_space)
    uold, pold = upold.subfunctions

    uold.assign(initial_condition)

    a = ( inner(u,v) + tau*( 1.0/Re*inner(grad(u), grad(v)) - inner(p, div(v)) + inner(div(u), q) ) )*dx
    L = ( inner(uold,v) - 1.0/Re*tau*inner(bodyforce1,v)*(1+W) + 2*tau*t*inner(bodyforce2,v) + dW*inner(noise_coefficient,v) )*dx

    up = Function(space_disc.mixed_space)
    u, p = up.subfunctions

    initial_time, time_increments = trajectory_to_incremets(time_grid)
    time = initial_time
    accumulatedNoise = 0

 
    time_to_velocity = dict()
    time_to_pressure = dict()

    time_to_velocity[time] = deepcopy(uold)
    time_to_pressure[time] = deepcopy(pold)

    for index in tqdm(range(len(time_increments))):
        accumulatedNoise += noise_increments[index]
        W.assign(accumulatedNoise)
        dW.assign(noise_increments[index])
        tau.assign(time_increments[index])
        time += time_increments[index]
        t.assign(time)
            
        solve(a == L, up, bcs=space_disc.bcs_mixed, nullspace=space_disc.null)

        #Mean correction
        mean_p = Constant(assemble( inner(p,1)*dx ))
        p.dat.data[:] = p.dat.data - Function(space_disc.pressure_space).assign(mean_p).dat.data

        time_to_velocity[time] = deepcopy(u)
        time_to_pressure[time] = deepcopy(p)

        upold.assign(up)

    return time_to_velocity, time_to_pressure