from firedrake import *
from copy import deepcopy
from tqdm import tqdm

from src.discretisation.time import trajectory_to_incremets
from src.discretisation.space import SpaceDiscretisation

def _poly(x,y):
    return 2*x*x*(1-x)*(1-x)*y*(y-1)*(2*y-1)

def exact_velocity(mesh, velocity_space) -> Function:
    x, y = SpatialCoordinate(mesh)
    expr = as_vector([
        _poly(x,y),
        -1*_poly(y,x)
        ])
    return project(expr, velocity_space)

def exact_pressure_det(mesh, pressure_space) -> Function:
    x, y = SpatialCoordinate(mesh)
    expr = x*x + y*y - 2.0/3.0
    return project(expr, pressure_space)

def exact_pressure_sto(mesh, pressure_space) -> Function:
    x, y = SpatialCoordinate(mesh)
    expr = x*x*(1-x)*(1-x)*y*y*(1-y)*(1-y) - 1.0/900.0
    return project(expr, pressure_space)

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
    L = ( inner(uold,v) - 1.0/Re*tau*inner(bodyforce1,v) + 2*tau*t*inner(bodyforce2,v) + dW*inner(noise_coefficient,v) )*dx

    up = Function(space_disc.mixed_space)
    u, p = up.subfunctions

    initial_time, time_increments = trajectory_to_incremets(time_grid)
    time = initial_time
    accumulatedNoise = 0
    dNoise = 0

    #approximate solution
    time_to_velocity = dict()
    time_to_pressure = dict()

    time_to_velocity[time] = deepcopy(uold)
    time_to_pressure[time] = deepcopy(pold)

    #handling of exact solution
    time_to_velError = dict()
    time_to_preError = dict()

    exactVelocity = exact_velocity(space_disc.mesh,space_disc.velocity_space)
    exactPressureDet = exact_pressure_det(space_disc.mesh,space_disc.pressure_space)
    exactPressureSto = exact_pressure_sto(space_disc.mesh,space_disc.pressure_space)

    solError = Function(space_disc.mixed_space)
    velError, preError = solError.subfunctions

    velError.dat.data[:] = exactVelocity.dat.data*(1+accumulatedNoise) - uold.dat.data
    preError.dat.data[:] = exactPressureDet.dat.data*time + exactPressureSto.dat.data*dNoise - pold.dat.data

    time_to_velError[time] = deepcopy(velError)
    time_to_preError[time] = deepcopy(preError)

    for index in tqdm(range(len(time_increments))):
        dNoise = noise_increments[index]
        accumulatedNoise += dNoise
        W.assign(accumulatedNoise)
        dW.assign(dNoise)

        dtime = time_increments[index]
        time += dtime
        tau.assign(dtime)
        t.assign(time)
            
        solve(a == L, up, bcs=space_disc.bcs_mixed, nullspace=space_disc.null)

        #Mean correction
        mean_p = Constant(assemble( inner(p,1)*dx ))
        p.dat.data[:] = p.dat.data - Function(space_disc.pressure_space).assign(mean_p).dat.data

        time_to_velocity[time] = deepcopy(u)
        time_to_pressure[time] = deepcopy(p)

        upold.assign(up)

        velError.dat.data[:] = exactVelocity.dat.data*(1+accumulatedNoise) - uold.dat.data
        preError.dat.data[:] = exactPressureDet.dat.data*time + exactPressureSto.dat.data*(dNoise/dtime) - pold.dat.data

        time_to_velError[time] = deepcopy(velError)
        time_to_preError[time] = deepcopy(preError)

    return time_to_velocity, time_to_pressure, time_to_velError, time_to_preError