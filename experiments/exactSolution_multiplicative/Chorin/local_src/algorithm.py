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

def exact_pressure(mesh, pressure_space) -> Function:
    x, y = SpatialCoordinate(mesh)
    expr = x*x + y*y - 2.0/3.0
    return project(expr, pressure_space)

def Chorin_splitting(space_disc: SpaceDiscretisation,
                           time_grid: list[float],
                           noise_increments: list[int],
                           noise_coefficient: Function,
                           initial_condition: Function,
                           bodyforce1: Function,
                           bodyforce2: Function,
                           Reynolds_number: float = 1,
                           Lambda: float = 1) -> tuple[dict[float,Function], dict[float,Function]]:
    """Solve Stokes system with Chorin splitting. 
    
    Return 'time -> velocity' and 'time -> pressure' dictionaries. """

    Re = Constant(Reynolds_number)
    tau = Constant(1.0)
    dW = Constant(1.0)
    expW = Constant(1.0)
    t = Constant(1.0)

    u = TrialFunction(space_disc.velocity_space)
    v = TestFunction(space_disc.velocity_space)
    uold = Function(space_disc.velocity_space)
    unew = Function(space_disc.velocity_space)
    utilde = Function(space_disc.velocity_space)

    p = TrialFunction(space_disc.pressure_space)
    q = TestFunction(space_disc.pressure_space)
    pold = Function(space_disc.pressure_space)
    pnew = Function(space_disc.pressure_space)

    #setup variational form
    a1 = ( inner(u,v) + 1/Re*tau*inner(grad(u), grad(v)) )*dx
    L1 = ( inner(uold,v) - 1.0/Re*tau*inner(bodyforce1,v)*expW + 2*tau*t*inner(bodyforce2,v)*expW + Lambda*dW*inner(uold,v) )*dx

    a2 = inner(grad(p),grad(q))*dx
    L2 = 1/tau*inner(utilde,grad(q))*dx

    a3 = inner(u,v)*dx
    L3 = ( inner(utilde,v) - tau*inner(grad(pnew),v) )*dx

    V_basis = VectorSpaceBasis(constant=True)

    uold.assign(initial_condition)

    initial_time, time_increments = trajectory_to_incremets(time_grid)
    time = initial_time
    accumulatedNoise = 0
    expNoise = exp(Lambda*accumulatedNoise - Lambda**2/2.0*time)
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
    exactPressure = exact_pressure(space_disc.mesh,space_disc.pressure_space)

    solError = Function(space_disc.mixed_space)
    velError, preError = solError.subfunctions

    velError.dat.data[:] = exactVelocity.dat.data*expNoise - uold.dat.data
    preError.dat.data[:] = exactPressure.dat.data*time*expNoise - pold.dat.data

    time_to_velError[time] = deepcopy(velError)
    time_to_preError[time] = deepcopy(preError)

    for index in tqdm(range(len(time_increments))):
        dNoise = noise_increments[index]
        accumulatedNoise += dNoise
        dW.assign(dNoise)

        dtime = time_increments[index]
        time += dtime
        tau.assign(dtime)
        t.assign(time)

        expNoise = exp(Lambda*accumulatedNoise - Lambda**2/2.0*time)
        expW.assign(expNoise)
        
        solve(a1 == L1, utilde, bcs=space_disc.bcs_vel)
        solve(a2 == L2, pnew, nullspace = V_basis)
        solve(a3 == L3, unew)

        #Mean correction
        mean_p = Constant(assemble( inner(pnew,1)*dx ))
        pnew.dat.data[:] = pnew.dat.data - Function(space_disc.pressure_space).assign(mean_p).dat.data

        time_to_velocity[time] = deepcopy(unew)
        time_to_pressure[time] = deepcopy(pnew)

        uold.assign(unew)
        pold.assign(pnew)

        velError.dat.data[:] = exactVelocity.dat.data*expNoise - utilde.dat.data
        preError.dat.data[:] = exactPressure.dat.data*time*expNoise - pold.dat.data

        time_to_velError[time] = deepcopy(velError)
        time_to_preError[time] = deepcopy(preError)

    return time_to_velocity, time_to_pressure, time_to_velError, time_to_preError