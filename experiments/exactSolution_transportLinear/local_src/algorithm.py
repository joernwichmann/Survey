from firedrake import *
from copy import deepcopy
from tqdm import tqdm

from src.discretisation.time import trajectory_to_incremets
from src.discretisation.space import SpaceDiscretisation

from local_src.transport import initialCondition, transformation, noiseCoefficient, exact_pressure, exact_velocity, bodyforce1, bodyforce2


def implicitEuler_mixedFEM(space_disc: SpaceDiscretisation,
                           time_grid: list[float],
                           noise_increments: list[int],
                           Reynolds_number: float = 1,
                           Lambda: float = 0.1) -> tuple[dict[float,Function], dict[float,Function]]:
    """Solve Stokes system with mixed finite elements for multiplicative noise. 
    
    Return 'time -> velocity' and 'time -> pressure' dictionaries. """

    u, p = TrialFunctions(space_disc.mixed_space)
    v, q = TestFunctions(space_disc.mixed_space)

    Re = Constant(Reynolds_number)
    tau = Constant(1.0)
    dW = Constant(1.0)
    expW = Constant(1.0)
    exp_1W = Constant(1.0)
    exp_2W = Constant(1.0)
    t = Constant(1.0)

    initial_time, time_increments = trajectory_to_incremets(time_grid)
    time = initial_time
    accumulatedNoise = 0
    dNoise = 0

    x, y = SpatialCoordinate(space_disc.mesh)
    xhat, yhat = transformation(x,y,expW)

    upold = Function(space_disc.mixed_space)
    uold, pold = upold.subfunctions

    uold.assign(project(initialCondition(x,y),space_disc.velocity_space))

    a = ( inner(u,v) + tau*( 1.0/Re*inner(grad(u), grad(v)) - inner(p, div(v)) + inner(div(u), q) ) - Lambda*dW/2.0*inner(dot(grad(u), as_vector([x,y])), v) )*dx
    L = ( inner(uold,v) - 1.0/Re*tau*inner(bodyforce1(xhat,yhat),v)*(2-exp_2W)+ 2*tau*t*inner(bodyforce2(xhat,yhat),v)*(2-exp_1W) + Lambda*dW/2.0*inner(dot(grad(uold), as_vector([x,y])), v) )*dx

    up = Function(space_disc.mixed_space)
    u, p = up.subfunctions

    #approximate solution
    time_to_velocity = dict()
    time_to_pressure = dict()

    time_to_velocity[time] = deepcopy(uold)
    time_to_pressure[time] = deepcopy(pold)

    #handling of exact solution
    time_to_velError = dict()
    time_to_preError = dict()

    exactVelocity = project(exact_velocity(xhat,yhat),space_disc.velocity_space)
    exactPressure = project(exact_pressure(xhat,yhat),space_disc.pressure_space)
    mean_exactPressure = Constant(assemble( inner(exactPressure,1)*dx ))
    exactPressure.dat.data[:] = exactPressure.dat.data - Function(space_disc.pressure_space).assign(mean_exactPressure).dat.data

    
    solError = Function(space_disc.mixed_space)
    velError, preError = solError.subfunctions

    velError.dat.data[:] = exactVelocity.dat.data - uold.dat.data
    preError.dat.data[:] = exactPressure.dat.data*time - pold.dat.data

    time_to_velError[time] = deepcopy(velError)
    time_to_preError[time] = deepcopy(preError)

    for index in tqdm(range(len(time_increments))):
        dNoise = noise_increments[index]
        accumulatedNoise += Lambda*dNoise
        dW.assign(dNoise)
        expW.assign(exp(Lambda*accumulatedNoise))
        exp_1W.assign(exp(-Lambda*accumulatedNoise))
        exp_2W.assign(exp(-2*Lambda*accumulatedNoise))

        dtime = time_increments[index]
        time += dtime
        tau.assign(dtime)
        t.assign(time)

        bcs = [DirichletBC(space_disc.mixed_space.sub(0), project(initialCondition(xhat,yhat),space_disc.velocity_space), (1, 2, 3, 4))]
            
        solve(a == L, up, bcs=bcs, nullspace=space_disc.null)

        #Mean correction
        mean_p = Constant(assemble( inner(p,1)*dx ))
        p.dat.data[:] = p.dat.data - Function(space_disc.pressure_space).assign(mean_p).dat.data

        time_to_velocity[time] = deepcopy(u)
        time_to_pressure[time] = deepcopy(p)

        upold.assign(up)

        #update exact solution
        exactVelocity = project(exact_velocity(xhat,yhat),space_disc.velocity_space)
        exactPressure = project(exact_pressure(xhat,yhat),space_disc.pressure_space)
        mean_exactPressure = Constant(assemble( inner(exactPressure,1)*dx ))
        exactPressure.dat.data[:] = exactPressure.dat.data - Function(space_disc.pressure_space).assign(mean_exactPressure).dat.data

        #compute error
        velError.dat.data[:] = exactVelocity.dat.data - uold.dat.data
        preError.dat.data[:] = exactPressure.dat.data*time - pold.dat.data

        time_to_velError[time] = deepcopy(velError)
        time_to_preError[time] = deepcopy(preError)

    return time_to_velocity, time_to_pressure, time_to_velError, time_to_preError