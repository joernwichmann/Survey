from firedrake import *
from copy import deepcopy
from tqdm import tqdm

from src.discretisation.time import trajectory_to_incremets
from src.discretisation.space import SpaceDiscretisation

from local_src.transport import initialCondition, transformation, noiseCoefficient, exact_pressure, exact_velocity, bodyforce1, bodyforce2

DIRECT_SOLVE_PARAMETERS = {'snes_max_it': 120,
           "snes_atol": 1e-8,
           "snes_rtol": 1e-8,
           'snes_linesearch_type': 'nleqerr',
           'ksp_type': 'preonly',
           'pc_type': 'lu', 
           'mat_type': 'aij',
           'pc_factor_mat_solver_type': 'mumps',
           "mat_mumps_icntl_14": 5000,
           "mat_mumps_icntl_24": 1,
           }

def Chorin_splitting(space_disc: SpaceDiscretisation,
                           time_grid: list[float],
                           noise_increments: list[int],
                           Reynolds_number: float = 1,
                           Lambda: float = 1) -> tuple[dict[float,Function], dict[float,Function]]:
    """Solve Stokes system with Chorin splitting. 
    
    Return 'time -> velocity' and 'time -> pressure' dictionaries. """

    Re = Constant(Reynolds_number)
    tau = Constant(1.0)
    dW = Constant(1.0)
    W = Constant(0.0)
    t = Constant(1.0)

    initial_time, time_increments = trajectory_to_incremets(time_grid)
    time = initial_time
    accumulatedNoise = 0
    dNoise = 0

    x, y = SpatialCoordinate(space_disc.mesh)
    xhat, yhat = transformation(x,y,W)

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
    uold.assign(project(initialCondition(x,y),space_disc.velocity_space))

    a1 = ( inner(u,v) + tau/Re*inner(grad(u), grad(v)) - Lambda*dW/2.0*inner(dot(grad(u), as_vector([1,1])), v) )*dx
    L1 = ( inner(uold,v) - tau/Re*inner(bodyforce1(xhat,yhat),v)+ 2*tau*t*inner(bodyforce2(xhat,yhat),v) + Lambda*dW/2.0*inner(dot(grad(uold), as_vector([1,1])), v) )*dx

    a2 = inner(grad(p),grad(q))*dx
    L2 = 1/tau*inner(utilde,grad(q))*dx

    a3 = inner(u,v)*dx
    L3 = ( inner(utilde,v) - tau*inner(grad(pnew),v) )*dx

    V_basis = VectorSpaceBasis(constant=True)

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
        W.assign(accumulatedNoise)

        dtime = time_increments[index]
        time += dtime
        tau.assign(dtime)
        t.assign(time)

        bcs = [DirichletBC(space_disc.velocity_space, project(initialCondition(xhat,yhat),space_disc.velocity_space), (1, 2, 3, 4))]
        
        solve(a1 == L1, utilde, bcs=bcs, solver_parameters=DIRECT_SOLVE_PARAMETERS)
        solve(a2 == L2, pnew, nullspace = V_basis)
        solve(a3 == L3, unew)

        #Mean correction
        mean_p = Constant(assemble( inner(pnew,1)*dx ))
        pnew.dat.data[:] = pnew.dat.data - Function(space_disc.pressure_space).assign(mean_p).dat.data

        time_to_velocity[time] = deepcopy(unew)
        time_to_pressure[time] = deepcopy(pnew)

        uold.assign(unew)
        pold.assign(pnew)

        #update exact solution
        exactVelocity = project(exact_velocity(xhat,yhat),space_disc.velocity_space)
        exactPressure = project(exact_pressure(xhat,yhat),space_disc.pressure_space)
        mean_exactPressure = Constant(assemble( inner(exactPressure,1)*dx ))
        exactPressure.dat.data[:] = exactPressure.dat.data - Function(space_disc.pressure_space).assign(mean_exactPressure).dat.data

        #compute error
        velError.dat.data[:] = exactVelocity.dat.data - utilde.dat.data
        preError.dat.data[:] = exactPressure.dat.data*time - pold.dat.data

        time_to_velError[time] = deepcopy(velError)
        time_to_preError[time] = deepcopy(preError)

    return time_to_velocity, time_to_pressure, time_to_velError, time_to_preError