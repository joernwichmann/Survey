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

def implicitEuler_mixedFEM(space_disc: SpaceDiscretisation,
                           time_grid: list[float],
                           noise_increments: list[int],
                           Reynolds_number: float = 1,
                           Lambda: float = 1) -> tuple[dict[float,Function], dict[float,Function]]:
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
    L = ( inner(uold,v) - 1.0/Re*tau*inner(bodyforce1(xhat,yhat),v)*(exp_2W)+ 2*tau*t*inner(bodyforce2(xhat,yhat),v)*(exp_1W) + Lambda*dW/2.0*inner(dot(grad(uold), as_vector([x,y])), v) )*dx

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
        exp_1W.assign(exp(Lambda*accumulatedNoise))
        exp_2W.assign(exp(2*Lambda*accumulatedNoise))

        dtime = time_increments[index]
        time += dtime
        tau.assign(dtime)
        t.assign(time)

        bcs = [DirichletBC(space_disc.mixed_space.sub(0), project(initialCondition(xhat,yhat),space_disc.velocity_space), (1, 2, 3, 4))]
            
        solve(a == L, up, bcs=bcs, nullspace=space_disc.null, solver_parameters=DIRECT_SOLVE_PARAMETERS)

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

def Chorin_splitting(space_disc: SpaceDiscretisation,
                           time_grid: list[float],
                           noise_increments: list[int],
                           Reynolds_number: float = 1,
                           Lambda: float = 1)  -> tuple[dict[float,Function], dict[float,Function]]:
    """Solve Stokes system with Chorin splitting. 
    
    Return 'time -> velocity' and 'time -> pressure' dictionaries. """

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

    u = TrialFunction(space_disc.velocity_space)
    v = TestFunction(space_disc.velocity_space)
    uold = Function(space_disc.velocity_space)
    unew = Function(space_disc.velocity_space)
    utilde = Function(space_disc.velocity_space)
    utildeOld = Function(space_disc.velocity_space)

    p = TrialFunction(space_disc.pressure_space)
    q = TestFunction(space_disc.pressure_space)
    pold = Function(space_disc.pressure_space)
    pnew = Function(space_disc.pressure_space)

    #setup variational form
    uold.assign(project(initialCondition(x,y),space_disc.velocity_space))
    utildeOld.assign(project(initialCondition(x,y),space_disc.velocity_space))

    a1 = ( inner(u,v) + tau*( 1.0/Re*inner(grad(u), grad(v)) ) - Lambda*dW/2.0*inner(dot(grad(u), as_vector([x,y])), v) )*dx
    L1 = ( inner(uold,v) - 1.0/Re*tau*inner(bodyforce1(xhat,yhat),v)*(exp_2W)+ 2*tau*t*inner(bodyforce2(xhat,yhat),v)*(exp_1W) + Lambda*dW/2.0*inner(dot(grad(utildeOld), as_vector([x,y])), v) )*dx

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

    velError.dat.data[:] = exactVelocity.dat.data - utildeOld.dat.data
    preError.dat.data[:] = exactPressure.dat.data*time - pold.dat.data

    time_to_velError[time] = deepcopy(velError)
    time_to_preError[time] = deepcopy(preError)

    for index in tqdm(range(len(time_increments))):
        dNoise = noise_increments[index]
        accumulatedNoise += Lambda*dNoise
        dW.assign(dNoise)
        expW.assign(exp(Lambda*accumulatedNoise))
        exp_1W.assign(exp(Lambda*accumulatedNoise))
        exp_2W.assign(exp(2*Lambda*accumulatedNoise))

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
        utildeOld.assign(utilde)
        pold.assign(pnew)

        #update exact solution
        exactVelocity = project(exact_velocity(xhat,yhat),space_disc.velocity_space)
        exactPressure = project(exact_pressure(xhat,yhat),space_disc.pressure_space)
        mean_exactPressure = Constant(assemble( inner(exactPressure,1)*dx ))
        exactPressure.dat.data[:] = exactPressure.dat.data - Function(space_disc.pressure_space).assign(mean_exactPressure).dat.data

        #compute error
        velError.dat.data[:] = exactVelocity.dat.data - utildeOld.dat.data
        preError.dat.data[:] = exactPressure.dat.data*time - pold.dat.data

        time_to_velError[time] = deepcopy(velError)
        time_to_preError[time] = deepcopy(preError)

    return time_to_velocity, time_to_pressure, time_to_velError, time_to_preError

def Chorin_splitting_with_pressure_correction(space_disc: SpaceDiscretisation,
                           time_grid: list[float],
                           noise_increments: list[int],
                           Reynolds_number: float = 1,
                           Lambda: float = 1) -> tuple[dict[float,Function], dict[float,Function], dict[float,Function], dict[float,Function]]:
    """Solve Stokes system with modified Chorin splitting. 
    
    Return 'time -> velocity', 'time -> total pressure', 'time -> stochastic pressure', and 'time -> deterministic pressure' dictionaries. """

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

    u = TrialFunction(space_disc.velocity_space)
    v = TestFunction(space_disc.velocity_space)
    uold = Function(space_disc.velocity_space)
    unew = Function(space_disc.velocity_space)
    utilde = Function(space_disc.velocity_space)
    utildeOld = Function(space_disc.velocity_space)
    noise_projected = Function(space_disc.velocity_space)

    p = TrialFunction(space_disc.pressure_space)
    q = TestFunction(space_disc.pressure_space)
    pold = Function(space_disc.pressure_space)
    pnew = Function(space_disc.pressure_space)
    psto = Function(space_disc.pressure_space)
    pdet = Function(space_disc.pressure_space)

    #setup variational form
    uold.assign(project(initialCondition(x,y),space_disc.velocity_space))
    utildeOld.assign(project(initialCondition(x,y),space_disc.velocity_space))

    #variational form: stochastich pressure
    a0 = inner(grad(p),grad(q))*dx
    L0 = Lambda*dW/(tau*2.0)*inner(dot(grad(utildeOld), as_vector([x,y])), grad(q))*dx

    #variational form: Helmholtz-projected noise
    a1 = inner(u,v)*dx
    L1 = ( Lambda*dW/(tau*2.0)*inner(dot(grad(utildeOld), as_vector([x,y])), v) -inner(grad(psto),v) )*dx

    #variational form: artificial velocity
    a2 = ( inner(u,v) + 1/Re*tau*inner(grad(u), grad(v)) - Lambda*dW/2.0*inner(dot(grad(u), as_vector([1,1])), v) )*dx
    L2 = ( inner(uold,v) - tau/Re*inner(bodyforce1(xhat,yhat),v)*(exp_2W)+ 2*tau*t*inner(bodyforce2(xhat,yhat),v)*(exp_1W) + tau*inner(noise_projected,v) )*dx

    #variational form: deterministic pressure
    a3 = inner(grad(p),grad(q))*dx
    L3 = 1/tau*inner(utilde,grad(q))*dx

    #variational form: velocity
    a4 = inner(u,v)*dx
    L4 = ( inner(utilde,v) - tau*inner(grad(pdet),v) )*dx

    V_basis = VectorSpaceBasis(constant=True)

    #handling of approximate solution
    time_to_velocity = dict()
    time_to_pressure = dict()
    time_to_stochastic_pressure = dict()
    time_to_deterministic_pressure = dict()

    time_to_velocity[time] = deepcopy(uold)
    time_to_pressure[time] = deepcopy(pnew)
    time_to_stochastic_pressure[time] = deepcopy(pnew)
    time_to_deterministic_pressure[time] = deepcopy(pnew)

    #handling of exact solution and errors
    time_to_velError = dict()
    time_to_preError = dict()
    time_to_preErrorSto = dict()
    time_to_preErrorDet = dict()

    exactVelocity = project(exact_velocity(xhat,yhat),space_disc.velocity_space)
    exactPressure = project(exact_pressure(xhat,yhat),space_disc.pressure_space)
    mean_exactPressure = Constant(assemble( inner(exactPressure,1)*dx ))
    exactPressure.dat.data[:] = exactPressure.dat.data - Function(space_disc.pressure_space).assign(mean_exactPressure).dat.data

    solError = Function(space_disc.mixed_space)
    velError, preError = solError.subfunctions
    preErrorSto = Function(space_disc.pressure_space)
    preErrorDet = Function(space_disc.pressure_space)

    velError.dat.data[:] = exactVelocity.dat.data - utildeOld.dat.data
    preError.dat.data[:] = exactPressure.dat.data*time - pold.dat.data

    time_to_velError[time] = deepcopy(velError)
    time_to_preError[time] = deepcopy(preError)
    time_to_preErrorSto[time] = deepcopy(preError)
    time_to_preErrorDet[time] = deepcopy(preError)

    for index in tqdm(range(len(time_increments))):
        dNoise = noise_increments[index]
        accumulatedNoise += Lambda*dNoise
        dW.assign(dNoise)
        expW.assign(exp(Lambda*accumulatedNoise))
        exp_1W.assign(exp(Lambda*accumulatedNoise))
        exp_2W.assign(exp(2*Lambda*accumulatedNoise))

        dtime = time_increments[index]
        time += dtime
        tau.assign(dtime)
        t.assign(time)

        bcs = [DirichletBC(space_disc.velocity_space, project(initialCondition(xhat,yhat),space_disc.velocity_space), (1, 2, 3, 4))]
            
        #Solve variational forms
        solve(a0 == L0, psto, nullspace = V_basis)
        solve(a1 == L1, noise_projected)
        solve(a2 == L2, utilde, bcs=bcs, solver_parameters=DIRECT_SOLVE_PARAMETERS)
        solve(a3 == L3, pdet, nullspace = V_basis)
        solve(a4 == L4, unew)

        #Mean correction
        mean_psto = Constant(assemble( inner(psto,1)*dx ))
        psto.dat.data[:] = psto.dat.data - Function(space_disc.pressure_space).assign(mean_psto).dat.data
        mean_pdet = Constant(assemble( inner(pdet,1)*dx ))
        pdet.dat.data[:] = pdet.dat.data - Function(space_disc.pressure_space).assign(mean_pdet).dat.data

        #Compute total pressure
        pnew.dat.data[:] = pdet.dat.data + psto.dat.data

        time_to_velocity[time] = deepcopy(unew)
        time_to_pressure[time] = deepcopy(pnew)
        time_to_stochastic_pressure[time] = deepcopy(psto)
        time_to_deterministic_pressure[time] = deepcopy(pdet)

        uold.assign(unew)
        utildeOld.assign(utilde)
        pold.assign(pnew)

        #update exact solution
        exactVelocity = project(exact_velocity(xhat,yhat),space_disc.velocity_space)
        exactPressure = project(exact_pressure(xhat,yhat),space_disc.pressure_space)
        mean_exactPressure = Constant(assemble( inner(exactPressure,1)*dx ))
        exactPressure.dat.data[:] = exactPressure.dat.data - Function(space_disc.pressure_space).assign(mean_exactPressure).dat.data

        #compute error
        velError.dat.data[:] = exactVelocity.dat.data - utildeOld.dat.data
        preError.dat.data[:] = exactPressure.dat.data*time - pold.dat.data
        preErrorSto.dat.data[:] = psto.dat.data
        preErrorDet.dat.data[:] = exactPressure.dat.data*time - pdet.dat.data

        time_to_velError[time] = deepcopy(velError)
        time_to_preError[time] = deepcopy(preError)
        time_to_preErrorSto[time] = deepcopy(preErrorSto)
        time_to_preErrorDet[time] = deepcopy(preErrorDet)

    return time_to_velocity, time_to_pressure, time_to_velError, time_to_preError, time_to_stochastic_pressure, time_to_deterministic_pressure, time_to_preErrorSto, time_to_preErrorDet