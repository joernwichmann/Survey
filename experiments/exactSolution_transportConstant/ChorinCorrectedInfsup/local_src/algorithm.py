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

def Chorin_splitting_with_pressure_correctionINFSUP(space_disc: SpaceDiscretisation,
                           time_grid: list[float],
                           noise_increments: list[int],
                           Reynolds_number: float = 1,
                           Lambda: float = 1) -> tuple[dict[float,Function], dict[float,Function], dict[float,Function], dict[float,Function]]:
    """Solve Stokes system with modified Chorin splitting. 
    
    Return 'time -> velocity', 'time -> total pressure', 'time -> stochastic pressure', and 'time -> deterministic pressure' dictionaries. """

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

    uMix, pMix = TrialFunctions(space_disc.mixed_space)
    vMix, qMix = TestFunctions(space_disc.mixed_space)

    u = TrialFunction(space_disc.velocity_space)
    v = TestFunction(space_disc.velocity_space)
    uold = Function(space_disc.velocity_space)
    unew = Function(space_disc.velocity_space)
    utilde = Function(space_disc.velocity_space)
    utildeOld = Function(space_disc.velocity_space)
    #noise_projected = Function(space_disc.velocity_space)

    p = TrialFunction(space_disc.pressure_space)
    q = TestFunction(space_disc.pressure_space)
    pold = Function(space_disc.pressure_space)
    pnew = Function(space_disc.pressure_space)
    #psto = Function(space_disc.pressure_space)
    pdet = Function(space_disc.pressure_space)

    up_projected = Function(space_disc.mixed_space)
    noise_projected, psto = up_projected.subfunctions


    uold.assign(project(initialCondition(x,y),space_disc.velocity_space))
    utildeOld.assign(project(initialCondition(x,y),space_disc.velocity_space))

    #variational form: Helmholtz-projected noise  
    a0 = ( inner(uMix,vMix) - pMix*div(vMix) + qMix*div(uMix) )*dx
    L0 = Lambda*dW/(tau*2.0)*inner(dot(grad(utildeOld), as_vector([1,1])), vMix)*dx

    #variational form: artificial velocity
    a2 = ( inner(u,v) + 1/Re*tau*inner(grad(u), grad(v)) - Lambda*dW/2.0*inner(dot(grad(u), as_vector([1,1])), v) )*dx
    L2 = ( inner(uold,v) - tau/Re*inner(bodyforce1(xhat,yhat),v)+ 2*tau*t*inner(bodyforce2(xhat,yhat),v) + tau*inner(noise_projected,v) )*dx

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
        W.assign(accumulatedNoise)

        dtime = time_increments[index]
        time += dtime
        tau.assign(dtime)
        t.assign(time)

        bcs = [DirichletBC(space_disc.velocity_space, project(initialCondition(xhat,yhat),space_disc.velocity_space), (1, 2, 3, 4))]
            
        #Solve variational forms
        solve(a0 == L0, up_projected, bcs=space_disc.bcs_mixed, nullspace = space_disc.null)
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