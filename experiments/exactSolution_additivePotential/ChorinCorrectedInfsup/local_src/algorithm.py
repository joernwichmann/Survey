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

def Chorin_splitting_with_pressure_correctionINFSUP(space_disc: SpaceDiscretisation,
                           time_grid: list[float],
                           noise_increments: list[int],
                           noise_coefficient: Function,
                           initial_condition: Function,
                           bodyforce1: Function,
                           bodyforce2: Function,
                           Reynolds_number: float = 1)-> tuple[dict[float,Function], dict[float,Function], dict[float,Function], dict[float,Function]]:
    """Solve Stokes system with modified Chorin splitting. 
    
    Return 'time -> velocity', 'time -> total pressure', 'time -> stochastic pressure', and 'time -> deterministic pressure' dictionaries. """

    Re = Constant(Reynolds_number)
    tau = Constant(1.0)
    dW = Constant(1.0)
    W = Constant(1.0)
    t = Constant(1.0)

    uMix, pMix = TrialFunctions(space_disc.mixed_space)
    vMix, qMix = TestFunctions(space_disc.mixed_space)

    u = TrialFunction(space_disc.velocity_space)
    v = TestFunction(space_disc.velocity_space)
    uold = Function(space_disc.velocity_space)
    unew = Function(space_disc.velocity_space)
    utilde = Function(space_disc.velocity_space)
    #noise_projected = Function(space_disc.velocity_space)

    p = TrialFunction(space_disc.pressure_space)
    q = TestFunction(space_disc.pressure_space)
    pold = Function(space_disc.pressure_space)
    pnew = Function(space_disc.pressure_space)
    #psto = Function(space_disc.pressure_space)
    pdet = Function(space_disc.pressure_space)

    up_projected = Function(space_disc.mixed_space)
    noise_projected, psto = up_projected.subfunctions

    #variational form: Helmholtz-projected noise    
    a0 = ( inner(uMix,vMix) - pMix*div(vMix) + qMix*div(uMix) )*dx
    L0 =  dW/tau*inner(noise_coefficient,vMix)*dx

    #variational form: artificial velocity
    a2 = ( inner(u,v) + 1/Re*tau*inner(grad(u), grad(v)) )*dx
    L2 = ( inner(uold,v) - 1.0/Re*tau*inner(bodyforce1,v) + 2*tau*t*inner(bodyforce2,v) + tau*inner(noise_projected,v) )*dx

    #variational form: deterministic pressure
    a3 = inner(grad(p),grad(q))*dx
    L3 = 1/tau*inner(utilde,grad(q))*dx

    #variational form: velocity
    a4 = inner(u,v)*dx
    L4 = ( inner(utilde,v) - tau*inner(grad(pdet),v) )*dx

    V_basis = VectorSpaceBasis(constant=True)

    uold.assign(initial_condition)

    initial_time, time_increments = trajectory_to_incremets(time_grid)
    time = initial_time
    accumulatedNoise = 0
    dNoise = 0

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

    exactVelocity = exact_velocity(space_disc.mesh,space_disc.velocity_space)
    exactPressureDet = exact_pressure_det(space_disc.mesh,space_disc.pressure_space)
    exactPressureSto = exact_pressure_sto(space_disc.mesh,space_disc.pressure_space)

    solError = Function(space_disc.mixed_space)
    velError, preError = solError.subfunctions
    preErrorSto = Function(space_disc.pressure_space)
    preErrorDet = Function(space_disc.pressure_space)

    velError.dat.data[:] = exactVelocity.dat.data - uold.dat.data
    preError.dat.data[:] = exactPressureDet.dat.data*time + exactPressureSto.dat.data*dNoise - pold.dat.data

    time_to_velError[time] = deepcopy(velError)
    time_to_preError[time] = deepcopy(preError)
    time_to_preErrorSto[time] = deepcopy(preError)
    time_to_preErrorDet[time] = deepcopy(preError)

    for index in tqdm(range(len(time_increments))):
        dNoise = noise_increments[index]
        accumulatedNoise += dNoise
        W.assign(accumulatedNoise)
        dW.assign(dNoise)

        dtime = time_increments[index]
        time += dtime
        tau.assign(dtime)
        t.assign(time)
            
        #Solve variational forms
        solve(a0 == L0, up_projected, bcs=space_disc.bcs_mixed, nullspace = space_disc.null)
        solve(a2 == L2, utilde, bcs=space_disc.bcs_vel)
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
        pold.assign(pnew)

        #Compute errors
        velError.dat.data[:] = exactVelocity.dat.data - utilde.dat.data
        preError.dat.data[:] = exactPressureDet.dat.data*time + exactPressureSto.dat.data*dNoise - pold.dat.data
        preErrorSto.dat.data[:] = exactPressureSto.dat.data*dNoise - psto.dat.data
        preErrorDet.dat.data[:] = exactPressureDet.dat.data*time - pdet.dat.data

        time_to_velError[time] = deepcopy(velError)
        time_to_preError[time] = deepcopy(preError)
        time_to_preErrorSto[time] = deepcopy(preErrorSto)
        time_to_preErrorDet[time] = deepcopy(preErrorDet)

    return time_to_velocity, time_to_pressure, time_to_velError, time_to_preError, time_to_stochastic_pressure, time_to_deterministic_pressure, time_to_preErrorSto, time_to_preErrorDet