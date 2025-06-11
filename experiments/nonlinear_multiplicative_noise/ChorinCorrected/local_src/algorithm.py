from firedrake import *
from copy import deepcopy
from tqdm import tqdm

from src.discretisation.time import trajectory_to_incremets
from src.discretisation.space import SpaceDiscretisation

def nonlinearity(u,v):
    return (u[0]**2 + 1)**(1/2.0)*v[0] + (u[1]**2 + 1)**(1/2.0)*v[1] 

def Chorin_splitting_with_pressure_correction(space_disc: SpaceDiscretisation,
                     time_grid: list[float],
                     noise_coeff_to_noise_increments: dict[Function,list[int]],
                     initial_condition: Function,
                     Reynolds_number: float = 1) -> tuple[dict[float,Function], dict[float,Function], dict[float,Function], dict[float,Function]]:
    """Solve Stokes system with modified Chorin splitting. 
    
    Return 'time -> velocity', 'time -> total pressure', 'time -> stochastic pressure', and 'time -> deterministic pressure' dictionaries. """

    Re = Constant(Reynolds_number)
    tau = Constant(1.0)
    noise_coeff_to_dW = {noise_coeff: Constant(1.0) for noise_coeff in noise_coeff_to_noise_increments}

    u = TrialFunction(space_disc.velocity_space)
    v = TestFunction(space_disc.velocity_space)
    uold = Function(space_disc.velocity_space)
    unew = Function(space_disc.velocity_space)
    utilde = Function(space_disc.velocity_space)
    noise_projected = Function(space_disc.velocity_space)

    p = TrialFunction(space_disc.pressure_space)
    q = TestFunction(space_disc.pressure_space)
    pnew = Function(space_disc.pressure_space)
    psto = Function(space_disc.pressure_space)
    pdet = Function(space_disc.pressure_space)

    #dummy zero velocity field
    fzero = Function(space_disc.velocity_space)

    #variational form: stochastich pressure
    a0 = inner(grad(p),grad(q))*dx
    L0 = 1/tau*inner(fzero,grad(q))*dx
    for noise_coeff in noise_coeff_to_noise_increments:
        L0 = L0 + 1/tau*noise_coeff_to_dW[noise_coeff]*nonlinearity(uold,grad(q))*noise_coeff[0]*dx

    #variational form: Helmholtz-projected noise
    a1 = inner(u,v)*dx
    L1 = -inner(grad(psto),v)*dx
    for noise_coeff in noise_coeff_to_noise_increments:
        L1 = L1 + 1/tau*noise_coeff_to_dW[noise_coeff]*nonlinearity(uold,v)*noise_coeff[0]*dx

    #variational form: artificial velocity
    a2 = ( inner(u,v) + 1/Re*tau*inner(grad(u), grad(v)) )*dx
    L2 = ( inner(uold,v) + tau*(v[0] + v[1]) + tau*inner(noise_projected,v))*dx

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

 
    time_to_velocity = dict()
    time_to_pressure = dict()
    time_to_stochastic_pressure = dict()
    time_to_deterministic_pressure = dict()

    time_to_velocity[time] = deepcopy(uold)
    time_to_pressure[time] = deepcopy(pnew)
    time_to_stochastic_pressure[time] = deepcopy(pnew)
    time_to_deterministic_pressure[time] = deepcopy(pnew)

    for index in tqdm(range(len(time_increments))):
        for noise_coeff in noise_coeff_to_noise_increments:
            noise_coeff_to_dW[noise_coeff].assign(noise_coeff_to_noise_increments[noise_coeff][index])
        tau.assign(time_increments[index])
        time += time_increments[index]
            
        #Solve variational forms
        solve(a0 == L0, psto, nullspace = V_basis)
        solve(a1 == L1, noise_projected)
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

    return time_to_velocity, time_to_pressure, time_to_stochastic_pressure, time_to_deterministic_pressure