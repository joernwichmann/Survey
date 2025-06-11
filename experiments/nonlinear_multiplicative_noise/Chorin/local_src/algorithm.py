from firedrake import *
from copy import deepcopy
from tqdm import tqdm

from src.discretisation.time import trajectory_to_incremets
from src.discretisation.space import SpaceDiscretisation

def nonlinearity(u,v):
    return (u[0]**2 + 1)**(1/2.0)*v[0] + (u[1]**2 + 1)**(1/2.0)*v[1] 

def Chorin_splitting(space_disc: SpaceDiscretisation,
                     time_grid: list[float],
                     noise_coeff_to_noise_increments: dict[Function,list[int]],
                     initial_condition: Function,
                     Reynolds_number: float = 1) -> tuple[dict[float,Function], dict[float,Function]]:
    """Solve Stokes system with Chorin splitting. 
    
    Return 'time -> velocity' and 'time -> pressure' dictionaries. """

    Re = Constant(Reynolds_number)
    tau = Constant(1.0)
    noise_coeff_to_dW = {noise_coeff: Constant(1.0) for noise_coeff in noise_coeff_to_noise_increments}

    u = TrialFunction(space_disc.velocity_space)
    v = TestFunction(space_disc.velocity_space)
    uold = Function(space_disc.velocity_space)
    unew = Function(space_disc.velocity_space)
    utilde = Function(space_disc.velocity_space)

    p = TrialFunction(space_disc.pressure_space)
    q = TestFunction(space_disc.pressure_space)
    pnew = Function(space_disc.pressure_space)

    a1 = ( inner(u,v) + 1/Re*tau*inner(grad(u), grad(v)) )*dx
    L1 = ( inner(uold,v) + tau*(v[0] + v[1]) )*dx
    for noise_coeff in noise_coeff_to_noise_increments:
        L1 = L1 + noise_coeff_to_dW[noise_coeff]*nonlinearity(uold,v)*noise_coeff[0]*dx

    a2 = inner(grad(p),grad(q))*dx
    L2 = 1/tau*inner(utilde,grad(q))*dx

    a3 = inner(u,v)*dx
    L3 = ( inner(utilde,v) - tau*inner(grad(pnew),v) )*dx

    V_basis = VectorSpaceBasis(constant=True)

    uold.assign(initial_condition)

    initial_time, time_increments = trajectory_to_incremets(time_grid)
    time = initial_time

 
    time_to_velocity = dict()
    time_to_pressure = dict()

    time_to_velocity[time] = deepcopy(uold)
    time_to_pressure[time] = deepcopy(pnew)

    for index in tqdm(range(len(time_increments))):
        for noise_coeff in noise_coeff_to_noise_increments:
            noise_coeff_to_dW[noise_coeff].assign(noise_coeff_to_noise_increments[noise_coeff][index])
        tau.assign(time_increments[index])
        time += time_increments[index]
        
        solve(a1 == L1, utilde, bcs=space_disc.bcs_vel)
        solve(a2 == L2, pnew, nullspace = V_basis)
        solve(a3 == L3, unew)

        #Mean correction
        mean_p = Constant(assemble( inner(pnew,1)*dx ))
        pnew.dat.data[:] = pnew.dat.data - Function(space_disc.pressure_space).assign(mean_p).dat.data

        time_to_velocity[time] = deepcopy(unew)
        time_to_pressure[time] = deepcopy(pnew)

        uold.assign(unew)

    return time_to_velocity, time_to_pressure