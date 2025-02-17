"""Generate samples with configuration specified in 'config.py'."""
from firedrake import *
import numpy as np
import logging
from time import process_time_ns
from functools import partial

from src.discretisation.space import get_space_discretisation_from_CONFIG, SpaceDiscretisation
from src.discretisation.time import TimeDiscretisation
from src.data_dump.setup import  update_logfile
from src.algorithms.select import Algorithm, select_algorithm
from src.noise import SamplingStrategy, select_sampling
from src.predefined_data import get_function
from src.string_formatting import format_runtime, format_header
from src.utils import logstring_to_logger

from src.math.distances.space import l2_distance, h1_distance, V_distance, V_sym_distance
from src.math.distances.Bochner_time import linf_X_distance, l2_X_distance, end_time_X_distance, h_minus1_X_distance, w_minus1_inf_X_distance
from src.math.norms.space import l2_space, h1_space, hdiv_space
from src.math.norms.Bochner_time import linf_X_norm, l2_X_norm, end_time_X_norm, h_minus1_X_norm
from src.math.energy import kinetic_energy, potential_energy
from src.discretisation.projections import HL_projection_withBC
from src.postprocess.time_convergence import TimeComparison
from src.postprocess.stability_check import StabilityCheck
from src.postprocess.energy_check import Energy
from src.postprocess.statistics import StatisticsObject
from src.postprocess.point_statistics import PointStatistics
from src.postprocess.increments_check import IncrementCheck
from src.postprocess.processmanager import ProcessManager

#load global and lokal configs
from configs import nse_p_variation_exp1 as cf
from configs import nse_p_variation_global as gcf

def generate_one(time_disc: TimeDiscretisation,
                 space_disc: SpaceDiscretisation,
                 noise_coefficients: list[Function],
                 initial_velocity: Function,
                 initial_pressure: Function,
                 p_value: float,
                 kappa_value: float,
                 ref_to_time_to_det_forcing: dict[int,dict[float,Function]],
                 algorithm: Algorithm,
                 sampling_strategy: SamplingStrategy) -> tuple[dict[int,list[float]],
                                                               dict[int,dict[float,Function]],
                                                               dict[int,dict[float,Function]],
                                                               dict[int,dict[float,Function]],
                                                               dict[int,dict[float,Function]]]:
    """Run the numerical experiment once. 
    
    Return noise and solution."""
    ### Generate noise on all refinement levels
    noise_coeff_to_ref_to_noise_increments: dict[Function,dict[int,list[floats]]] = {noise_coeff: sampling_strategy(time_disc.refinement_levels,time_disc.initial_time,time_disc.end_time) 
                               for noise_coeff in noise_coefficients}
    ### initialise storage 
    ref_to_time_to_velocity = dict()
    ref_to_time_to_velocity_midpoints = dict()
    ref_to_time_to_pressure = dict()
    ref_to_time_to_pressure_midpoints = dict()
    for level in time_disc.refinement_levels:
        ### select noise_increments relative to refinement level
        noise_coeff_to_noise_increments = {noise_coeff: noise_coeff_to_ref_to_noise_increments[noise_coeff][level] 
                                           for noise_coeff in noise_coefficients}
        ### Solve algebraic system
        (ref_to_time_to_velocity[level],
         ref_to_time_to_pressure[level],
         ref_to_time_to_velocity_midpoints[level],
         ref_to_time_to_pressure_midpoints[level])  = algorithm(
            space_disc=space_disc,
            time_grid=time_disc.ref_to_time_grid[level],
            noise_coeff_to_noise_increments= noise_coeff_to_noise_increments,
            initial_condition=initial_velocity,
            p_value=p_value,
            kappa_value=kappa_value,
            time_to_det_forcing = ref_to_time_to_det_forcing[level],
            Reynolds_number=gcf.REYNOLDS_NUMBER
            )
    return (noise_coeff_to_ref_to_noise_increments,
            ref_to_time_to_velocity,
            ref_to_time_to_pressure,
            ref_to_time_to_velocity_midpoints,
            ref_to_time_to_pressure_midpoints)

def generate(deterministic: bool = False) -> None:
    """Runs the experiment.
    deterministic = True: run without stochastic forcing 
    deterministic = False: run with stochastic forcing
    """

    logging.basicConfig(filename=cf.NAME_LOGFILE_GENERATE,format='%(asctime)s| \t %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p', 
                        level=logstring_to_logger(gcf.LOG_LEVEL),force=True)


    # define discretisation
    space_disc = get_space_discretisation_from_CONFIG(name_mesh=gcf.MESH_NAME,
                                                      space_points=gcf.NUMBER_SPACE_POINTS,
                                                      velocity_element=gcf.VELOCITY_ELEMENT,
                                                      velocity_degree=gcf.VELOCITY_DEGREE,
                                                      pressure_element=gcf.PRESSURE_ELEMENT,
                                                      pressure_degree=gcf.PRESSURE_DEGREE,
                                                      name_bc=gcf.NAME_BOUNDARY_CONDITION
                                                      )
    logging.info(space_disc)

    time_disc = TimeDiscretisation(initial_time=gcf.INITIAL_TIME, end_time=gcf.END_TIME,refinement_levels=gcf.REFINEMENT_LEVELS)
    logging.info(time_disc)

    ####### DEFINE DATA
    ### initial condition
    logging.info(f"\nINITIAL CONDITION:\t{gcf.INITIAL_CONDITION_NAME}\nINITIAL INTENSITY:\t{gcf.INITIAL_INTENSITY}")
    #interprete string as function
    unprocessed_initial_velocity = gcf.INITIAL_INTENSITY*get_function(gcf.INITIAL_CONDITION_NAME,space_disc,gcf.INITIAL_FREQUENZY_X,gcf.INITIAL_FREQUENZY_Y)
    initial_velocity, initial_pressure = HL_projection_withBC(vector_field=unprocessed_initial_velocity,space_disc=space_disc)

    ### noise coefficient
    logging.info(f"\nNOISE COEFFICIENT:\t{gcf.NOISE_COEFFICIENT_NAME}\nNOISE INTENSITY:\t{gcf.NOISE_INTENSITY}")
    noise_coefficients_prescaled = get_function(gcf.NOISE_COEFFICIENT_NAME,space_disc,gcf.NOISE_FREQUENZY_X,gcf.NOISE_FREQUENZY_Y)
    noise_coefficients = [gcf.NOISE_INTENSITY*noise_coefficient for noise_coefficient in noise_coefficients_prescaled]
    
    logging.info(f"\nREYNOLDS NUMBER:\t{gcf.REYNOLDS_NUMBER}")

    if deterministic:
        print(f"length of noise coefficient list: \t{len(noise_coefficients)}")
        #plotting of noise coefficients
        ncplot = Function(space_disc.velocity_space)
        for k, noise_coefficient in enumerate(noise_coefficients):
            ncplot.assign(noise_coefficient)
            outfile = File(f"noise_coefficients/{cf.NAME_EXPERIMENT}/nc_{k}.pvd")
            outfile.write(ncplot)
        noise_coefficients = [Function(space_disc.velocity_space)]


    ### deterministic forcing
    logging.info(f"\nDETERMINISTIC FORCING:\t{gcf.FORCING}\nFORCING INTENSITY:\t{gcf.FORCING_INTENSITY}")
    ref_to_time_to_det_forcing = {level: {time: gcf.FORCING_INTENSITY*get_function(gcf.FORCING,space_disc,gcf.FORCING_FREQUENZY_X,gcf.FORCING_FREQUENZY_Y) for time in time_disc.ref_to_time_grid[level]} for level in time_disc.refinement_levels}
    
    ### shear stress
    logging.info(f"\nP-VALUE:\t{cf.P_VALUE}\nKAPPA-VALUE:\t{gcf.KAPPA_VALUE}")
    p_value = cf.P_VALUE
    kappa_value = gcf.KAPPA_VALUE
    

    # select algorithm
    algorithm = select_algorithm(gcf.MODEL_NAME,cf.ALGORITHM_NAME)

    #select sampling
    sampling_strategy = select_sampling(gcf.NOISE_INCREMENTS)

    #Initialise process managers to handle data processing
    if gcf.TIME_CONVERGENCE:
        time_convergence_velocity = ProcessManager([
            TimeComparison(time_disc.ref_to_time_stepsize,"Linf_L2_velocity",linf_X_distance,l2_distance,gcf.TIME_COMPARISON_TYPE),
            TimeComparison(time_disc.ref_to_time_stepsize,"End_time_L2_velocity",end_time_X_distance,l2_distance,gcf.TIME_COMPARISON_TYPE),
            TimeComparison(time_disc.ref_to_time_stepsize,"L2_H1_velocity",l2_X_distance,h1_distance,gcf.TIME_COMPARISON_TYPE),
            ])
        time_convergence_pressure = ProcessManager([
            TimeComparison(time_disc.ref_to_time_stepsize,"L2_L2_pressure",l2_X_distance,l2_distance,gcf.TIME_COMPARISON_TYPE),
            TimeComparison(time_disc.ref_to_time_stepsize,"H-1_L2_pressure",h_minus1_X_distance,l2_distance,gcf.TIME_COMPARISON_TYPE),
            TimeComparison(time_disc.ref_to_time_stepsize,"W-1_inf_L2_pressure",w_minus1_inf_X_distance,l2_distance,gcf.TIME_COMPARISON_TYPE)
            ])

    if gcf.STABILITY_CHECK:
        stability_check_velocity = ProcessManager([
            StabilityCheck(time_disc.ref_to_time_stepsize,"Linf_L2_velocity",linf_X_norm,l2_space),
            StabilityCheck(time_disc.ref_to_time_stepsize,"End_time_L2_velocity",end_time_X_norm,l2_space),
            StabilityCheck(time_disc.ref_to_time_stepsize,"L2_H1_velocity",l2_X_norm,h1_space),
            StabilityCheck(time_disc.ref_to_time_stepsize,"L2_Hdiv_velocity",l2_X_norm,hdiv_space)
        ])
        stability_check_pressure = ProcessManager([
            StabilityCheck(time_disc.ref_to_time_stepsize,"L2_L2_pressure",l2_X_norm,l2_space),
            StabilityCheck(time_disc.ref_to_time_stepsize,"H-1_L2_pressure",h_minus1_X_norm,l2_space)
        ])

    if gcf.ENERGY_CHECK:
        energy_check_velocity = ProcessManager([
            Energy(time_disc,"kinetic_energy",kinetic_energy),
            #Energy(time_disc,"potential_energy",potential_energy)
        ])

    if gcf.IND_ENERGY_CHECK:
        sample_to_energy_check_velocity = dict()

    if gcf.STATISTICS_CHECK:
        statistics_velocity = StatisticsObject("velocity",time_disc.ref_to_time_grid,space_disc.velocity_space)
        statistics_velocity_midpoints = StatisticsObject("velocity_midpoints",time_disc.ref_to_time_grid,space_disc.velocity_space)
        statistics_pressure = StatisticsObject("pressure",time_disc.ref_to_time_grid,space_disc.pressure_space)
        statistics_pressure_midpoints = StatisticsObject("pressure_midpoints",time_disc.ref_to_time_grid,space_disc.pressure_space)

    if gcf.POINT_STATISTICS_CHECK:
        point_statistics_velocity = ProcessManager([
            PointStatistics(time_disc,"p1",gcf.POINT_1,2),
            PointStatistics(time_disc,"p2",gcf.POINT_2,2),
            PointStatistics(time_disc,"p3",gcf.POINT_3,2),
        ])

    if gcf.INCREMENT_CHECK:
        increment_check = ProcessManager([
            IncrementCheck(ref_to_stepsize=time_disc.ref_to_time_stepsize,
                           coarse_timeMesh=time_disc.ref_to_time_grid[time_disc.refinement_levels[0]],
                           distance_name="L2-inc",
                           space_distance=l2_distance)
        ])

    
    runtimes = {"solving": 0,"comparison": 0, "stability": 0, "energy": 0, "statistics": 0, "point-statistics": 0, "increment": 0}
    
    if deterministic:
        print(format_header("RUN DETERMINISTIC EXPERIMENT"))
        new_seeds = range(1)
    else: 
        print(format_header("START MONTE CARLO ITERATION") + f"\nRequested samples:\t{gcf.MC_SAMPLES}")
        new_seeds = range(gcf.MC_SAMPLES)

    ### start MC iteration 
    for k in new_seeds:
        ### get solution
        print(f"{k*100/len(new_seeds):4.2f}% completed")
        time_mark = process_time_ns()
        (ref_to_noise_increments, 
         ref_to_time_to_velocity, 
         ref_to_time_to_pressure, 
         ref_to_time_to_velocity_midpoints, 
         ref_to_time_to_pressure_midpoints) = generate_one(time_disc=time_disc,
                                                           space_disc=space_disc,
                                                           noise_coefficients=noise_coefficients,
                                                           initial_velocity=initial_velocity,
                                                           initial_pressure=initial_pressure,
                                                           p_value=p_value,
                                                           kappa_value=kappa_value,
                                                           ref_to_time_to_det_forcing=ref_to_time_to_det_forcing,
                                                           algorithm=algorithm,
                                                           sampling_strategy=sampling_strategy)
        runtimes["solving"] += process_time_ns()-time_mark

        #update data using solution
        if gcf.TIME_CONVERGENCE:
            time_mark = process_time_ns()
            time_to_fine_velocity = ref_to_time_to_velocity[time_disc.refinement_levels[-1]]
            time_convergence_velocity.update(ref_to_time_to_velocity,time_to_fine_velocity)
            time_to_fine_pressure = ref_to_time_to_pressure[time_disc.refinement_levels[-1]]
            time_convergence_pressure.update(ref_to_time_to_pressure,time_to_fine_pressure)
            runtimes["comparison"] += process_time_ns()-time_mark

        if gcf.STABILITY_CHECK:
            time_mark = process_time_ns()
            stability_check_velocity.update(ref_to_time_to_velocity)
            stability_check_pressure.update(ref_to_time_to_pressure)
            runtimes["stability"] += process_time_ns()-time_mark

        if gcf.ENERGY_CHECK:
            time_mark = process_time_ns()
            energy_check_velocity.update(ref_to_time_to_velocity,ref_to_noise_increments)
            runtimes["energy"] += process_time_ns()-time_mark

        if gcf.IND_ENERGY_CHECK and k <= gcf.IND_ENERGY_NUMBER:
            time_mark = process_time_ns()
            ind_energy_check_velocity = ProcessManager([
                Energy(time_disc,f"ind_kinetic_energy_{k}",kinetic_energy),
                #Energy(time_disc,f"ind_potential_energy_{k}",potential_energy)
                ])
            ind_energy_check_velocity.update(ref_to_time_to_velocity,ref_to_noise_increments)
            sample_to_energy_check_velocity[k] = ind_energy_check_velocity
            runtimes["energy"] += process_time_ns()-time_mark

        if gcf.STATISTICS_CHECK:
            time_mark = process_time_ns()
            statistics_velocity.update(ref_to_time_to_velocity)
            statistics_velocity_midpoints.update(ref_to_time_to_velocity_midpoints)
            statistics_pressure.update(ref_to_time_to_pressure)
            statistics_pressure_midpoints.update(ref_to_time_to_pressure_midpoints)
            runtimes["statistics"] += process_time_ns()-time_mark

        if gcf.POINT_STATISTICS_CHECK:
            time_mark = process_time_ns()
            point_statistics_velocity.update(ref_to_time_to_velocity,ref_to_noise_increments)
            runtimes["point-statistics"] += process_time_ns()-time_mark

        if gcf.INCREMENT_CHECK:
            time_mark = process_time_ns()
            increment_check.update(ref_to_time_to_velocity)
            runtimes["increment"] += process_time_ns()-time_mark



    
    ### storing processed data 
    if gcf.TIME_CONVERGENCE:
        logging.info(format_header("TIME CONVERGENCE") + f"\nComparisons are stored in:\t {cf.TIME_DIRECTORYNAME}/")
        logging.info(time_convergence_velocity)
        logging.info(time_convergence_pressure)

        if deterministic:
            time_convergence_velocity.save(cf.TIME_DIRECTORYNAME + "/deterministic")
            time_convergence_pressure.save(cf.TIME_DIRECTORYNAME + "/deterministic")
        else:
            time_convergence_velocity.save(cf.TIME_DIRECTORYNAME)
            time_convergence_pressure.save(cf.TIME_DIRECTORYNAME)

    if gcf.STABILITY_CHECK:
        logging.info(format_header("STABILITY CHECK") + f"\nStability checks are stored in:\t {cf.STABILITY_DIRECTORYNAME}/")
        logging.info(stability_check_velocity)
        logging.info(stability_check_pressure)

        if deterministic:
            stability_check_velocity.save(cf.STABILITY_DIRECTORYNAME + "/deterministic")
            stability_check_pressure.save(cf.STABILITY_DIRECTORYNAME + "/deterministic")
        else:
            stability_check_velocity.save(cf.STABILITY_DIRECTORYNAME)
            stability_check_pressure.save(cf.STABILITY_DIRECTORYNAME)

    if gcf.ENERGY_CHECK:
        logging.info(format_header("ENERGY CHECK") + f"\nEnergy checks are stored in:\t {cf.ENERGY_DIRECTORYNAME}/")
        if deterministic:  
            energy_check_velocity.save(cf.ENERGY_DIRECTORYNAME + "/deterministic")
            energy_check_velocity.plot(cf.ENERGY_DIRECTORYNAME + "/deterministic")
        else:
            energy_check_velocity.save(cf.ENERGY_DIRECTORYNAME)
            energy_check_velocity.plot(cf.ENERGY_DIRECTORYNAME)

    if gcf.IND_ENERGY_CHECK and not deterministic:
        logging.info(format_header("ENERGY CHECK") + f"\nIndividual energy checks are stored in:\t {cf.ENERGY_DIRECTORYNAME}/individual/")
        for sample in sample_to_energy_check_velocity.keys():
            sample_to_energy_check_velocity[sample].save(cf.ENERGY_DIRECTORYNAME + "/individual")
            #sample_to_energy_check_velocity[sample].plot(cf.ENERGY_DIRECTORYNAME + "/individual")
        #energy_check_velocity.save(cf.ENERGY_DIRECTORYNAME)


    if gcf.STATISTICS_CHECK:
        logging.info(format_header("STATISTICS") + f"\nStatistics are stored in:\t {cf.VTK_DIRECTORY + '/' + cf.STATISTICS_DIRECTORYNAME}/")
        if deterministic:
            statistics_velocity.save(cf.VTK_DIRECTORY + "/" + cf.STATISTICS_DIRECTORYNAME + "/deterministic")
            statistics_velocity_midpoints.save(cf.VTK_DIRECTORY + "/" + cf.STATISTICS_DIRECTORYNAME + "/deterministic")
            statistics_pressure.save(cf.VTK_DIRECTORY + "/" + cf.STATISTICS_DIRECTORYNAME + "/deterministic")
            statistics_pressure_midpoints.save(cf.VTK_DIRECTORY + "/" + cf.STATISTICS_DIRECTORYNAME + "/deterministic")
        else:
            statistics_velocity.save(cf.VTK_DIRECTORY + "/" + cf.STATISTICS_DIRECTORYNAME)
            statistics_velocity_midpoints.save(cf.VTK_DIRECTORY + "/" + cf.STATISTICS_DIRECTORYNAME)
            statistics_pressure.save(cf.VTK_DIRECTORY + "/" + cf.STATISTICS_DIRECTORYNAME)
            statistics_pressure_midpoints.save(cf.VTK_DIRECTORY + "/" + cf.STATISTICS_DIRECTORYNAME)

    if gcf.POINT_STATISTICS_CHECK:
        logging.info(format_header("POINT STATISTICS") + f"\nPoint statistics are stored in:\t {cf.POINT_STATISTICS_DIRECTORYNAME}/")
        if deterministic:
            point_statistics_velocity.save(cf.POINT_STATISTICS_DIRECTORYNAME + "/deterministic")
        else:
            point_statistics_velocity.save(cf.POINT_STATISTICS_DIRECTORYNAME)
            point_statistics_velocity.save_individual(cf.POINT_STATISTICS_DIRECTORYNAME,gcf.IND_POINT_STATISTICS_CHECK_NUMBER)

    if gcf.INCREMENT_CHECK:
        logging.info(format_header("INCREMENT CHECK") + f"\nIncrement check is stored in:\t {cf.INCREMENT_DIRECTORYNAME}/")
        logging.info(increment_check)
        if deterministic:
            increment_check.save(cf.INCREMENT_DIRECTORYNAME + "/deterministic")
            increment_check.plot(cf.INCREMENT_DIRECTORYNAME + "/deterministic")
        else:
            increment_check.save(cf.INCREMENT_DIRECTORYNAME)
            increment_check.plot(cf.INCREMENT_DIRECTORYNAME)
            increment_check.plot_individual(cf.INCREMENT_DIRECTORYNAME)
            


    #show runtimes
    logging.info(format_runtime(runtimes) + "\n\n")

if __name__ == "__main__":
    #remove old logfile
    update_logfile(gcf.DUMP_LOCATION,cf.NAME_LOGFILE_GENERATE)

    #run deterministic experiment
    generate(deterministic=True)
    
    #run stochastic experiment
    generate(deterministic=False)
    
    #display storage location of log file
    print(f"Logs saved in:\t {cf.NAME_LOGFILE_GENERATE}")