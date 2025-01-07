import csv
from firedrake import Function
import os
from functools import cached_property
from numpy import ndarray

from src.utils import swap_dictionary_keys
from src.discretisation.time import TimeDiscretisation
from src.math.norms.stochastic import l1_stochastic, l2_stochastic, linf_stochastic
from src.math.statistics import standard_deviation, mean_value
from src.plotter import plot_ref_to_time_to_function, plot_seed_to_time_to_number, plot_seed_to_time_to_number_and_increments
from src.postprocess.processmanager import ProcessObject

def _evaluate_funcAtpoint(time_to_function: dict[float,Function], point: list[float], func_dim: int) -> list[dict[float,float]]:
    """Evaluate the dictionary "time -> function" at specified point. 

    Return 'component -> time -> function value'."""
    return {component: {time: time_to_function[time].at(point)[component] for time in time_to_function.keys()} for component in range(func_dim)}

class PointStatistics(ProcessObject):
    """Class that contains tools for computing the energy."""
    def __init__(self, 
                 time_disc: TimeDiscretisation,
                 point_name: str, 
                 point: list[float],
                 func_dim: int) -> None:
        self.seed_to_ref_to_comp_to_time_to_value = dict()
        self.seed_Id = 0
        self.time_disc = time_disc
        self.seed_to_ref_to_noise_increments = dict()
        self.point_name = point_name
        self.point = point
        self.func_dim = func_dim

    def update(self, ref_to_time_to_function: dict[int,dict[float,Function]], ref_to_noise_increments: list[int,ndarray],) -> None:
        """Add an entry to the 'seed -> refinement level -> time -> energy ' dictionary."""
        self.seed_to_ref_to_comp_to_time_to_value[self.seed_Id] = {level: _evaluate_funcAtpoint(ref_to_time_to_function[level],self.point,self.func_dim) 
                                                       for level in ref_to_time_to_function.keys()}
        self.seed_to_ref_to_noise_increments[self.seed_Id] = ref_to_noise_increments
        self.seed_Id += 1

    @cached_property
    def ref_to_seed_to_comp_to_time_to_value(self):
        if len(self.seed_to_ref_to_comp_to_time_to_value) == 0:
            return dict()
        return swap_dictionary_keys(self.seed_to_ref_to_comp_to_time_to_value)
    
    @cached_property
    def ref_to_comp_to_seed_to_time_to_value(self):
        if len(self.ref_to_seed_to_comp_to_time_to_value) == 0:
            return dict()
        return {level: swap_dictionary_keys(self.ref_to_seed_to_comp_to_time_to_value[level]) for level in self.ref_to_seed_to_comp_to_time_to_value}
    
    @cached_property
    def ref_to_comp_to_time_to_seed_to_value(self):
        if len(self.ref_to_comp_to_seed_to_time_to_value) == 0:
            return dict()
        return {level: {component: swap_dictionary_keys(self.ref_to_comp_to_seed_to_time_to_value[level][component]) 
                        for component in self.ref_to_comp_to_seed_to_time_to_value[level]}  
                        for level in self.ref_to_seed_to_comp_to_time_to_value}
    
    @cached_property
    def ref_to_seed_to_noise_increments(self):
        if len(self.seed_to_ref_to_noise_increments) == 0:
            return dict()
        return swap_dictionary_keys(self.seed_to_ref_to_noise_increments)
    
    @property
    def ref_to_comp_to_time_to_value_mean(self) -> dict[int,dict[int,dict[float,list[float]]]]:
        ref_to_time_to_funcAtpoint = {level: {component: {time: mean_value(self.ref_to_comp_to_time_to_seed_to_value[level][component][time].values()) 
                                                          for time in self.ref_to_comp_to_time_to_seed_to_value[level][component]} 
                                              for component in self.ref_to_comp_to_time_to_seed_to_value[level]} 
                                for level in self.ref_to_comp_to_time_to_seed_to_value}
        return ref_to_time_to_funcAtpoint
    
    @property
    def ref_to_comp_to_time_to_value_SD(self) -> dict[int,dict[int,dict[float,list[float]]]]:
        ref_to_time_to_funcAtpoint = {level: {component: {time: standard_deviation(self.ref_to_comp_to_time_to_seed_to_value[level][component][time].values()) 
                                                          for time in self.ref_to_comp_to_time_to_seed_to_value[level][component]} 
                                              for component in self.ref_to_comp_to_time_to_seed_to_value[level]} 
                                for level in self.ref_to_comp_to_time_to_seed_to_value}
        return ref_to_time_to_funcAtpoint
    
    def save(self, name_directory: str) -> None:
        """Save 'time -> funcAtpoint' in .csv files."""
        #construct header
        header = ["time"]
        header += [f"f_{component}-mean" for component in range(self.func_dim)]
        header += [f"f_{component}-SD" for component in range(self.func_dim)]

        ref_to_data = {}
        for level in self.time_disc.refinement_levels:
            data = []
            for time in self.time_disc.ref_to_time_grid[level]:
                mean = []
                sd = []
                for component in range(self.func_dim):
                    mean.append(self.ref_to_comp_to_time_to_value_mean[level][component][time])
                    sd.append(self.ref_to_comp_to_time_to_value_SD[level][component][time])
                data.append([time] + mean + sd)
            ref_to_data[level] = data
            
        if not os.path.isdir(name_directory):
            os.makedirs(name_directory)

        new_dict_name = name_directory + "/" + self.point_name + f"/mean"
        if not os.path.isdir(new_dict_name):
            os.makedirs(new_dict_name)

        for level in self.time_disc.refinement_levels:
            outfile = new_dict_name + "/refinement_" + str(level) + ".csv"
            with open(outfile,"w",newline="") as file:
                writer = csv.writer(file)
                writer.writerow(header)
                writer.writerows(ref_to_data[level])

    def save_individual(self, name_directory: str, requested_samples: int) -> None:
        """Save 'time -> funcAtpoint' in .csv files."""
        #construct header
        header = ["time"]
        header += [f"f_{component}" for component in range(self.func_dim)]

        #construct data
        seed_to_ref_to_data = dict()
        for sample in range(min(requested_samples,self.seed_Id)):
            ref_to_data = {}
            for level in self.time_disc.refinement_levels:
                data = []
                for time in self.time_disc.ref_to_time_grid[level]:
                    value = []
                    for component in range(self.func_dim):
                        value.append(self.ref_to_comp_to_time_to_seed_to_value[level][component][time][sample])
                    data.append([time] + value)
                ref_to_data[level] = data
            seed_to_ref_to_data[sample] = ref_to_data
        
        #check if directory is avaiable
        if not os.path.isdir(name_directory):
            os.makedirs(name_directory)

        #write data
        for sample in range(min(requested_samples,self.seed_Id)):
            new_dict_name = name_directory + "/" + self.point_name + f"/seed_{sample}"
            if not os.path.isdir(new_dict_name):
                os.makedirs(new_dict_name)
            for level in self.time_disc.refinement_levels:
                outfile = new_dict_name + "/refinement_" + str(level) + ".csv"
                with open(outfile,"w",newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(header)
                    writer.writerows(seed_to_ref_to_data[sample][level])

    def plot(self, name_directory: str) -> None:
        return
    #    """Save 'time -> energy'."""
    #    if not os.path.isdir(name_directory):
    #        os.makedirs(name_directory)

    #    new_dict_name = name_directory + "/" + self.energy_name 
    #    if not os.path.isdir(new_dict_name):
    #        os.makedirs(new_dict_name)
    #    plot_ref_to_time_to_function({self.energy_name + "_L1": self.ref_to_time_to_energy_l1, 
    #                                  self.energy_name + "_Deviation": self.ref_to_time_to_energy_deviation},new_dict_name + "/L1.png",["log","log"])
    #    plot_ref_to_time_to_function({self.energy_name + "_L2": self.ref_to_time_to_energy_l2, 
    #                                  self.energy_name + "_Deviation": self.ref_to_time_to_energy_deviation},new_dict_name + "/L2.png",["log","log"])
    #    plot_ref_to_time_to_function({self.energy_name + "_Linf": self.ref_to_time_to_energy_linf, 
    #                                  self.energy_name + "_Deviation": self.ref_to_time_to_energy_deviation},new_dict_name + "/Linf.png",["log","log"])

    def plot_individual(self, name_directory: str) -> None:
        return
