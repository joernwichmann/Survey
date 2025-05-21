from firedrake import Function
import csv
import os
import numpy as np
from functools import cached_property
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.utils import swap_dictionary_keys
from src.math.norms.stochastic import l1_stochastic
from src.math.distances.space import SpaceDistance
from src.math.statistics import standard_deviation
from src.postprocess.eoc import get_ref_to_EOC
from src.plotter import COLOR_LIST
from src.postprocess.processmanager import ProcessObject

def _evaluate_space_distance_of_increments(ref_to_time_to_function: dict[int,dict[float,Function]],
                                           space_distance: SpaceDistance) -> dict[int,dict[int,float]]:
    """Evaluate increments of the function with respect to 'space_distance'. 

    Return 'refinement level -> time -> increment comparison' dictionary."""
    ref_to_time_to_incrementValue = dict()
    for level in ref_to_time_to_function.keys():
        ###WARNING: list might be disordered (should be ordered with respect to time)
        funcs = list(ref_to_time_to_function[level].values())
        times = list(ref_to_time_to_function[level].keys())
        increments = [space_distance(funcs[index+1],funcs[index])**2 
                              for index in range(len(funcs) - 1)]
        summed_increments = np.cumsum(increments)
        rescaled_increments = [sInc/(index + 1) for index, sInc in enumerate(summed_increments)]
        time_to_incrementValue = {time: rInc for time, rInc in zip(times[1:],rescaled_increments)}
        ref_to_time_to_incrementValue[level] = time_to_incrementValue
    return ref_to_time_to_incrementValue


class IncrementCheck(ProcessObject):
    """Class that contains tools for checking norm stablity."""
    def __init__(self, ref_to_stepsize: dict[int,float],
                 coarse_timeMesh: list[float], 
                 distance_name: str, 
                 space_distance: SpaceDistance) -> None:
        self.seed_to_ref_to_time_to_incrementValue = dict()
        self.seed_Id = 0
        self.ref_to_stepsize = ref_to_stepsize
        self.distance_name = distance_name
        self.space_distance = space_distance
        self.coarse_timeMesh = coarse_timeMesh[1:]

    def update(self, ref_to_time_to_function: dict[int,dict[float,Function]]) -> None:
        """Add an entry to the 'seed -> refinement level -> norm ' dictionary."""
        self.seed_to_ref_to_time_to_incrementValue[self.seed_Id] = _evaluate_space_distance_of_increments(ref_to_time_to_function,self.space_distance)
        self.seed_Id += 1

    @cached_property
    def ref_to_seed_to_time_to_incrementValue(self):
        if len(self.seed_to_ref_to_time_to_incrementValue) == 0:
            return dict()
        return swap_dictionary_keys(self.seed_to_ref_to_time_to_incrementValue)
    
    @cached_property
    def ref_to_time_to_seed_to_incrementValue(self):
        if len(self.ref_to_seed_to_time_to_incrementValue) == 0:
            return dict()
        return {level: swap_dictionary_keys(self.ref_to_seed_to_time_to_incrementValue[level])
                for level in self.ref_to_seed_to_time_to_incrementValue.keys()}
    
    @cached_property
    def ref_to_time_to_norm_l1(self) -> dict[int,dict[int,float]]:
        return {level: {time: l1_stochastic(self.ref_to_time_to_seed_to_incrementValue[level][time].values())
                        for time in self.ref_to_time_to_seed_to_incrementValue[level].keys()}
                for level in self.ref_to_time_to_seed_to_incrementValue.keys()}
    
    @cached_property
    def time_to_ref_to_norm_l1(self) -> dict[int,dict[int,float]]:
        if len(self.ref_to_time_to_norm_l1) == 0:
            return dict()
        return swap_dictionary_keys(self.ref_to_time_to_norm_l1)
    
    @property
    def ref_to_time_to_norm_SD(self) -> dict[int,dict[int,float]]:
        return {level: {time: standard_deviation(self.ref_to_time_to_seed_to_incrementValue[level][time].values())
                        for time in self.ref_to_time_to_seed_to_incrementValue[level].keys()}
                for level in self.ref_to_time_to_seed_to_incrementValue.keys()}
    
    @cached_property
    def time_to_ref_to_norm_SD(self) -> dict[int,dict[int,float]]:
        if len(self.ref_to_time_to_norm_SD) == 0:
            return dict()
        return swap_dictionary_keys(self.ref_to_time_to_norm_SD)
      
    @property
    def time_to_ref_to_EOC_l1(self) -> dict[int,dict[int,float]]:
        return {time: get_ref_to_EOC(self.time_to_ref_to_norm_l1[time],self.ref_to_stepsize)
                for time in self.coarse_timeMesh}
    
    
    def save(self,name_directory) -> None:
        """Save increment and EOC in .csv-file."""
        header = ["time","dt","mean","EOC-mean","SD"]
        data = []
        for time in self.coarse_timeMesh:
            for level in self.time_to_ref_to_norm_l1[time].keys():
                data.append([time,
                             self.ref_to_stepsize[level],
                             self.time_to_ref_to_norm_l1[time][level],
                             self.time_to_ref_to_EOC_l1[time][level],
                             self.time_to_ref_to_norm_SD[time][level]
                             ])
                
        if not os.path.isdir(name_directory):
            os.makedirs(name_directory)

        outfile = name_directory + "/" + self.distance_name + ".csv"
        with open(outfile,"w",newline="") as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(data)

    def plot(self, name_directory: str) -> None:
        """Plot 'time -> increment'."""
        if not os.path.isdir(name_directory):
            os.makedirs(name_directory)
        new_dict_name = name_directory + "/" + self.distance_name 
        
        plt.figure()
        legendMarkers = []
        legendEntries = []
        for id, level in enumerate(self.ref_to_time_to_norm_l1.keys()):
            times = list(self.ref_to_time_to_norm_l1[level].keys())
            values = list(self.ref_to_time_to_norm_l1[level].values())
            plt.plot(times,values,color=COLOR_LIST[id])
            legendMarkers.append(Line2D([0], [0.1], color=COLOR_LIST[id]))
            legendEntries.append(f"level: {level}")
        plt.yscale("log")
        plt.xlabel("time")
        plt.ylabel("Increments")
        plt.legend(legendMarkers,legendEntries)
        plt.savefig(new_dict_name + ".png")
        plt.close()

    def plot_individual(self, name_directory: str) -> None:
        """Plot samples of 'time -> increment'."""
        if not os.path.isdir(name_directory):
            os.makedirs(name_directory)
        new_dict_name = name_directory + "/" + self.distance_name
        
        plt.figure()
        #plot samples
        for sample in self.seed_to_ref_to_time_to_incrementValue.keys():
            for id, level in enumerate(self.seed_to_ref_to_time_to_incrementValue[sample].keys()):
                times = list(self.seed_to_ref_to_time_to_incrementValue[sample][level].keys())
                values = list(self.seed_to_ref_to_time_to_incrementValue[sample][level].values())
                plt.plot(times,values,color=COLOR_LIST[id])
        #plot mean
        legendMarkers = []
        legendEntries = []
        for id, level in enumerate(self.ref_to_time_to_norm_l1.keys()):
            times = list(self.ref_to_time_to_norm_l1[level].keys())
            values = list(self.ref_to_time_to_norm_l1[level].values())
            plt.plot(times,values,color=COLOR_LIST[id])
            legendMarkers.append(Line2D([0], [0.1], color=COLOR_LIST[id]))
            legendEntries.append(f"level: {level}")
        plt.yscale("log")
        plt.xlabel("time")
        plt.ylabel("Increments")
        plt.legend(legendMarkers,legendEntries)
        plt.savefig(new_dict_name + "_samples.png")
        plt.close()
    
            
    def __str__(self):
        eoc_message = f"\n\nIncrement check on: \t{self.distance_name}"
        endtime = list(self.time_to_ref_to_norm_l1.keys())[-1]
        eoc_message += f"\nTime: \t{endtime}"
        eoc_message += "\n|__dt__|\t|____L1____|_EOC|\t|_DEVIATION|"  
        for level in self.ref_to_stepsize.keys():
            eoc_message += f"\n|{self.ref_to_stepsize[level]:5.04f}|\t"
            eoc_message += f"|{self.time_to_ref_to_norm_l1[endtime][level]:10.08f}|{self.time_to_ref_to_EOC_l1[endtime][level]:3.02f}|\t"
            eoc_message += f"|{self.time_to_ref_to_norm_SD[endtime][level]:10.08f}|"
        return eoc_message


