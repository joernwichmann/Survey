import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tools import read_datafile, organize_output
from tqdm import tqdm
import os

### select the experiments whose data will be visualised 
from configs import lid_driven as cf

if __name__=="__main__":

    ### create output directory
    if not os.path.isdir(cf.OUTPUT_LOCATION):
            os.makedirs(cf.OUTPUT_LOCATION)
    print(f"Start plot of trajectories in dataformat '.{cf.TRAJ_FILEFORMAT}' with dpi '{cf.TRAJ_DPI}'")
    
    ################################################################### 
    ################## comparison between noise types
    ################################################################### 
    print(f'\nPlotting comparsion of different noise types')
    plt.figure()

    #plot individual energy trajectory
    print(f"\tPlotting individual trajectories")
    for n in tqdm(range(cf.NUMBER_SAMPLES)):
        for expID in cf.EXPERIMENTS.keys():
            file_location = cf.ROOT_LOCATION + cf.EXPERIMENTS[expID] + cf.IND_LOCATION + f"_{n}/" + cf.DATA_SOURCE
            complete_data = read_datafile(file_location)
            time, L1, L2, Linf, SD = organize_output(complete_data)
            plt.plot(time,L2,color = cf.COLOURS_INDIVIDUAL[expID],linewidth=cf.LINEWIDTH_INDIVIDUAL,alpha=cf.LINEOPACITY_INDIVIDUAL)

    #plot mean and standard deviations
    print(f"\tPlotting mean, standard devations and deterministic")
    for expID in cf.EXPERIMENTS.keys():
        file_location = cf.ROOT_LOCATION + cf.EXPERIMENTS[expID] + cf.MEAN_LOCATION + cf.DATA_SOURCE
        complete_data = read_datafile(file_location)
        time, L1, L2, Linf, SD = organize_output(complete_data)
        conf_plus = []
        conf_minus = []
        for l2, sd in zip(L2, SD):
            conf_plus.append(l2 + sd)
            conf_minus.append(l2-sd) 
        #mean energy trajectory
        plt.plot(time,L2,color = cf.COLOURS_MEAN[expID],linewidth=cf.LINEWIDTH_MEAN,alpha=cf.LINEOPACITY_MEAN)
        #standard deviation
        plt.plot(time,conf_plus,color = cf.COLOURS_MEAN[expID],linewidth=cf.LINEWIDTH_SD,alpha=cf.LINEOPACITY_SD,linestyle=cf.LINESTYLE_SD)
        plt.plot(time,conf_minus,color = cf.COLOURS_MEAN[expID],linewidth=cf.LINEWIDTH_SD,alpha=cf.LINEOPACITY_SD,linestyle=cf.LINESTYLE_SD)

        file_location = cf.ROOT_LOCATION + cf.EXPERIMENTS[expID] + cf.DET_LOCATION + cf.DATA_SOURCE
        complete_data = read_datafile(file_location)
        time, L1, L2, Linf, SD = organize_output(complete_data)
        plt.plot(time,L2,color = cf.BLACK,linestyle=cf.LINESTYLES_DET[expID],linewidth=1,alpha=cf.LINEOPACITY_MEAN)
        
    # styling the plot
    plt.ylabel("Kinetic energy")
    plt.xlabel("Time")
    plt.yscale(cf.TRAJ_YAXIS_SCALE)
    #plt.xlim(0.0,1.0)
    plt.tight_layout()
    
    ##legend
    # Create dummy Line2D objects for legend
    legendMarkers = []
    legendPvalues = []
    for expID in cf.EXPERIMENTS.keys():
        #add stochastic
        legendMarkers.append(Line2D([0], [0.1], color=cf.COLOURS_MEAN[expID], linewidth = cf.LINEWIDTH_MEAN))
        legendPvalues.append(f"p = {cf.P_VALUE[expID]}: sto")
        #add deterministic
        legendMarkers.append(Line2D([0], [0.1], color=cf.BLACK, linestyle=cf.LINESTYLES_DET[expID], linewidth = cf.LINEWIDTH_MEAN))
        legendPvalues.append(f"p = {cf.P_VALUE[expID]}: det")
    #plt.legend(legendMarkers,legendPvalues)

    plt.savefig(cf.OUTPUT_LOCATION + f"{cf.EXPERIMENT_NAME}-all-trajectories.{cf.TRAJ_FILEFORMAT}",dpi=cf.TRAJ_DPI)
    plt.close()
    print(f"Plot saved in '{cf.OUTPUT_LOCATION}{cf.EXPERIMENT_NAME}-all-trajectories.{cf.TRAJ_FILEFORMAT}'")

    ################################################################### 
    ############# Plot of energies for fixed noise type
    ###################################################################     
    # #plot individual energy trajectory
    for expID in cf.EXPERIMENTS.keys():
        print(f'\nGenerating trajectory plot for energies of {cf.EXPERIMENTS[expID]}')
        plt.figure()

        print(f"\tPlotting individual trajectories")
        for n in tqdm(range(cf.NUMBER_SAMPLES)):
            file_location = cf.ROOT_LOCATION + cf.EXPERIMENTS[expID] + cf.IND_LOCATION + f"_{n}/" + cf.DATA_SOURCE
            complete_data = read_datafile(file_location)
            time, L1, L2, Linf, SD = organize_output(complete_data)
            plt.plot(time,L2,color = cf.COLOURS_INDIVIDUAL[expID],linewidth=cf.LINEWIDTH_INDIVIDUAL,alpha=cf.LINEOPACITY_INDIVIDUAL)

        #plot mean and standard deviations
        print(f"\tPlotting mean and standard devations")
        file_location = cf.ROOT_LOCATION + cf.EXPERIMENTS[expID] + cf.MEAN_LOCATION + cf.DATA_SOURCE
        complete_data = read_datafile(file_location)
        time, L1, L2, Linf, SD = organize_output(complete_data)
        conf_plus = []
        conf_minus = []
        for l2, sd in zip(L2, SD):
            conf_plus.append(l2 + sd)
            conf_minus.append(l2-sd) 
        #mean energy trajectory
        plt.plot(time,L2,color = cf.COLOURS_MEAN[expID],linewidth=cf.LINEWIDTH_MEAN,alpha=cf.LINEOPACITY_MEAN)
        #standard deviation
        plt.plot(time,conf_plus,color = cf.COLOURS_MEAN[expID],linewidth=cf.LINEWIDTH_SD,alpha=cf.LINEOPACITY_SD,linestyle=cf.LINESTYLE_SD)
        plt.plot(time,conf_minus,color = cf.COLOURS_MEAN[expID],linewidth=cf.LINEWIDTH_SD,alpha=cf.LINEOPACITY_SD,linestyle=cf.LINESTYLE_SD)

        print(f"\tPlotting deterministic energy")
        file_location = cf.ROOT_LOCATION + cf.EXPERIMENTS[expID] + cf.DET_LOCATION + cf.DATA_SOURCE
        complete_data = read_datafile(file_location)
        time, L1, L2, Linf, SD = organize_output(complete_data)
        plt.plot(time,L2,color = cf.BLACK,linewidth=1,alpha=cf.LINEOPACITY_MEAN)
        

        ### styling the plot
        plt.ylabel("Kinetic energy", fontsize=cf.LABEL_FONTSIZE)
        plt.xlabel("Time", fontsize=cf.LABEL_FONTSIZE)
        plt.yscale(cf.TRAJ_YAXIS_SCALE)
        plt.yticks(fontsize=cf.TICK_FONTSIZE)
        plt.xticks(fontsize=cf.TICK_FONTSIZE)
        #plt.xlim(0.0,1.0)
        plt.tight_layout()
        plt.savefig(cf.OUTPUT_LOCATION+ f"{cf.EXPERIMENTS[expID]}-trajectories.{cf.TRAJ_FILEFORMAT}",dpi=cf.TRAJ_DPI)
        plt.close()
        print(f"Plot saved in '{cf.OUTPUT_LOCATION}{cf.EXPERIMENTS[expID]}-trajectories.{cf.TRAJ_FILEFORMAT}'")





