import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tools_inc import read_datafile, organize_output
import os

### select the experiments whose data will be visualised 
from configs import p_variation_inc as cf

if __name__=="__main__":
    print(f"Start plot of increments in dataformat '.{cf.TRAJ_FILEFORMAT}' with dpi '{cf.TRAJ_DPI}'")

    ### create output directory
    if not os.path.isdir(cf.OUTPUT_LOCATION):
            os.makedirs(cf.OUTPUT_LOCATION)
    
    ################################################################### 
    ################## comparison between noise types
    ################################################################### 
    print(f'\nPlotting comparsion of different growth rates')
    plt.figure()

    legendMarkers = [] 
    legendEntries = []
    #plot mean and standard deviations
    print(f"\tPlotting mean, standard devations and deterministic")
    for expID in cf.EXPERIMENTS.keys():
        file_location = cf.ROOT_LOCATION + cf.EXPERIMENTS[expID] + cf.MEAN_LOCATION + cf.DATA_SOURCE
        complete_data = read_datafile(file_location)
        time, stepsize, mean, eoc_mean, sd = organize_output(complete_data)

        #mean 
        plt.plot(time,mean,color = cf.COLOURS_MEAN[expID],linewidth=cf.LINEWIDTH_MEAN,alpha=cf.LINEOPACITY_MEAN)
        
        #standard deviation
        sd_minus = [mv + sv for mv, sv in zip(mean, sd)]
        sd_plus = [mv - sv for mv, sv in zip(mean, sd)]
        plt.plot(time,sd_plus,color = cf.COLOURS_MEAN[expID],linewidth=cf.LINEWIDTH_SD,alpha=cf.LINEOPACITY_SD,linestyle=cf.LINESTYLE_SD)
        plt.plot(time,sd_minus,color = cf.COLOURS_MEAN[expID],linewidth=cf.LINEWIDTH_SD,alpha=cf.LINEOPACITY_SD,linestyle=cf.LINESTYLE_SD)
        
        #plot det
        file_location = cf.ROOT_LOCATION + cf.EXPERIMENTS[expID] + cf.DET_LOCATION + cf.DATA_SOURCE
        complete_data = read_datafile(file_location)
        time, stepsize, mean, eoc_mean, sd = organize_output(complete_data)
        plt.plot(time,mean,color = cf.BLACK,linewidth=1,alpha=cf.LINEOPACITY_MEAN)
        
        #define legend
        legendMarkers.append(Line2D([0], [0.1], color=cf.COLOURS_MEAN[expID], linewidth = cf.LINEWIDTH_MEAN))
        legendEntries.append(f"p = {cf.P_VALUE[expID]}")
    
    # styling the plot
    plt.ylabel("Increments")
    plt.xlabel("Time")
    plt.yscale("log")
    #plt.xlim(0.0,1.0)
    plt.legend(legendMarkers,legendEntries)
    plt.tight_layout()
    plt.savefig(cf.OUTPUT_LOCATION+f"{cf.EXPERIMENT_NAME}-comp-increments.{cf.TRAJ_FILEFORMAT}",dpi=cf.TRAJ_DPI)
    plt.close()
    print(f"Plot saved in '{cf.OUTPUT_LOCATION}{cf.EXPERIMENT_NAME}-comp-increments.{cf.TRAJ_FILEFORMAT}'")

    ################################################################### 
    ############# Plot of energies for fixed noise type
    ###################################################################     
    # #plot individual energy trajectory
    for k, expID in enumerate(cf.EXPERIMENTS.keys()):
        print(f'\nGenerating increment plot for Experiment "{cf.EXPERIMENTS[expID]}"')
        plt.figure()

        #plot mean and standard deviations
        print(f"\tPlotting mean, standard devation, and deterministic")
        file_location = cf.ROOT_LOCATION + cf.EXPERIMENTS[expID] + cf.MEAN_LOCATION + cf.DATA_SOURCE
        complete_data = read_datafile(file_location)
        time, stepsize, mean, eoc_mean, sd = organize_output(complete_data)

        #mean 
        plt.plot(time,mean,color = cf.COLOURS_MEAN[expID],linewidth=cf.LINEWIDTH_MEAN,alpha=cf.LINEOPACITY_MEAN)
        
        #standard deviation
        sd_minus = [mv + sv for mv, sv in zip(mean, sd)]
        sd_plus = [mv - sv for mv, sv in zip(mean, sd)]
        plt.plot(time,sd_plus,color = cf.COLOURS_MEAN[expID],linewidth=cf.LINEWIDTH_SD,alpha=cf.LINEOPACITY_SD,linestyle=cf.LINESTYLE_SD)
        plt.plot(time,sd_minus,color = cf.COLOURS_MEAN[expID],linewidth=cf.LINEWIDTH_SD,alpha=cf.LINEOPACITY_SD,linestyle=cf.LINESTYLE_SD)
        
        #plot det
        file_location = cf.ROOT_LOCATION + cf.EXPERIMENTS[expID] + cf.DET_LOCATION + cf.DATA_SOURCE
        complete_data = read_datafile(file_location)
        time, stepsize, mean, eoc_mean, sd = organize_output(complete_data)
        plt.plot(time,mean,color = cf.BLACK,linewidth=1,alpha=cf.LINEOPACITY_MEAN)
        

        # styling the plot
        plt.ylabel("Increments")
        plt.xlabel("Time")
        plt.yscale("log")
        #plt.xlim(0.0,1.0)
        plt.legend(legendMarkers,legendEntries)
        plt.tight_layout()
        plt.savefig(cf.OUTPUT_LOCATION+f"{cf.EXPERIMENTS[expID]}-increments.{cf.TRAJ_FILEFORMAT}",dpi=cf.TRAJ_DPI)
        plt.close()
        print(f"Plot saved in '{cf.OUTPUT_LOCATION}{cf.EXPERIMENTS[expID]}-increments.{cf.TRAJ_FILEFORMAT}'")





