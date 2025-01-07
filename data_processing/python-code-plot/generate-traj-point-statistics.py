import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tools_point_statistics import read_datafile, organize_output, organize_output_single
from tqdm import tqdm
import os

### select the experiments whose data will be visualised 
from configs import p_variation_point_statistics as cf

if __name__=="__main__":
    print(f"Start plot of trajectories in dataformat '.{cf.TRAJ_FILEFORMAT}' with dpi '{cf.TRAJ_DPI}'")
    
    ### create output directory
    if not os.path.isdir(cf.OUTPUT_LOCATION):
            os.makedirs(cf.OUTPUT_LOCATION)


    ################################################################### 
    ################## comparison between noise types
    ################################################################### 
    print(f'\nPlotting comparsion of different noise types')
    plt.figure()

    #plot individual energy trajectory
    print(f"\tPlotting individual trajectories")
    for n in tqdm(range(cf.NUMBER_SAMPLES)):
        for expID in cf.EXPERIMENTS.keys():
            file_location = cf.ROOT_LOCATION + cf.EXPERIMENTS[expID] + cf.IND_LOCATION + f"_{n}" + cf.DATA_SOURCE
            complete_data = read_datafile(file_location)
            time, val_0, val_1 = organize_output_single(complete_data)
            plt.plot(val_0,val_1,color = cf.COLOURS_INDIVIDUAL[expID],linewidth=cf.LINEWIDTH_INDIVIDUAL,alpha=cf.LINEOPACITY_INDIVIDUAL)

    
    #plot stochastic
    print(f"\tPlotting mean, standard devations and deterministic")
    for expID in cf.EXPERIMENTS.keys():
        file_location = cf.ROOT_LOCATION + cf.EXPERIMENTS[expID] + cf.MEAN_LOCATION + cf.DATA_SOURCE
        complete_data = read_datafile(file_location)
        time, mean_0, mean_1, sd_0, sd_1 = organize_output(complete_data)

        #mean energy trajectory
        plt.plot(mean_0,mean_1,color = cf.COLOURS_MEAN[expID],linewidth=cf.LINEWIDTH_MEAN,alpha=cf.LINEOPACITY_MEAN)

    #plot deterministic
    legendMarkers = [] 
    legendEntries = []
    det_mean_x = dict()
    det_mean_y = dict()
    for expID in cf.EXPERIMENTS.keys():
        #plot time evolution
        file_location = cf.ROOT_LOCATION + cf.EXPERIMENTS[expID] + cf.DET_LOCATION + cf.DATA_SOURCE
        complete_data = read_datafile(file_location)
        time, mean_0, mean_1, sd_0, sd_1 = organize_output(complete_data)
        plt.plot(mean_0,mean_1,color = cf.BLACK,linestyle=cf.LINESTYLES_DET[expID],linewidth=1,alpha=cf.LINEOPACITY_MEAN)
        #store deterministic stationary energy
        det_mean_x[expID] = mean_0[-1]
        det_mean_y[expID] = mean_1[-1]
        #define legend
        legendMarkers.append(Line2D([0], [0.1], color=cf.COLOURS_MEAN[expID], linewidth = cf.LINEWIDTH_MEAN))
        legendEntries.append(f"p = {cf.P_VALUE[expID]}")

    for expID in cf.EXPERIMENTS.keys():   
        plt.plot(det_mean_x[expID],det_mean_y[expID], marker = "o",markeredgecolor = cf.BLACK,color=cf.COLOURS_MEAN[expID])
        
        
    
    # styling the plot
    plt.ylabel("y")
    plt.xlabel("x")
    #plt.title('velocity at (0.5,0.75)')
    #plt.xlim(0.0,1.0)
    #plt.legend(legendMarkers,legendEntries)
    plt.grid()
    plt.tight_layout()
    plt.savefig(cf.OUTPUT_LOCATION + f"point-{cf.EXPERIMENT_NAME}-all-trajectories.{cf.TRAJ_FILEFORMAT}",dpi=cf.TRAJ_DPI)
    plt.close()
    print(f"Plot saved in '{cf.OUTPUT_LOCATION}point-{cf.EXPERIMENT_NAME}-all-trajectories.{cf.TRAJ_FILEFORMAT}'")

    ################################################################### 
    ############# Plot of energies for fixed noise type
    ###################################################################     
    # #plot individual energy trajectory
    for expID in cf.EXPERIMENTS.keys():
        print(f'\nGenerating trajectory plot for Experiment {cf.EXPERIMENTS[expID]}')
        plt.figure()

        print(f"\tPlotting individual trajectories")
        for n in tqdm(range(cf.NUMBER_SAMPLES)):
            file_location = cf.ROOT_LOCATION + cf.EXPERIMENTS[expID] + cf.IND_LOCATION + f"_{n}" + cf.DATA_SOURCE
            complete_data = read_datafile(file_location)
            time, val_0, val_1 = organize_output_single(complete_data)
            plt.plot(val_0,val_1,color = cf.COLOURS_INDIVIDUAL[expID],linewidth=cf.LINEWIDTH_INDIVIDUAL,alpha=cf.LINEOPACITY_INDIVIDUAL)

        #plot stochastic mean
        print(f"\tPlotting mean energy")
        file_location = cf.ROOT_LOCATION + cf.EXPERIMENTS[expID] + cf.MEAN_LOCATION + cf.DATA_SOURCE
        complete_data = read_datafile(file_location)
        time, mean_0, mean_1, sd_0, sd_1  = organize_output(complete_data)
        plt.plot(mean_0,mean_1,color = cf.COLOURS_MEAN[expID],linewidth=cf.LINEWIDTH_MEAN,alpha=cf.LINEOPACITY_MEAN)

        #plot deterministic
        print(f"\tPlotting deterministic energy")
        file_location = cf.ROOT_LOCATION + cf.EXPERIMENTS[expID] + cf.DET_LOCATION + cf.DATA_SOURCE
        complete_data = read_datafile(file_location)
        time, mean_0, mean_1, sd_0, sd_1 = organize_output(complete_data)
        plt.plot(mean_0,mean_1,color = cf.BLACK,linewidth=1,alpha=cf.LINEOPACITY_MEAN)
        plt.plot(mean_0[-1],mean_1[-1], marker = "o",color=cf.BLACK)
        

        ### styling the plot
        plt.ylabel("y", fontsize=cf.LABEL_FONTSIZE)
        plt.xlabel("x", fontsize=cf.LABEL_FONTSIZE)
        #plt.title('velocity at (0.5,0.75)', fontsize=cf.LABEL_FONTSIZE)
        plt.yticks(fontsize=cf.TICK_FONTSIZE)
        plt.xticks(fontsize=cf.TICK_FONTSIZE)
        #plt.xlim(0.0,1.0)
        #plt.legend([legendMarkers[k]],[legendEntries[k]])
        plt.grid()
        plt.tight_layout()
        plt.savefig(cf.OUTPUT_LOCATION + f"point-{cf.EXPERIMENTS[expID]}-trajectories.{cf.TRAJ_FILEFORMAT}",dpi=cf.TRAJ_DPI)
        plt.close()
        print(f"Plot saved in '{cf.OUTPUT_LOCATION}point-{cf.EXPERIMENTS[expID]}-trajectories.{cf.TRAJ_FILEFORMAT}'")





