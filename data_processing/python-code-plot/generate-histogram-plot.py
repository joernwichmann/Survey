import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams['patch.edgecolor'] = 'none'
import seaborn as sns
import os
from tools import read_datafile, organize_output


### select the experiments whose data will be visualised 
from configs import p_variation as cf
 
if __name__=="__main__":

    ### create output directory
    if not os.path.isdir(cf.OUTPUT_LOCATION):
            os.makedirs(cf.OUTPUT_LOCATION)
    print(f"Start plot of histograms in dataformat '.{cf.HIST_FILEFORMAT}' with dpi '{cf.HIST_DPI}'")
    all_data = {expID: [] for expID in cf.EXPERIMENTS.keys()}

    for expID in cf.EXPERIMENTS.keys():
        print(f'\nLoading data for {cf.EXPERIMENTS[expID]}') 
        for n in range(cf.NUMBER_SAMPLES):
            file_location = cf.ROOT_LOCATION + cf.EXPERIMENTS[expID] + cf.IND_LOCATION + f"_{n}/" + cf.DATA_SOURCE
            complete_data = read_datafile(file_location)
            time, L1, L2, Linf, SD = organize_output(complete_data)
            for t, l2 in zip(time,L2):
                if t >= cf.STATIONARY_TIME[expID]:
                    all_data[expID].append(l2)
        
        print(f'Plotting data for {cf.EXPERIMENTS[expID]}')
        if cf.LINEAR_PLOT:
            print(f'\tGenerating linear plot of histogram') 
            plt.figure()
            sns.histplot(all_data[expID], bins="auto", stat="probability", kde=True, color=cf.COLOURS_MEAN[expID], log_scale=(cf.HIST_XAXIS_LOG,False))
            plt.xlabel("Kinetic energy",fontsize=cf.LABEL_FONTSIZE)
            plt.ylabel("Probability",fontsize=cf.LABEL_FONTSIZE)
            plt.yticks(fontsize=cf.TICK_FONTSIZE)
            plt.xticks(fontsize=cf.TICK_FONTSIZE)
            plt.tight_layout()
            plt.savefig(cf.OUTPUT_LOCATION+f"hist-{cf.EXPERIMENTS[expID]}.{cf.HIST_FILEFORMAT}",dpi=cf.HIST_DPI)
            plt.close()
            print(f"Plot saved in '{cf.OUTPUT_LOCATION}hist-{cf.EXPERIMENTS[expID]}.{cf.HIST_FILEFORMAT}.{cf.TRAJ_FILEFORMAT}'")        
        
        if cf.LOG_PLOT:
            print(f'\tGenerating log plot of histogram') 
            plt.figure()
            sns.histplot(all_data[expID], bins="auto", stat="probability", kde=True, color=cf.COLOURS_MEAN[expID], log_scale=(True,True))
            plt.xlabel("Kinetic energy")
            plt.tight_layout()
            plt.savefig(cf.OUTPUT_LOCATION+f"hist-{cf.EXPERIMENTS[expID]}-log.{cf.HIST_FILEFORMAT}",dpi=cf.HIST_DPI)
            plt.close()
            print(f"Plot saved in '{cf.OUTPUT_LOCATION}hist-{cf.EXPERIMENTS[expID]}-log.{cf.HIST_FILEFORMAT}.{cf.TRAJ_FILEFORMAT}'")
    
        

    print(f'\nPlotting data for noise comparisons') 
    ##legend
    # Create dummy Line2D objects for legend
    legendMarkers = [Line2D([0], [0.1], color=cf.COLOURS_MEAN[expID], linewidth = cf.LINEWIDTH_MEAN) for expID in cf.EXPERIMENTS.keys()]
    legendPvalues = [f"p = {cf.P_VALUE[expID]}" for expID in cf.EXPERIMENTS.keys()]
    #legendPvalues.append("det")
    if cf.LINEAR_PLOT:
        print(f'\tGenerating linear plot of histogram')
        plt.figure()
        for expID in cf.EXPERIMENTS.keys():
            sns.histplot(all_data[expID], bins="auto", stat="probability", kde=True, color=cf.COLOURS_MEAN[expID], log_scale=(cf.HIST_XAXIS_LOG,False))
            #plt.vlines(x=cf.STATIONARY_ENERGY[noise],ymin=0,ymax=cf.YMAX[noise],colors="black",linestyles="solid")
        plt.xlabel("Kinetic energy")
        #plt.legend(legendMarkers,legendPvalues)
        plt.tight_layout()
        plt.savefig(cf.OUTPUT_LOCATION+f"hist-{cf.EXPERIMENT_NAME}-all.{cf.HIST_FILEFORMAT}",dpi=cf.HIST_DPI)
        plt.close()
        print(f"Plot saved in '{cf.OUTPUT_LOCATION}hist-{cf.EXPERIMENT_NAME}-all.{cf.HIST_FILEFORMAT}'")

    if cf.LOG_PLOT:
        print(f'\tGenerating log plot of histogram')
        plt.figure()
        for expID in cf.EXPERIMENTS.keys():
            sns.histplot(all_data[expID], bins="auto", stat="probability", kde=True, color=cf.COLOURS_MEAN[expID], log_scale=(True,True))
            #plt.vlines(x=cf.STATIONARY_ENERGY[noise],ymin=0,ymax=cf.YMAX[noise],colors="black",linestyles="solid")
        plt.xlabel("Kinetic energy")
        plt.legend(legendMarkers,legendPvalues)
        plt.tight_layout()
        plt.savefig(cf.OUTPUT_LOCATION+f"hist-{cf.EXPERIMENT_NAME}-all-log.{cf.HIST_FILEFORMAT}",dpi=cf.HIST_DPI)
        plt.close()
        print(f"Plot saved in '{cf.OUTPUT_LOCATION}hist-{cf.EXPERIMENT_NAME}-all-log.{cf.HIST_FILEFORMAT}'")