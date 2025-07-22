import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams['patch.edgecolor'] = 'none'
import seaborn as sns
import numpy as np
import pandas as pd
import os
from tools_point_statistics import read_datafile, organize_output_single, organize_output
 
### select the experiments whose data will be visualised 
from configs import lid_driven_point_statistics as cf

if __name__=="__main__":
    ### create output directory
    if not os.path.isdir(cf.OUTPUT_LOCATION):
            os.makedirs(cf.OUTPUT_LOCATION)

    print(f"Start plot of histograms in dataformat '.{cf.HIST_FILEFORMAT}' with dpi '{cf.HIST_DPI}'")
    all_data_x = dict()
    all_data_y = dict()

    det_data_x = dict()
    det_data_y = dict()
    for expID in cf.EXPERIMENTS.keys():
        DESIRED_EXPERIMENT = 3
        if True: #expID == DESIRED_EXPERIMENT:
            data_x = []
            data_y = []
            print(f'\nLoading data for {cf.EXPERIMENTS[expID]}')
            #load stochastic
            for n in range(cf.NUMBER_SAMPLES):
                file_location = cf.ROOT_LOCATION + cf.EXPERIMENTS[expID] + cf.IND_LOCATION + f"_{n}" + cf.DATA_SOURCE
                complete_data = read_datafile(file_location)
                time, val_0, val_1 = organize_output_single(complete_data)
                for t, v0, v1 in zip(time,val_0,val_1):
                    if t >= cf.STATIONARY_TIME[expID]:
                        data_x.append(v0)
                        data_y.append(v1)

            all_data_x[expID] = np.array(data_x)
            all_data_y[expID] = np.array(data_y)

            ##load deterministic
            file_location = cf.ROOT_LOCATION + cf.EXPERIMENTS[expID] + cf.DET_LOCATION + cf.DATA_SOURCE
            complete_data = read_datafile(file_location)
            _, det_data_x[expID], det_data_y[expID] , _, _ = organize_output(complete_data)

            print(f'Plotting data for {cf.EXPERIMENTS[expID]}') 
            plt.figure()
            #sns.histplot(all_data[expID], bins="auto", stat="probability", kde=True, color=cf.COLOURS_MEAN[expID], log_scale=(True,False))
            sns.jointplot(x = all_data_x[expID], y=all_data_y[expID], color=cf.COLOURS_MEAN[expID], kind="kde")
            plt.plot(det_data_x[expID][-1],det_data_y[expID][-1], marker = "o", markeredgecolor = cf.BLACK, color=cf.BLACK)

            # create manual symbols for legend
            line = Line2D([0], [0], label=cf.P_VALUE[expID], color=cf.COLOURS_MEAN[expID])
            point = Line2D([0], [0], label='deterministic', marker = "o", markeredgecolor = cf.BLACK, color=cf.BLACK, linestyle='')

            #plt.xticks(np.arange(np.floor(min(all_data_x[expID])), np.ceil(max(all_data_x[expID])), 1.0))
            plt.xlabel("u",fontsize=cf.LABEL_FONTSIZE)
            plt.ylabel("v",fontsize=cf.LABEL_FONTSIZE)
            plt.xticks(fontsize=cf.TICK_FONTSIZE)
            plt.yticks(fontsize=cf.TICK_FONTSIZE)
            plt.ticklabel_format(axis='x', style='sci')       
            ax = plt.gca()
            if False:
                if expID == 1:
                    ax.set_xlim(-1.2,1)
                    ax.set_ylim(-1.2,1)
                if expID == 2:
                    ax.set_xlim(-0.18,0.15)
                    ax.set_ylim(-0.12,0.12)
                    xticks = ax.xaxis.get_major_ticks()
                    xticks[2].set_visible(False)
                    xticks[3].set_visible(False)
                    xticks[5].set_visible(False)
                    xticks[6].set_visible(False)
                if expID == 3:
                    ax.set_xlim(-0.02,0.002)
                    ax.set_ylim(-0.002,0.02)
                    xticks = ax.xaxis.get_major_ticks()
                    xticks[1].set_visible(False)
                    xticks[3].set_visible(False)
                    
            plt.grid()
            #plt.legend([],[], frameon=False)
            plt.legend(handles=[line,point],fontsize=cf.LABEL_FONTSIZE)
            #plt.title('velocity at (0.5,0.75)')
            #plt.axes([-1,-1,2,2])
            plt.tight_layout()
            plt.savefig(cf.OUTPUT_LOCATION + f"hist-point-{cf.EXPERIMENTS[expID]}.{cf.HIST_FILEFORMAT}",dpi=cf.HIST_DPI)
            plt.close()        
            print(f"Plot saved in '{cf.OUTPUT_LOCATION}hist-point-{cf.EXPERIMENTS[expID]}.{cf.HIST_FILEFORMAT}'")

    compare = True
    if compare:
        print(f'\nPlotting data for noise comparisons') 

        #build pandas.dataframe
        prebuild = []
        for expID in cf.EXPERIMENTS.keys():
            for x,y in zip(all_data_x[expID],all_data_y[expID]):
                prebuild.append([f"{cf.P_VALUE[expID]}",x,y])
        build_data = pd.DataFrame(data=prebuild, columns=["Noise","u","v"])

        #build color palette
        colors = [cols for cols in cf.COLOURS_MEAN.values()]
        customPalette = sns.set_palette(sns.color_palette(colors))

        
        plt.figure()
        sns.jointplot(data = build_data, x = "u", y="v",  hue="Noise", kind="kde", palette=customPalette)
        for expID in cf.EXPERIMENTS.keys():
            plt.plot(det_data_x[expID][-1],det_data_y[expID][-1], marker = "o", markeredgecolor = cf.BLACK, color=cf.COLOURS_MEAN[expID])
        #plt.title('velocity at (0.5,0.75)')
        plt.xlabel("u",fontsize=cf.LABEL_FONTSIZE)
        plt.ylabel("v",fontsize=cf.LABEL_FONTSIZE)
        plt.xticks(fontsize=cf.TICK_FONTSIZE)
        plt.yticks(fontsize=cf.TICK_FONTSIZE)
        plt.grid()
        #plt.legend([],[], frameon=False)
        plt.tight_layout()
        plt.savefig(cf.OUTPUT_LOCATION + f"hist-point-{cf.EXPERIMENT_NAME}-all.{cf.HIST_FILEFORMAT}",dpi=cf.HIST_DPI)
        plt.close()
        print(f"Plot saved in '{cf.OUTPUT_LOCATION}hist-point-{cf.EXPERIMENT_NAME}-all.{cf.HIST_FILEFORMAT}'")