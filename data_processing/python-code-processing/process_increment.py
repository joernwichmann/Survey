from tools_increment import read_datafile, organize_output, organize_output_by_stepsize, glue_sto_and_det
import os
import csv

### select the experiments whose data will be visualised 
from configs import lid_driven_inc as cf

if __name__=="__main__":
    print(f"Start processing of increments")
    ### create output directory
    if not os.path.isdir(cf.OUTPUT_LOCATION):
        os.makedirs(cf.OUTPUT_LOCATION)
    
    headerFile1 = ["time", "mean", "meanPlusSd", "meanMinusSd","deterministic"]
    headerFile2 = ["stepsize", "eoc"]

    for expID in cf.EXPERIMENTS.keys():
        #load stochastic data
        file_location_sto = cf.ROOT_LOCATION + cf.EXPERIMENTS[expID] + cf.MEAN_LOCATION + cf.DATA_SOURCE
        complete_data_sto = read_datafile(file_location_sto)
        stepsize_to_timeMeanSD_sto, stepsize_to_eoc_sto = organize_output_by_stepsize(cf.EOC_AT_TIME,complete_data_sto)

        file_location_det = cf.ROOT_LOCATION + cf.EXPERIMENTS[expID] + cf.DET_LOCATION + cf.DATA_SOURCE
        complete_data_det = read_datafile(file_location_det)
        stepsize_to_timeMeanSD_det, _ = organize_output_by_stepsize(cf.EOC_AT_TIME,complete_data_det)

        stepsize_to_timeMeanSDGlued = glue_sto_and_det(stepsize_to_timeMeanSD_sto,stepsize_to_timeMeanSD_det)
        print(f"Loaded data for {cf.EXPERIMENTS[expID]}\nStart processing")

        #create ouput directory for each experiment
        output_dir = cf.OUTPUT_LOCATION + cf.EXPERIMENTS[expID]
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        for stepsize in stepsize_to_timeMeanSD_sto:
            outfile = output_dir + f"/stepsize_{stepsize}.csv"
            
            with open(outfile,"w",newline="") as file:
                writer = csv.writer(file)
                writer.writerow(headerFile1)
                writer.writerows(stepsize_to_timeMeanSDGlued[stepsize])

        outfile = output_dir + f"/eoc_at_{cf.EOC_AT_TIME}.csv"
        with open(outfile,"w",newline="") as file:
            writer = csv.writer(file)
            writer.writerow(headerFile2)
            for stepsize in stepsize_to_eoc_sto.keys():
                writer.writerow([stepsize,stepsize_to_eoc_sto[stepsize]])
        print(f"Finished processing of {cf.EXPERIMENTS[expID]}\nData is stored in {output_dir}/")