import csv

def read_datafile(location: str):
    with open(location, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        #skip header
        next(reader)

        output = []
        for row in reader:
            row_proc = [float(entry) for entry in row]
            output.append(row_proc) 
    return output
   
def organize_output(output):
    time = []
    stepsize = []
    mean = []
    eoc_mean = []
    sd = []
    for row in output:
        time.append(row[0])
        stepsize.append(row[1])
        mean.append(row[2])
        eoc_mean.append(row[3])
        sd.append(row[4])
    
    return time, stepsize, mean, eoc_mean, sd

def organize_output_by_stepsize(eoc_at_time,output):
    #get unique stepsize
    _, list_of_stepsizes, _, _, _ = organize_output(output)
    unique_stepsizes = set(list_of_stepsizes)

    #initialise dictionaries
    stepsize_to_timeMeanSD = {stepsize: [] for stepsize in unique_stepsizes}
    stepsize_to_eoc = dict()

    #fill dictionaries
    for row in output:
        stepsize_to_timeMeanSD[row[1]].append([row[0],row[2],row[2]-row[4],row[2]+row[4]])
        if row[0] == eoc_at_time:
            stepsize_to_eoc[row[1]] = row[3]
    
    return stepsize_to_timeMeanSD, stepsize_to_eoc

def glue_sto_and_det(stepsize_to_timeMeanSD_sto,stepsize_to_timeMeanSD_det):
    stepsize_to_timeMeanSDGlued = dict()
    for stepsize in stepsize_to_timeMeanSD_sto.keys():
        glued_list = []
        for timeMeanSD_sto, timeMeanSD_det in zip(stepsize_to_timeMeanSD_sto[stepsize],stepsize_to_timeMeanSD_det[stepsize]):
            glued_list.append(timeMeanSD_sto + [timeMeanSD_det[1]])
        stepsize_to_timeMeanSDGlued[stepsize] = glued_list

    return stepsize_to_timeMeanSDGlued