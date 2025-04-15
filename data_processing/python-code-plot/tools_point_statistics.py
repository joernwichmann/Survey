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
    mean_0 = []
    mean_1 = []
    sd_0 = []
    sd_1 = []
    for row in output:
        time.append(row[0])
        mean_0.append(row[1])
        mean_1.append(row[2])
        sd_0.append(row[3])
        sd_1.append(row[4])
    return time, mean_0, mean_1, sd_0, sd_1

def organize_output_single(output):
    time = []
    val_0 = []
    val_1 = []
    for row in output:
        time.append(row[0])
        val_0.append(row[1])
        val_1.append(row[2])
    return time, val_0, val_1