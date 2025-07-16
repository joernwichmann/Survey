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
    L1 = []
    L2 = []
    Linf = []
    SD = []
    for row in output:
        time.append(row[0])
        L1.append(row[1])
        L2.append(row[2])
        Linf.append(row[3])
        SD.append(row[4])
    
    return time, L1, L2, Linf, SD