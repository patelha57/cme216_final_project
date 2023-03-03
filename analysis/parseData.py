import os
import glob 
import shutil 
import numpy as np
import scipy
import csv 


def parse_text_file(file_path):
    # Initialize variables
    variables = []
    node_count = 0
    element_count = 0
    data = []

    # Open and read the text file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Parse the metadata
    for line in lines:
        if line.startswith('TITLE'):
            continue
        elif line.startswith('VARIABLES'):
            continue
        elif line.startswith('ZONE'):
            zone_info = line.strip().split(',')
            node_count = int(zone_info[0].split('=')[1].strip())
            element_count = int(zone_info[1].split('=')[1].strip())
            continue
        else:
            break

    # Parse the data
    data_lines = lines[3:3+node_count]
    for line in data_lines:
        row = list(map(float, line.strip().split()))
        data.append(row)

    # Convert the data to a numpy array
    data_array = np.array(data)

    return data_array



def list_subfolders(folder):
    subfolders = []
    for dirpath, dirnames, filenames in os.walk(folder):
        for dirname in dirnames:
            print("Entering dir ......", dirname)
            subfolder = os.path.join(dirpath, dirname)
            subfolders.append(subfolder)
    return subfolders




# obtain the subfolder list of the root directory 
eulerRootPath = "/home/yrshen/Desktop/TLMF/euler_data/" 
#   ^dummpy path, for entering the absolute path to data root dir
eulerSubDirList = list_subfolders(eulerRootPath)
totalSubDirCount = len(eulerSubDirList)
print("Total %i directories found, parsing data\n"%totalSubDirCount)

# euler data

eulerData = []

# loop over dir list to read data
for ii in range(totalSubDirCount):
    curSubPath = eulerSubDirList[ii]
    fileRelPath = curSubPath + "/surface_flow.dat"
    curdataArray = parse_text_file(fileRelPath)
    print(np.shape(curdataArray))
    eulerData.append(curdataArray)



print(np.shape(eulerData))



