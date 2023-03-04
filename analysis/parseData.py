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
    #                   ^^ data extends from row 4 : row 4+#nodes
    #       ^ connectivities not parsed, can be considered latter 
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



def parse_data(fileRootPath, fileName):
    #parsing data into numpy array 
    #INNNNNPut
    #fileRootPath: string of the absolute path to dir containing subfolder of all data 
    #fileName: the name of the file to be parsed 
    #OUUUUTput
    #Data: numPy array contained (Ndata*Nnodes*Nvariables)

    Data = []

    subDirList = list_subfolders(fileRootPath)
    totalSubDirCount = len(subDirList)
    print("Total %i directories found, parsing data\n"%totalSubDirCount)


    # loop over dir list to read data
    for ii in range(totalSubDirCount):
        curSubPath = subDirList[ii]
        if fileName[0] != '/':
            fileName = '/'+fileName

        fileRelPath = curSubPath + fileName
        curdataArray = parse_text_file(fileRelPath)
        print(np.shape(curdataArray))
        Data.append(curdataArray)



    print(np.shape(Data))

    return Data


#unit tests 
### euler data


# obtain the subfolder list of the root directory 
eulerRootPath = "/home/yrshen/Desktop/TLMF/euler_data/" 
#   ^dummpy path, for entering the absolute path to data root dir
eulerSubDirList = list_subfolders(eulerRootPath)
totalSubDirCount = len(eulerSubDirList)
print("Total %i directories found, parsing data\n"%totalSubDirCount)

eulerData = []


# loop over dir list to read data
for ii in range(totalSubDirCount):
    curSubPath = eulerSubDirList[ii]
    fileRelPath = curSubPath + "/surface_flow.dat"
    curdataArray = parse_text_file(fileRelPath)
    print(np.shape(curdataArray))
    eulerData.append(curdataArray)



print(np.shape(eulerData))



### RANS data 

ransData = []

# obtain the subfolder list of the root directory 
ransRootPath = "/home/yrshen/Desktop/TLMF/rans_data/" 
#   ^dummpy path, for entering the absolute path to data root dir
ransSubDirList = list_subfolders(ransRootPath)
totalSubDirCount = len(ransSubDirList)
print("Total %i directories found, parsing data\n"%totalSubDirCount)


# loop over dir list to read data
for ii in range(totalSubDirCount):
    curSubPath = ransSubDirList[ii]
    fileRelPath = curSubPath + "/surface_flow.dat"
    curdataArray = parse_text_file(fileRelPath)
    print(np.shape(curdataArray))
    ransData.append(curdataArray)


print(np.shape(ransData))


# loop over dir list to read data 