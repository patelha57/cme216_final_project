import os
import numpy as np


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


def read_tecplot_history_file(filename):
    """Function. Read tabular data for multiple zones from ASCII Tecplot file.
    Parameters
        filename - Tecplot data file
    Outputs
        data - history data dictionary
    """
    # extract variable names
    with open(filename, 'r') as file:
        file.readline()
        line = file.readline()
        variables = [var.strip().strip("\"") for var in line.split(',')]

    # organize the data
    data = {}
    hist_data = np.loadtxt(filename, delimiter=',', skiprows=2, unpack=True)
    for idx, variable in enumerate(variables):
        data[variable] = hist_data[idx]

    return data


def get_force_data(filename='forces_breakdown.dat'):
    """Function. Extract force coefficient data from forces breakdown file.
    Parameters
        filename - forces breakdown file
    Outputs
        force_data - force data dictionary
    """
    force_data = {}
    with open(filename) as file:
        line = file.readline()
        while line:
            if 'Total' in line.split(':')[0]:
                coeff = line.split(':')[0].split('Total')[1]
                split_text = line.split('|')
                for text in split_text:
                    if 'Total' in text:
                        force_data[coeff] = float(text.strip().split()[-1])
                    elif 'Pressure' in text:
                        force_data[coeff + 'p'] = float(text.strip().split()[-1])
                    elif 'Friction' in text:
                        force_data[coeff + 'v'] = float(text.strip().split()[-1])
                if 'CFy' in line:
                    break
            line = file.readline()

    return force_data

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
