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


def read_tecplot_file(filename):
    """Function. Read tabular data for multiple zones from ASCII Tecplot file.
    Parameters
        filename - Tecplot data file
    Outputs
        data - history data dictionary
    """
    data = {}
    with open(filename, 'r') as file:
        variables = []
        lines = file.readlines()
        for idx, line in enumerate(lines):

            # ignore any comment line
            if '#' in line:
                line = line.split('#')[0]
                if not line:
                    continue

            # skip title line if there is one
            elif 'title' in line.lower():
                continue

            # read list of variable names
            elif 'variable' in line.lower():
                line = line.split('=', 1)[1]
                if '\\' in line:
                    line = lines[idx + 1]
                variables = [var.strip().strip("\"") for var in line.split(',')]
                continue

            # check to see if there is a new zone definition
            elif 'zone' in line.lower():
                # extract zone name
                # zone_name = line.split('=', 1)[1].strip()
                # zone_name = zone_name.strip("\"")
                zone_name = 'ZONE0'      # temporary hack
                data[zone_name] = {}

                # if variable names are defined, assign an empty numpy array to that variable
                if variables:
                    for var in variables:
                        data[zone_name][var] = np.empty(0)
                continue

            # sort data out into the different variables
            else:
                if '"' in line:
                    continue

                # create a dummy zone named ZONE0 if no zone is defined
                if 'zone_name' not in locals():
                    zone_name = 'ZONE0'
                    data[zone_name] = {}

                    # assign an empty numpy array if variable names are defined
                    if variables:
                        for var in variables:
                            data[zone_name][var] = np.empty(0)

                # comma-delimited data
                if ',' in line:
                    line_list = [val.strip() for val in line.split(',')]

                # space-delimited data
                else:
                    line_list = line.split()

                for i, val in enumerate(line_list):
                    if variables:
                        data[zone_name][variables[i]] = np.append(data[zone_name][variables[i]], float(val))

                    # name variables according to order of appearance if no variables are defined
                    else:
                        if not data[zone_name]:
                            for j in range(0, len(line.split())):
                                data[zone_name]['var' + str(j)] = np.empty(0)
                        data[zone_name]['var' + str(i)] = np.append(data[zone_name]['var' + str(i)], float(val))

    return data


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


def parse_data(fileRootPath, fileName, variables):
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
        curdata_dict = read_tecplot_file(fileRelPath)
        data_dict = curdata_dict['ZONE0']

        curdataArray = np.array([], dtype=float)
        for variable, data in data_dict.items():
            if variable in variables:
                curdataArray = np.append(curdataArray, data)

        curdataArray = np.reshape(curdataArray, (-1, len(variables)))

        Data.append(curdataArray)

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
