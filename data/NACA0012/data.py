import os
import glob
import shutil
import numpy as np

from utils import mpi

import pysu2

################################################################################
# Pre-processing
################################################################################

FMT = '% .16e'
FMT_HEADER = '%22.22s'

primitive_vars = ['DENSITY', 'VELOCITY-X', 'VELOCITY-Y', 'PRESSURE', 'TEMPERATURE', 'VISCOSITY']
state_vars = ['DENSITY', 'MOMENTUM-X', 'MOMENTUM-Y', 'ENERGY']

primitives_header = ', '.join((FMT_HEADER % var) for var in primitive_vars)[1:]
states_header     = ', '.join((FMT_HEADER % var) for var in state_vars)[1:]

if mpi.isroot():
    # remove old folders
    for folder in ['output_files']:
        if os.path.exists(folder):
            shutil.rmtree(folder)

    # create folder for simulation result files
    output_path = f'{os.getcwd()}/output_files'
    if not os.path.exists(output_path):
        os.mkdir(output_path)

################################################################################
# SU2 setup
################################################################################

# user-defined options
config = 'euler.cfg'
nzone = 1
marker_id = 0

# specify range of angle of attack
alpha_delta = 0.1
alpha_min   = -5
alpha_max   = 5
num_samples = int(np.ceil((alpha_max - alpha_min) / alpha_delta)) + 1
alpha_data  = np.linspace(alpha_min, alpha_max, num=num_samples, dtype=float)

# specify range of Mach numbers
mach_delta  = 0.04
mach_min    = 0.3
mach_max    = 0.7
num_samples = int(np.ceil((mach_max - mach_min) / mach_delta)) + 1
mach_data   = np.linspace(mach_min, mach_max, num=num_samples, dtype=float)

# mach_data = np.array([0.3, 0.7])   # this is for testing purposes only
# alpha_data = np.array([0., 1.])    # this is for testing purposes only

mpi.barrier()

################################################################################
# SU2 execution
################################################################################

# initialize flow solver
solver = pysu2.CSinglezoneDriver(config, nzone, mpi.COMM)

# set flight conditions
for mach in mach_data:
    for alpha in alpha_data:
        # set angle of attack and Mach number
        solver.SetAngleOfAttack(alpha)
        solver.SetMachNumber(mach)

        if mpi.isroot():
            print(f'Finished setting Mach number = {mach} and angle of attack = {alpha} [deg] for CFD run...')

        # run CFD iterations
        solver.ResetConvergence()
        solver.Preprocess(0)
        solver.Run()
        solver.Postprocess()
        solver.Monitor(0)
        solver.Output(0)

        mpi.barrier()

        # extract flow data on the airfoil surface
        primitives = np.asarray(solver.GetMarkerPrimitiveStates(marker_id), dtype=float)
        temps      = np.asarray(solver.GetMarkerTemperatures(marker_id), dtype=float)
        viscosity  = np.asarray(solver.GetMarkerLaminarViscosities(marker_id), dtype=float)
        states     = np.asarray(solver.GetMarkerStates(marker_id), dtype=float)

        # extract aerodynamic drag coefficient
        Cd = solver.GetDrag(coefficient=True)

        if mpi.isroot():
            print(f'Aerodynamic drag coefficient = {Cd}')

            # save simulation results to file
            np.savetxt('primitives.dat', np.hstack((primitives, temps, viscosity)), fmt=FMT, header=primitives_header)
            np.savetxt('states.dat', states, fmt=FMT, header=states_header)

            # create output folder and move simulation result files
            dir_name = f'mach_{mach}_alpha_{alpha}'
            output_name = f'{output_path}/{dir_name}'
            os.mkdir(output_name)

            for keyword in ['dat', 'szplt', 'csv']:
                files = glob.glob(f'*.{keyword}')
                for file in files:
                    shutil.move(file, output_name)

################################################################################
# Post-processing
################################################################################

# deallocate memory for CFD solver
solver.Postprocessing()
del solver
