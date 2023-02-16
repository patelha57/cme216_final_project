import os
import glob
import shutil
import numpy as np

import mpi
import models

import pysu2

################################################################################
# Pre-processing
################################################################################

FMT = '% .16e'
FMT_HEADER = '%22.22s'

primitive_vars = ['DENSITY', 'VELOCITY-X', 'VELOCITY-Y', 'PRESSURE', 'TEMPERATURE', 'VISCOSITY']
state_vars     = ['DENSITY', 'MOMENTUM-X', 'MOMENTUM-Y', 'ENERGY']
geometry_vars  = ['NORMAL-X', 'NORMAL-Y', 'AREA']

primitives_header = ', '.join((FMT_HEADER % var) for var in primitive_vars)[1:]
states_header     = ', '.join((FMT_HEADER % var) for var in state_vars)[1:]
geometry_header   = ', '.join((FMT_HEADER % var) for var in geometry_vars)[1:]

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
markers = ['AIRFOIL']
nzone = 1

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

mach_data = np.array([0.3, 0.7])   # this is for testing purposes only
alpha_data = np.array([0., 1.])    # this is for testing purposes only

mpi.barrier()

################################################################################
# SU2 execution
################################################################################

# initialize flow solver
config = f'../data/NACA0012/{config}'
solver = pysu2.CSinglezoneDriver(config, nzone, mpi.COMM)
model = models.preprocess_solver(solver, markers)

# set flight conditions
for mach in mach_data:
    for alpha in alpha_data:
        # set angle of attack and Mach number
        solver.SetAngleOfAttack(alpha)
        solver.SetMachNumber(mach)

        if mpi.isroot():
            print(f'Finished setting Mach number = {mach} and angle of attack = {alpha} [deg] for CFD run...')

        mpi.barrier()

        # run flow solver
        solver.ResetConvergence()
        solver.Run()
        solver.Postprocess()
        solver.Monitor(0)
        solver.Output(0)

        # extract flow data on the airfoil surface
        for marker_tag in markers:
            marker = model.grid.marker(marker_tag)

            if marker.index is None:
                primitives_local = ()
                temps_local      = ()
                viscosity_local  = ()
                states_local     = ()
                normals_local    = ()
            else:
                primitives_local = solver.GetMarkerPrimitiveStates(marker.index)
                temps_local      = solver.GetMarkerTemperatures(marker.index)
                viscosity_local  = solver.GetMarkerLaminarViscosities(marker.index)
                states_local     = solver.GetMarkerStates(marker.index)
                normals_local    = solver.GetMarkerVertexNormals(marker.index, False)

            primitives_array = np.asarray(primitives_local, dtype=float).reshape(-1, model.nvar)
            temps_array      = np.asarray(temps_local, dtype=float).reshape(-1, 1)
            viscosity_array  = np.asarray(viscosity_local, dtype=float).reshape(-1, 1)
            states_array     = np.asarray(states_local, dtype=float).reshape(-1, model.nvar)
            normals_array    = np.asarray(normals_local, dtype=float).reshape(-1, model.ndim)

            if mpi.isparallel():
                primitives = marker.partitions.gather(primitives_array)
                temps      = marker.partitions.gather(temps_array)
                viscosity  = marker.partitions.gather(viscosity_array)
                states     = marker.partitions.gather(states_array)
                normals    = marker.partitions.gather(normals_array)

            else:
                primitives = primitives_array
                temps      = temps_array
                viscosity  = viscosity_array
                states     = states_array
                normals    = normals_array

            areas = np.sqrt(np.sum(normals**2, axis=1))

        # extract aerodynamic drag coefficient
        Cd = solver.GetDrag(coefficient=True)

        if mpi.isroot():
            print(f'Aerodynamic drag coefficient = {Cd}')

            # save simulation results to file
            np.savetxt('primitives.dat', np.column_stack((primitives, temps, viscosity)), fmt=FMT, header=primitives_header)
            np.savetxt('states.dat', states, fmt=FMT, header=states_header)
            np.savetxt('geometry.dat', np.column_stack((normals, areas)), fmt=FMT, header=geometry_header)

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
