import time
import numpy as np
import torch
import os
from tlmf_network import *
from tlmf_utils import *

from analysis import parser

'''
"Accelerated training of neural networks via multi-fidelity simulations"
    https://github.com/DDMS-ERE-Stanford/Transfer_Learning_on_Multi_Fidelity_Data.git

Sample code to train model on 128x128(hfs) and 64x64(lfs) data. 
Surrogate model will be trained on muitiple fidelity of data.
Final model will be saved.

The original encoder-decoder model was based on that of:
    "Convolutional Dense Encoder-Decoder Networks":
    https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py
    "Deep Autoregressive Neural Networks for High-Dimensional Inverse Problems in Groundwater Contaminant Source Identification"
    https://github.com/cics-nd/cnn-inversion

Stanford University, DDMS group

Dong Hee Song (dhsong@stanford.edu)
Mar31,2021
'''

tic_all = time.time()


'''
#################################################################
#################################################################
Experiment controls
#################################################################
#################################################################
'''
# Data quantity selection
n_hfs = 3   # number of high fidelity RANS data
n_lfs = 3   # number of low fidelity Euler data
n_test = 2  # number of test data

model_file_name = "final_model.pth"

# PHASE1 Experimental controls
reps_phase1 = 2          # repetitions for phase1
n_epochs_phase1 = 50     # epochs in phase1
lr_phase1 = 0.0005       # learning rate in phase1
wd_phase1 = 1e-5         # weight decay in phase1
factor_phase1 = 0.6      # factor for "ReduceLROnPlateau" in phase1
min_lr_phase1 = 1.5e-06  # minimum learning rate in phase1

# PHASE2 Experimental controls
reps_phase2 = 2          # repetitions for phase2
n_epochs_phase2 = 50     # epochs in phase2
lr_phase2 = 0.00005      # learning rate in phase2
wd_phase2 = 1e-5         # weight decay in phase2
factor_phase2 = 0.6      # factor for "ReduceLROnPlateau" in phase2
min_lr_phase2 = 1.5e-06  # minimum learning rate in phase2

# PHASE3 Experimental controls
reps_phase3 = 2          # repetitions for phase3
n_epochs_phase3 = 50     # epochs in phase3
lr_phase3 = 0.00001      # learning rate in phase3
wd_phase3 = 1e-5         # weight decay in phase3
factor_phase3 = 0.6      # factor for "ReduceLROnPlateau" in phase3
min_lr_phase3 = 5e-07    # minimum learning rate in phase3

'''
#################################################################
#################################################################
Device Selection (CUDA)
#################################################################
#################################################################
'''
USE_GPU = True
dtype = torch.float32  # we will be using float throughout this tutorial
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('using device:', device)

'''
#################################################################
#################################################################
Import 128x128 data

Reads the data in the .hdf5 files and adds it to "dat" variable
#################################################################
#################################################################
'''
current_dir = os.getcwd()

data_dir = f"{current_dir}/dataset"
variables = ['Density', 'Momentum_x', 'Momentum_y', 'Energy', 'Pressure', 'Temperature', 'Mach', 'Pressure_Coefficient']

# parse Euler data
eulerDataRootPath = f"{data_dir}/euler_data"
dataFileName = "surface_flow.dat"
eulerData = parser.parse_data(eulerDataRootPath, dataFileName, variables)
eulerData = np.asarray(eulerData)

# parse RANS data
ransDataRootPath = f"{data_dir}/rans_data"
ransData = parser.parse_data(ransDataRootPath, dataFileName, variables)
ransData = np.asarray(ransData)
print(np.shape(eulerData))

dat = {}
for idx, variable in enumerate(variables):
    var_name = variable.lower()

    dat[f'{var_name}_lfs'] = eulerData[:, :, idx]
    dat[f'{var_name}_hfs'] = ransData[:, :, idx]

'''
#################################################################
#################################################################
Phase1 - experiment [LFS]

- Build data-loaders using LFS data
- Build model and modify to build model_phase1
- Train model_phase1 using LFS data
#################################################################
#################################################################
'''

npoints = 128
nvars = len(variables)

# Build LFS data loaders
test_loaders_lfs, dat = loader_test(data=dat, num_test=n_test, Nxy=(npoints, nvars), bs=40, scale='lfs', variables=variables)

train_loaders = loader_train(data=dat, scale='lfs', num_training=n_lfs, Nxy=(npoints, nvars), bs=40, variables=variables, order=0)

#########################################################
#########################################################
# Build model
model_orig = None
model_orig = DenseED(in_channels=1, out_channels=16, blocks=(7, 12, 7), growth_rate=40, drop_rate=0, bn_size=8, 
                     num_init_features=64, outsize_even=False, bottleneck=False).to(device)
model_phase1_orig = DenseED_phase1(model_orig, blocks=(7, 12, 7)).to(device)

##########################################################
##########################################################

model_phase1, rmse_best = model_train(train_loaders=train_loaders, test_loaders=test_loaders_lfs,
                                      reps=reps_phase1, n_epochs=n_epochs_phase1, log_interval=1, 
                                      model_orig=model_phase1_orig, 
                                      lr=lr_phase1, wd=wd_phase1, factor=factor_phase1, min_lr=min_lr_phase1)
print(f"PHASE1 RMSE BEST = {rmse_best}")

'''
#################################################################
#################################################################
Phase2 - experiment [HFS1]

- Build data-loaders using HFS data
- Build modify model_phase1 to build model_phase2
- Train model_phase2 using HFS data
#################################################################
#################################################################
'''
# Build HFS data loaders
test_loaders_hfs, dat = loader_test(data=dat, num_test=n_test, Nxy=(npoints, nvars), bs=40, scale='hfs', variables=variables)

train_loaders = loader_train(data=dat, scale='hfs', num_training=n_hfs, Nxy=(npoints, nvars), bs=40, variables=variables, order=0)

#########################################################
#########################################################
# Build model
model_phase2_orig = DenseED_phase2(model_orig, model_phase1)

# Freeze weights from being updated (whole model)
for param in model_phase2_orig.parameters():
    param.requires_grad = False

# Unfreeze the parts we do want to update
for param in model_phase2_orig.features.decblock2.parameters():
    param.requires_grad = True
for param in model_phase2_orig.features.up2.parameters():
    param.requires_grad = True

#########################################################
#########################################################
# Train
model_phase2, rmse_best = model_train(train_loaders=train_loaders, test_loaders=test_loaders_hfs,
                                      reps=reps_phase2, n_epochs=n_epochs_phase2, log_interval=1, 
                                      model_orig=model_phase2_orig, 
                                      lr=lr_phase2, wd=wd_phase2, factor=factor_phase2, min_lr=min_lr_phase2)
print(f"PHASE2 RMSE BEST = {rmse_best}")

'''
#################################################################
#################################################################
Phase3 - experiment [HFS2]

- Build modify model_phase2 to build model_phase3
- Train model_phase3 using HFS data
#################################################################
#################################################################
'''
# Build model
model_phase3_orig = model_phase2
# Unfreeze all weights
for param in model_phase3_orig.parameters():
    param.requires_grad = True

#########################################################
#########################################################
# Train
model_phase3, rmse_best = model_train(train_loaders=train_loaders, test_loaders=test_loaders_hfs,
                                      reps=reps_phase3, n_epochs=n_epochs_phase3, log_interval=1, 
                                      model_orig=model_phase3_orig, 
                                      lr=lr_phase3, wd=wd_phase3, factor=factor_phase3, min_lr=min_lr_phase3)
print(f"PHASE3 RMSE BEST = {rmse_best}")

# Save model
torch.save(model_phase3.state_dict(), model_file_name)

toc_all = time.time()
print(f'EXPERIMENT COMPLETE, TIME ELAPSED: {toc_all - tic_all} [sec]')
