import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as T
import torch.nn.functional as F
import copy
import time


'''
Transfer Learning on Multi-Fidelity Data

Reference:
    https://github.com/DDMS-ERE-Stanford/Transfer_Learning_on_Multi_Fidelity_Data.git
    
Dong Hee Song
Mar 31,2021
Sep 29,2021
'''

_DTYPE = torch.float32


def load_gpu_torch():
    """
    Sets torch device to cuda if cuda is avaliable
    """
    USE_GPU = True
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    return device


def loader_test(data, num_test, Nxy, bs, scale, variables, target_variables, phase):
    """
    Builds the data loaders for the test data.

    Normally, a test-train split can be done with one function,
    In this work the procedure is split into 2 differnt functions
    because when there is a large dataset to conserve vRAM ( we
    may need to split the process of building training data loaders)

    Inputs:
        data - data variable (int)
        num_test - number of test data sets (int)
        Nxy - dimension of data ((int,int))
        bs - batch size (int)
        scale - indicator for fidelity scale of data (str)
        variables - SU2 output flow variables
        target_variables - SU2 high-fidelity target flow variables
        phase - transfer learning phase

    Outputs:
        test_loader - data loader for the test data
        data - modified data file with the test data marked as such
    """
    if phase not in ['PHASE1', 'PHASE2']:
        raise ValueError(f"Training phase '{phase}' is invalid. Choose 'PHASE1' or 'PHASE2'.")

    device = load_gpu_torch()

    (npoints, nvars) = Nxy

    # process field variables
    # TODO: deal with RANS target variables for Phase 2
    input_tensors = []
    output_tensors = []
    for variable in variables:
        var_name       = f'{variable.lower()}_{scale}'
        var_name_test  = f'{var_name}_test'
        var_name_train = f'{var_name}_train'

        # randomly shuffle the dataset
        idx = np.array(torch.randperm(len(data[var_name])))
        var_data_idx = data[var_name][idx]

        # split data into test set and training set
        data[var_name_test]  = var_data_idx[:num_test]
        data[var_name_train] = var_data_idx[num_test:]

        # store test data as a PyTorch tensor
        var_data = data[var_name_test]
        var_torch = torch.stack([torch.Tensor(np.reshape(i, (1, npoints, 1))) for i in list(var_data)])

        # randomly shuffle test data
        idx_test = torch.randperm(len(var_data))
        var_test = var_torch[idx_test]

        # set input and output test data
        input_tensors.append(var_test)

        if phase == 'PHASE1':
            output_tensors.append(var_test)
        elif phase == 'PHASE2' and variable in target_variables:
            output_tensors.append(var_test)

    # pack loaders
    x_test = torch.cat([tensor for tensor in input_tensors], dim=3)
    y_test = torch.cat([tensor for tensor in output_tensors], dim=3)

    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=bs)

    return test_loader, data


def loader_train(data, scale, num_training, Nxy, bs, variables, target_variables, phase, order=0):
    """
    Builds the data loaders for the training data.

    Normally, a test-train split can be done with one function,
    In this work the procedure is split into 2 differnt functions
    because when there is a large dataset to conserve vRAM ( we
    may need to split the process of building training data loaders)

    Inputs:
        data - data variable (int)
        scale - indicator for fidelity scale of data (str)
        num_training - number of training data sets (int)
        Nxy - dimension of data ((int,int))
        bs - batch size (int)
        variables - SU2 output flow variables
        target_variables - SU2 high-fidelity target flow variables
        phase - transfer learning phase
        order - reference int marking the part of the data is being turned into the data loader (int)

    Outputs:
        train_loader - data loader with training data
    """
    if phase not in ['PHASE1', 'PHASE2']:
        raise ValueError(f"Training phase '{phase}' is invalid. Choose 'PHASE1' or 'PHASE2'.")

    device = load_gpu_torch()

    # turn into torch tensor of the correct form
    (npoints, nvars) = Nxy
    data_start = order * num_training
    data_end   = (order + 1) * num_training

    # process field variables
    # TODO: deal with RANS target variables for Phase 2
    input_tensors = []
    output_tensors = []
    for variable in variables:
        var_name       = f'{variable.lower()}_{scale}'
        var_name_train = f'{var_name}_train'

        # store training data as a PyTorch tensor
        var_data = data[var_name_train][data_start:data_end]
        var_torch = torch.stack([torch.Tensor(np.reshape(i, (1, npoints, 1))) for i in list(var_data)])

        # randomly shuffle training data
        idx_train = torch.randperm(len(var_data))
        var_train = var_torch[idx_train]

        # set input and output test data
        input_tensors.append(var_train)

        if phase == 'PHASE1':
            output_tensors.append(var_train)
        elif phase == 'PHASE2' and variable in target_variables:
            output_tensors.append(var_train)

    # pack loaders
    x_train = torch.cat([tensor for tensor in input_tensors], dim=3)
    y_train = torch.cat([tensor for tensor in output_tensors], dim=3)

    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=bs)

    return train_loader


def test(epoch, model, test_loader):
    """
    Evaluates test error at each epoch

    Inputs:
        epoch - current epoch (int)
        model - current model being trined
        test_loader - test loader

    Outputs:
        rmse_test - RMSE(Root Mean Square Error) of test
        mae_test - MAE(Mean Absolute Error) of test
    """
    device = load_gpu_torch()
    
    n_out_pixels_test = len(test_loader.dataset) * test_loader.dataset[0][1].numel()
    model.eval()
    loss = 0.  #initial value
    loss_l1 = 0.  #initial value
    for batch_idx, (input, target) in enumerate(test_loader):
        input, target = input.to(device), target.to(device)
        with torch.no_grad():
            output = model(input)
        loss += F.mse_loss(output, target, size_average=False).item()
        loss_l1 += F.l1_loss(output, target, size_average=False).item()

    rmse_test = np.sqrt(loss / n_out_pixels_test)
    mae_test = loss_l1 / n_out_pixels_test
    # r2_score = 1 - loss / y_test_var
    # print('epoch: {}, test rmse_test:  {:.6f}'.format(epoch, rmse_test))
    return rmse_test, mae_test


def model_train(train_loader, test_loader, reps, n_epochs, log_interval, model_orig, lr, wd, factor, min_lr):
    """
    Trains model for repetitions designated by "reps" and
    returns the best model and RMSE obtained by best model

    Inputs:
        train_loaders - train loaders
        test_loaders - test_ loaders
        reps - number of times to repeat training (int)
        n_epochs - number of epochs to train per each rep (int)
        log_interval - interval(epochs) to compute test error
        model_orig - original model
        lr - learning rate
        wd - weight decay
        factor - factor in ReduceLROnPlateau
        min_lr - minimum learning rate in ReduceLROnPlateau

    Outputs:
        model_best - model associated with the lowest test error
        rmse_best - RMSE associated with "model_best"
    """
    device = load_gpu_torch()

    model_best = None
    tic = time.time()
    rmse_best = 10**6  #initial value, just has to be large

    n_out_pixels_train = len(train_loader.dataset) * train_loader.dataset[0][1].numel()
    n_out_pixels_test  = len(test_loader.dataset) * test_loader.dataset[0][1].numel()

    for rep in range(reps):
        rmse_train, rmse_test = [], []
        model = copy.deepcopy(model_orig)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=10,
                                    verbose=True, threshold=0.0001, threshold_mode='rel',
                                    cooldown=0, min_lr=min_lr, eps=1e-08)

        for epoch in range(1, n_epochs + 1):
            model.train()
            mse = 0.
            for batch_idx, (input, target) in enumerate(train_loader):
                input, target = input.to(device), target.to(device)
                model.zero_grad()
                output = model(input)
                loss = F.l1_loss(output, target, size_average=False)
                loss.backward()
                optimizer.step()
                mse += F.mse_loss(output, target, size_average=False).item()

            rmse = np.sqrt(mse / n_out_pixels_train)
            scheduler.step(rmse)

            if epoch % log_interval == 0:
                rmse_train.append(rmse)
                rmse_t, _ = test(epoch, model=model, test_loader=test_loader)
                rmse_test.append(rmse_t)

        tic2 = time.time()
        print(f'Done training {n_epochs} epochs using {tic2 - tic} seconds. Test RMSE = {rmse_t}')

        if np.mean(rmse_test[-10:]) < rmse_best:
            model_best = copy.deepcopy(model)
            rmse_best = np.mean(rmse_test[-10:])

    return model_best, rmse_best
