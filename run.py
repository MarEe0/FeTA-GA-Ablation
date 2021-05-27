"""Runs the ablation study for adding Gestational Age to the FeTA challenge.

Authors:
--------
 * Mateus Riva (mateus.riva@telecom-paris.fr)
 """
import random

import numpy as np
import torch
import tqdm
from itertools import product

from data import FetaDataset
from unet import UNet3D, UNet3D_extratask
from train import train_model_base, train_model_extrainput, train_model_extratask, train_model_extraoutput

if __name__ == '__main__':
    # Experimental attributes and parameters:
    # Batch size
    batch_size=1
    # Learning rate
    lr=0.001
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # configuration label
    experimental_configs = ["base", "extra_input","extra_task","extra_output","extra_step"]
    # Number of repetitions per configuration (rep number is used as random seed)
    repetitions=1

    # Iterate over experimental configurations
    for experimental_config, repetition in product(experimental_configs, range(repetitions)):
        print("{}, repetition #{}".format(experimental_config, repetition))

        # Fixing seeds
        seed = repetition
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

        # Obtaining dataset
        dataset = FetaDataset(data_path="/home/mriva/Recherche/feta_2.0")
        train_set, val_set, test_set = torch.utils.data.random_split(dataset, (60, 10, 10))
        data_loaders = {"train": torch.utils.data.DataLoader(train_set,batch_size=batch_size,num_workers=2),
                        "val": torch.utils.data.DataLoader(val_set,batch_size=batch_size,num_workers=2),
                        "test": torch.utils.data.DataLoader(test_set,batch_size=batch_size,num_workers=2)}

        # Setting up model
        if experimental_config == "base": # Traditional, baseline UNet, random init
            model = UNet3D(input_channels=1, output_channels=8)
            model = model.to(device)
        elif experimental_config == "extra_input":
            model = UNet3D(input_channels=2, output_channels=8)
            model = model.to(device)
        elif experimental_config == "extra_task":
            model = UNet3D_extratask(input_channels=1, output_channels=8)
            model = model.to(device)
        elif experimental_config == "extra_output":
            model = UNet3D(input_channels=1, output_channels=9)
            model = model.to(device)
        else:
            continue

        # Setting up optimizer
        optimizer = torch.optim.Adam(model.parameters(),lr=lr)

        # Setting up loss(es)
        if experimental_config == "base":
            loss_functions = [torch.nn.CrossEntropyLoss()]
        elif experimental_config == "extra_input":
            loss_functions = [torch.nn.CrossEntropyLoss()]
        elif experimental_config == "extra_task" or experimental_config == "extra_output":
            loss_functions = [torch.nn.CrossEntropyLoss(), torch.nn.MSELoss()]
        else:
            continue

        # Train network
        if experimental_config == "base":
            model, train_history = train_model_base(model, optimizer, loss_functions, data_loaders, device,n_class=8)
        if experimental_config == "extra_input":
            model, train_history = train_model_extrainput(model, optimizer, loss_functions, data_loaders, device,n_class=8)
        if experimental_config == "extra_task":
            model, train_history = train_model_extratask(model, optimizer, loss_functions, data_loaders, device,n_class=8)
        if experimental_config == "extra_output":
            model, train_history = train_model_extraoutput(model, optimizer, loss_functions, data_loaders, device,n_class=9)
        else:
            continue
