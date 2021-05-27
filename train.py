"""Simple trainer for a pytorch network.

Authors
-------
 * Mateus Riva (mateus.riva@telecom-paris.fr)
"""

import numpy as np
from copy import deepcopy
from time import time
import math

import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# from util import plot_epoch_result

def train_model_base(model, optimizer, loss_functions, data_loaders, device, loss_function_weights=None,
                convergence_factor=1e-6, max_epochs=100, max_from_best=10, plot_from_epoch=-1, plot_to_epoch=-1, n_class=3):
    """Trains a neural network model until specified criteria are met.

    This function is a generic PyTorch NN training loop.

    Parameters
    ----------
    model : `torch.nn.Module`
        Network model to train.
    optimizer : `torch.optim.Optimizer`
        Optimizer to be used for training.
    loss_functions : list of `torch.nn.Loss`
        List of loss function classes to be used for training.
    data_loaders : `dict`
        Dictionary containing the 'train' and 'val' `DataLoader`s, keyed to those names.
    device : `torch.device`
        The device being used (CUDA or CPU).
    loss_function_weights : list of `floats`, optional
        List of weights for each loss function provided in `loss_functions`. Must be of same size.
    convergence_factor : `float`
        When the difference in loss between two epochs is smaller than this number, stop training.
    max_epochs : `int`
        Maximum number of training epochs.
    max_from_best : `int`, optional
        Maximum number of training epochs without beating best validation loss.
    plot_from_epoch : `int`
        If positive, plot epoch results at each epoch starting from `plot_from_epoch`.

    Returns
    -------
    trained_model
        The trained network model.
    """
    if loss_function_weights is not None:   # Checking enough weights for fcts
        assert len(loss_function_weights) == len(loss_functions), "Len of losses ({}) and weights ({}) mismatch".format(len(loss_functions), len(loss_function_weights))
        if sum(loss_function_weights) != 1: # Normalising weights
            max_weight = max(loss_function_weights)
            loss_function_weights = [w/max_weight for w in loss_function_weights]
    else: # If no weights are provided, equal weights
        loss_function_weights = [1/len(loss_functions) for _ in loss_functions]

    best_model_wts = deepcopy(model.state_dict())   # Stores the best model weights
    best_loss = float("inf")                        # Stores the best validation loss (for best weights)
    last_loss = float("inf")                        # Stores the last validation loss (for convergence)
    loss_difference = float("inf")                  # Stores the difference in loss between last two epochs

    # Train metrics dict key:
    # {
    #     epochs_to_best_model: <value>
    #     epochs_to_convergence: <value>
    #     per_epoch : [
    #       accuracy_val: { class: <acc> }
    #       loss_train: [losses]
    #       loss_val: [losses]
    #       time : <val>
    #     ]
    # }
    train_history = {"epochs_to_best_model": -1, "epochs_to_convergence": -1, "per_epoch": []}

    epoch = 0
    away_from_best = 0
    while loss_difference > convergence_factor and epoch < max_epochs and away_from_best <= max_from_best:

        since = time()

        # Dictionary for tracking in-epoch quality metrics, such as loss-per-phase
        epoch_metrics = {}

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:

            if phase == "train":
                model.train()  # Set model to training mode
            elif phase == "val":
                model.eval()   # Set model to evaluate mode

            # Accumulator variables for measuring average epoch loss
            running_losses = np.zeros(len(loss_functions))
            running_loss = 0
            running_samples = 0

            # Accumulator variables for measuring average epoch dice accuracy
            running_acc = np.zeros(n_class)

            items_pbar = tqdm.tqdm(enumerate(data_loaders[phase]), total=len(data_loaders[phase]),
                                   desc="Epoch {} - {} - last loss: {:.6f} ".format(epoch, phase, last_loss),
                                   postfix={"loss":running_loss, "acc": running_acc})
            for item_index, item_pair in items_pbar:

                # Move loaded data to CUDA if available
                inputs = item_pair["image"].to(device, dtype=torch.bfloat16)
                labels = item_pair["labelmap"].to(device).long()

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass: track history only if training, otherwise we're just evaluating metrics
                with torch.set_grad_enabled(phase == "train"):
                    # Perform forward pass
                    outputs = model(inputs.float())

                    # Note: several loss functions expect the "channel" dimension of the target to be suppressed
                    labels = labels.squeeze(1)
                    # Compute loss function
                    losses = [loss_function(outputs, labels) for loss_function in loss_functions]
                    loss = sum(each_loss * each_weight for each_loss, each_weight in zip(losses,loss_function_weights))

                    # Accumulating running losses
                    running_loss += outputs.shape[0]*loss.item()
                    for i, individual_loss in enumerate(losses):
                        running_losses[i] += individual_loss.item()

                    # backwards pass and optimization only in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                    # Compute accuracy
                    predictions = torch.argmax(outputs, dim=1)
                    for _class in range(n_class):
                        running_acc[_class] += predictions.shape[0]*2.0*torch.sum(torch.logical_and(labels == _class, predictions == _class)) / \
                                            (torch.sum(labels == _class) + torch.sum(predictions == _class))

                running_samples += inputs.size(0) # Accumulating running samples

                items_pbar.set_postfix({"loss":running_loss/running_samples, "acc": running_acc/running_samples})

            # Computing average loss of this phase
            phase_loss = running_loss/running_samples
            epoch_metrics["loss_" + phase] = np.array([each_running_loss/running_samples for each_running_loss in running_losses])

            # Computing average acc of this phase
            phase_acc = running_acc / running_samples
            epoch_metrics["acc_" + phase] = phase_acc

            # If we are on evaluation phase, computed loss gives quality of model
            if phase == 'val':
                # Compute loss difference for convergence
                loss_difference = math.fabs(last_loss-phase_loss)
                last_loss = phase_loss
                # deep copy the model if loss improves
                if phase_loss < best_loss:
                    if verbosity > 0:
                        print("New best model")
                    away_from_best = 0
                    best_loss = phase_loss
                    best_model_wts = deepcopy(model.state_dict())
                    train_history["epochs_to_best_model"] = epoch
                else:
                    away_from_best += 1

        # Reporting results
        time_elapsed = time() - since
        epoch_metrics["time"] = time_elapsed
        if verbosity > 0:
            print("Train loss:      {}".format(epoch_metrics["loss_train"]))
            print("Train accuracy: {}".format(epoch_metrics["acc_train"]))
            print("Validation loss: {}".format(epoch_metrics["loss_val"]))
            print("Validation accuracy: {}".format(epoch_metrics["acc_val"]))
            print("Loss difference: {}".format(loss_difference))
            print('Time elapsed: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print("-----------------------------------")
        epoch += 1

        train_history["per_epoch"].append(epoch_metrics)

        # # Visually reporting results
        # if -1 < plot_from_epoch <= epoch < plot_to_epoch:
        #     plot_epoch_result(model, device, data_loaders["test"], epoch, index=[0,1,2], save_fig=True, extra_title="{} - {}".format(loss_functions[1].get_name(), loss_function_weights[1]))
    # print('loss_difference |', loss_difference, '\n', 'convergence_factor |', convergence_factor, '\n', 'epoch |', epoch, '\n', 'max_epochs |', max_epochs, '\n', 'away_from_best |', away_from_best, '\n', 'max_from_best |', max_from_best)
    print('==Finished on epoch {}\n== with a loss diff of {}.\n== Best val loss: {} at epoch {}\n----------'.format(epoch, loss_difference, best_loss, train_history["epochs_to_best_model"]))
    train_history["epochs_to_convergence"] = epoch

    # Load best weights
    model.load_state_dict(best_model_wts)
    return model, train_history

def train_model_extrainput(model, optimizer, loss_functions, data_loaders, device, loss_function_weights=None,
                convergence_factor=1e-6, max_epochs=100, max_from_best=10, plot_from_epoch=-1, plot_to_epoch=-1, n_class=3):

    if loss_function_weights is not None:   # Checking enough weights for fcts
        assert len(loss_function_weights) == len(loss_functions), "Len of losses ({}) and weights ({}) mismatch".format(len(loss_functions), len(loss_function_weights))
        if sum(loss_function_weights) != 1: # Normalising weights
            max_weight = max(loss_function_weights)
            loss_function_weights = [w/max_weight for w in loss_function_weights]
    else: # If no weights are provided, equal weights
        loss_function_weights = [1/len(loss_functions) for _ in loss_functions]

    best_model_wts = deepcopy(model.state_dict())   # Stores the best model weights
    best_loss = float("inf")                        # Stores the best validation loss (for best weights)
    last_loss = float("inf")                        # Stores the last validation loss (for convergence)
    loss_difference = float("inf")                  # Stores the difference in loss between last two epochs

    train_history = {"epochs_to_best_model": -1, "epochs_to_convergence": -1, "per_epoch": []}

    epoch = 0
    away_from_best = 0
    while loss_difference > convergence_factor and epoch < max_epochs and away_from_best <= max_from_best:

        since = time()

        # Dictionary for tracking in-epoch quality metrics, such as loss-per-phase
        epoch_metrics = {}

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:

            if phase == "train":
                model.train()  # Set model to training mode
            elif phase == "val":
                model.eval()   # Set model to evaluate mode

            # Accumulator variables for measuring average epoch loss
            running_losses = np.zeros(len(loss_functions))
            running_loss = 0
            running_samples = 0

            # Accumulator variables for measuring average epoch dice accuracy
            running_acc = np.zeros(n_class)

            items_pbar = tqdm.tqdm(enumerate(data_loaders[phase]), total=len(data_loaders[phase]),
                                   desc="Epoch {} - {} - last loss: {:.6f} ".format(epoch, phase, last_loss),
                                   postfix={"loss":running_loss, "acc": running_acc})
            for item_index, item_pair in items_pbar:

                # Move loaded data to CUDA if available
                image = item_pair["image"]
                ga = item_pair["ga"]
                ga_voxel = torch.ones(image.size()) * ga
                inputs = torch.cat((image,ga_voxel),1).to(device, dtype=torch.bfloat16)
                labels = item_pair["labelmap"].to(device).long()

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass: track history only if training, otherwise we're just evaluating metrics
                with torch.set_grad_enabled(phase == "train"):
                    # Perform forward pass
                    outputs = model(inputs.float())

                    # Note: several loss functions expect the "channel" dimension of the target to be suppressed
                    labels = labels.squeeze(1)
                    # Compute loss function
                    losses = [loss_function(outputs, labels) for loss_function in loss_functions]
                    loss = sum(each_loss * each_weight for each_loss, each_weight in zip(losses,loss_function_weights))

                    # Accumulating running losses
                    running_loss += outputs.shape[0]*loss.item()
                    for i, individual_loss in enumerate(losses):
                        running_losses[i] += individual_loss.item()

                    # backwards pass and optimization only in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                    # Compute accuracy
                    predictions = torch.argmax(outputs, dim=1)
                    for _class in range(n_class):
                        running_acc[_class] += predictions.shape[0]*2.0*torch.sum(torch.logical_and(labels == _class, predictions == _class)) / \
                                            (torch.sum(labels == _class) + torch.sum(predictions == _class))

                running_samples += inputs.size(0) # Accumulating running samples

                items_pbar.set_postfix({"loss":running_loss/running_samples, "acc": running_acc/running_samples})

            # Computing average loss of this phase
            phase_loss = running_loss/running_samples
            epoch_metrics["loss_" + phase] = np.array([each_running_loss/running_samples for each_running_loss in running_losses])

            # Computing average acc of this phase
            phase_acc = running_acc / running_samples
            epoch_metrics["acc_" + phase] = phase_acc

            # If we are on evaluation phase, computed loss gives quality of model
            if phase == 'val':
                # Compute loss difference for convergence
                loss_difference = math.fabs(last_loss-phase_loss)
                last_loss = phase_loss
                # deep copy the model if loss improves
                if phase_loss < best_loss:
                    if verbosity > 0:
                        print("New best model")
                    away_from_best = 0
                    best_loss = phase_loss
                    best_model_wts = deepcopy(model.state_dict())
                    train_history["epochs_to_best_model"] = epoch
                else:
                    away_from_best += 1

        # Reporting results
        time_elapsed = time() - since
        epoch_metrics["time"] = time_elapsed
        if verbosity > 0:
            print("Train loss:      {}".format(epoch_metrics["loss_train"]))
            print("Train accuracy: {}".format(epoch_metrics["acc_train"]))
            print("Validation loss: {}".format(epoch_metrics["loss_val"]))
            print("Validation accuracy: {}".format(epoch_metrics["acc_val"]))
            print("Loss difference: {}".format(loss_difference))
            print('Time elapsed: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print("-----------------------------------")
        epoch += 1

        train_history["per_epoch"].append(epoch_metrics)

        # # Visually reporting results
        # if -1 < plot_from_epoch <= epoch < plot_to_epoch:
        #     plot_epoch_result(model, device, data_loaders["test"], epoch, index=[0,1,2], save_fig=True, extra_title="{} - {}".format(loss_functions[1].get_name(), loss_function_weights[1]))
    # print('loss_difference |', loss_difference, '\n', 'convergence_factor |', convergence_factor, '\n', 'epoch |', epoch, '\n', 'max_epochs |', max_epochs, '\n', 'away_from_best |', away_from_best, '\n', 'max_from_best |', max_from_best)
    print('==Finished on epoch {}\n== with a loss diff of {}.\n== Best val loss: {} at epoch {}\n----------'.format(epoch, loss_difference, best_loss, train_history["epochs_to_best_model"]))
    train_history["epochs_to_convergence"] = epoch

    # Load best weights
    model.load_state_dict(best_model_wts)
    return model, train_history

def train_model_extratask(model, optimizer, loss_functions, data_loaders, device, loss_function_weights=None,
                convergence_factor=1e-6, max_epochs=100, max_from_best=10, plot_from_epoch=-1, plot_to_epoch=-1, n_class=3):

    if loss_function_weights is not None:   # Checking enough weights for fcts
        assert len(loss_function_weights) == len(loss_functions), "Len of losses ({}) and weights ({}) mismatch".format(len(loss_functions), len(loss_function_weights))
        if sum(loss_function_weights) != 1: # Normalising weights
            max_weight = max(loss_function_weights)
            loss_function_weights = [w/max_weight for w in loss_function_weights]
    else: # If no weights are provided, equal weights
        loss_function_weights = [1/len(loss_functions) for _ in loss_functions]

    best_model_wts = deepcopy(model.state_dict())   # Stores the best model weights
    best_loss = float("inf")                        # Stores the best validation loss (for best weights)
    last_loss = float("inf")                        # Stores the last validation loss (for convergence)
    loss_difference = float("inf")                  # Stores the difference in loss between last two epochs

    train_history = {"epochs_to_best_model": -1, "epochs_to_convergence": -1, "per_epoch": []}

    epoch = 0
    away_from_best = 0
    while loss_difference > convergence_factor and epoch < max_epochs and away_from_best <= max_from_best:

        since = time()

        # Dictionary for tracking in-epoch quality metrics, such as loss-per-phase
        epoch_metrics = {}

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:

            if phase == "train":
                model.train()  # Set model to training mode
            elif phase == "val":
                model.eval()   # Set model to evaluate mode

            # Accumulator variables for measuring average epoch loss
            running_losses = np.zeros(len(loss_functions))
            running_loss = 0
            running_samples = 0

            # Accumulator variables for measuring average epoch dice accuracy
            running_acc = np.zeros(n_class)

            items_pbar = tqdm.tqdm(enumerate(data_loaders[phase]), total=len(data_loaders[phase]),
                                   desc="Epoch {} - {} - last loss: {:.6f} ".format(epoch, phase, last_loss),
                                   postfix={"loss":running_loss, "acc": running_acc})
            for item_index, item_pair in items_pbar:

                # Move loaded data to CUDA if available
                inputs = item_pair["image"]
                ga = item_pair["ga"]
                labels = item_pair["labelmap"].to(device).long()

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass: track history only if training, otherwise we're just evaluating metrics
                with torch.set_grad_enabled(phase == "train"):
                    # Perform forward pass
                    outputs = model(inputs.float())

                    # Note: several loss functions expect the "channel" dimension of the target to be suppressed
                    labels = labels.squeeze(1)
                    # Compute loss functions: where the first function is for the image task
                    # and the second function is for the regression task
                    losses = [loss_function[0](outputs[0],labels), loss_function[1](outputs[1],ga)]
                    loss = sum(each_loss * each_weight for each_loss, each_weight in zip(losses,loss_function_weights))

                    # Accumulating running losses
                    running_loss += outputs.shape[0]*loss.item()
                    for i, individual_loss in enumerate(losses):
                        running_losses[i] += individual_loss.item()

                    # backwards pass and optimization only in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                    # Compute accuracy
                    predictions = torch.argmax(outputs, dim=1)
                    for _class in range(n_class):
                        running_acc[_class] += predictions.shape[0]*2.0*torch.sum(torch.logical_and(labels == _class, predictions == _class)) / \
                                            (torch.sum(labels == _class) + torch.sum(predictions == _class))

                running_samples += inputs.size(0) # Accumulating running samples

                items_pbar.set_postfix({"loss":running_loss/running_samples, "acc": running_acc/running_samples})

            # Computing average loss of this phase
            phase_loss = running_loss/running_samples
            epoch_metrics["loss_" + phase] = np.array([each_running_loss/running_samples for each_running_loss in running_losses])

            # Computing average acc of this phase
            phase_acc = running_acc / running_samples
            epoch_metrics["acc_" + phase] = phase_acc

            # If we are on evaluation phase, computed loss gives quality of model
            if phase == 'val':
                # Compute loss difference for convergence
                loss_difference = math.fabs(last_loss-phase_loss)
                last_loss = phase_loss
                # deep copy the model if loss improves
                if phase_loss < best_loss:
                    if verbosity > 0:
                        print("New best model")
                    away_from_best = 0
                    best_loss = phase_loss
                    best_model_wts = deepcopy(model.state_dict())
                    train_history["epochs_to_best_model"] = epoch
                else:
                    away_from_best += 1

        # Reporting results
        time_elapsed = time() - since
        epoch_metrics["time"] = time_elapsed
        if verbosity > 0:
            print("Train loss:      {}".format(epoch_metrics["loss_train"]))
            print("Train accuracy: {}".format(epoch_metrics["acc_train"]))
            print("Validation loss: {}".format(epoch_metrics["loss_val"]))
            print("Validation accuracy: {}".format(epoch_metrics["acc_val"]))
            print("Loss difference: {}".format(loss_difference))
            print('Time elapsed: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print("-----------------------------------")
        epoch += 1

        train_history["per_epoch"].append(epoch_metrics)

        # # Visually reporting results
        # if -1 < plot_from_epoch <= epoch < plot_to_epoch:
        #     plot_epoch_result(model, device, data_loaders["test"], epoch, index=[0,1,2], save_fig=True, extra_title="{} - {}".format(loss_functions[1].get_name(), loss_function_weights[1]))
    # print('loss_difference |', loss_difference, '\n', 'convergence_factor |', convergence_factor, '\n', 'epoch |', epoch, '\n', 'max_epochs |', max_epochs, '\n', 'away_from_best |', away_from_best, '\n', 'max_from_best |', max_from_best)
    print('==Finished on epoch {}\n== with a loss diff of {}.\n== Best val loss: {} at epoch {}\n----------'.format(epoch, loss_difference, best_loss, train_history["epochs_to_best_model"]))
    train_history["epochs_to_convergence"] = epoch

    # Load best weights
    model.load_state_dict(best_model_wts)
    return model, train_history

def train_model_extraoutput(model, optimizer, loss_functions, data_loaders, device, loss_function_weights=None,
                convergence_factor=1e-6, max_epochs=100, max_from_best=10, plot_from_epoch=-1, plot_to_epoch=-1, n_class=3):

    if loss_function_weights is not None:   # Checking enough weights for fcts
        assert len(loss_function_weights) == len(loss_functions), "Len of losses ({}) and weights ({}) mismatch".format(len(loss_functions), len(loss_function_weights))
        if sum(loss_function_weights) != 1: # Normalising weights
            max_weight = max(loss_function_weights)
            loss_function_weights = [w/max_weight for w in loss_function_weights]
    else: # If no weights are provided, equal weights
        loss_function_weights = [1/len(loss_functions) for _ in loss_functions]

    best_model_wts = deepcopy(model.state_dict())   # Stores the best model weights
    best_loss = float("inf")                        # Stores the best validation loss (for best weights)
    last_loss = float("inf")                        # Stores the last validation loss (for convergence)
    loss_difference = float("inf")                  # Stores the difference in loss between last two epochs

    train_history = {"epochs_to_best_model": -1, "epochs_to_convergence": -1, "per_epoch": []}

    epoch = 0
    away_from_best = 0
    while loss_difference > convergence_factor and epoch < max_epochs and away_from_best <= max_from_best:

        since = time()

        # Dictionary for tracking in-epoch quality metrics, such as loss-per-phase
        epoch_metrics = {}

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:

            if phase == "train":
                model.train()  # Set model to training mode
            elif phase == "val":
                model.eval()   # Set model to evaluate mode

            # Accumulator variables for measuring average epoch loss
            running_losses = np.zeros(len(loss_functions))
            running_loss = 0
            running_samples = 0

            # Accumulator variables for measuring average epoch dice accuracy
            running_acc = np.zeros(n_class)

            items_pbar = tqdm.tqdm(enumerate(data_loaders[phase]), total=len(data_loaders[phase]),
                                   desc="Epoch {} - {} - last loss: {:.6f} ".format(epoch, phase, last_loss),
                                   postfix={"loss":running_loss, "acc": running_acc})
            for item_index, item_pair in items_pbar:

                # Move loaded data to CUDA if available
                inputs = item_pair["image"]
                ga = item_pair["ga"]
                labels = item_pair["labelmap"].to(device).long()
                ga_voxel = torch.ones(labels.size()) * ga
                labels = torch.cat((labels,ga_voxel),1).to(device, dtype=torch.bfloat16)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass: track history only if training, otherwise we're just evaluating metrics
                with torch.set_grad_enabled(phase == "train"):
                    # Perform forward pass
                    outputs = model(inputs.float())

                    # Note: several loss functions expect the "channel" dimension of the target to be suppressed
                    labels = labels.squeeze(1)
                    # Compute loss function: first one is loss on the 8 segmentation channels,
                    # second one is on the last, GA channel
                    losses = [loss_function[0](outputs[:-1], labels[:-1]),
                              loss_function[1](outputs[-1], labels[-1])]
                    loss = sum(each_loss * each_weight for each_loss, each_weight in zip(losses,loss_function_weights))

                    # Accumulating running losses
                    running_loss += outputs.shape[0]*loss.item()
                    for i, individual_loss in enumerate(losses):
                        running_losses[i] += individual_loss.item()

                    # backwards pass and optimization only in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                    # Compute accuracy
                    predictions = torch.argmax(outputs, dim=1)
                    for _class in range(n_class):
                        running_acc[_class] += predictions.shape[0]*2.0*torch.sum(torch.logical_and(labels == _class, predictions == _class)) / \
                                            (torch.sum(labels == _class) + torch.sum(predictions == _class))

                running_samples += inputs.size(0) # Accumulating running samples

                items_pbar.set_postfix({"loss":running_loss/running_samples, "acc": running_acc/running_samples})

            # Computing average loss of this phase
            phase_loss = running_loss/running_samples
            epoch_metrics["loss_" + phase] = np.array([each_running_loss/running_samples for each_running_loss in running_losses])

            # Computing average acc of this phase
            phase_acc = running_acc / running_samples
            epoch_metrics["acc_" + phase] = phase_acc

            # If we are on evaluation phase, computed loss gives quality of model
            if phase == 'val':
                # Compute loss difference for convergence
                loss_difference = math.fabs(last_loss-phase_loss)
                last_loss = phase_loss
                # deep copy the model if loss improves
                if phase_loss < best_loss:
                    if verbosity > 0:
                        print("New best model")
                    away_from_best = 0
                    best_loss = phase_loss
                    best_model_wts = deepcopy(model.state_dict())
                    train_history["epochs_to_best_model"] = epoch
                else:
                    away_from_best += 1

        # Reporting results
        time_elapsed = time() - since
        epoch_metrics["time"] = time_elapsed
        if verbosity > 0:
            print("Train loss:      {}".format(epoch_metrics["loss_train"]))
            print("Train accuracy: {}".format(epoch_metrics["acc_train"]))
            print("Validation loss: {}".format(epoch_metrics["loss_val"]))
            print("Validation accuracy: {}".format(epoch_metrics["acc_val"]))
            print("Loss difference: {}".format(loss_difference))
            print('Time elapsed: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print("-----------------------------------")
        epoch += 1

        train_history["per_epoch"].append(epoch_metrics)

        # # Visually reporting results
        # if -1 < plot_from_epoch <= epoch < plot_to_epoch:
        #     plot_epoch_result(model, device, data_loaders["test"], epoch, index=[0,1,2], save_fig=True, extra_title="{} - {}".format(loss_functions[1].get_name(), loss_function_weights[1]))
    # print('loss_difference |', loss_difference, '\n', 'convergence_factor |', convergence_factor, '\n', 'epoch |', epoch, '\n', 'max_epochs |', max_epochs, '\n', 'away_from_best |', away_from_best, '\n', 'max_from_best |', max_from_best)
    print('==Finished on epoch {}\n== with a loss diff of {}.\n== Best val loss: {} at epoch {}\n----------'.format(epoch, loss_difference, best_loss, train_history["epochs_to_best_model"]))
    train_history["epochs_to_convergence"] = epoch

    # Load best weights
    model.load_state_dict(best_model_wts)
    return model, train_history
