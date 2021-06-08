"""Converts the FeTA dataset to a series of 2D images.

We take the 16 median slices from every image in the hopes that this will capture
similar regions in different brains.

Authors
-------
 * Mateus Riva (mateus.riva@telecom-paris.fr)
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib

import matplotlib.pyplot as plt

image_train_folder = "/home/mriva/Recherche/nnUNet/preprocessed_feta_2.1/nnUNet_raw_data_base/nnUNet_raw_data/Task001_Brain/imagesTr"
label_train_folder = "/home/mriva/Recherche/nnUNet/preprocessed_feta_2.1/nnUNet_raw_data_base/nnUNet_raw_data/Task001_Brain/labelsTr"
image_test_folder = "/home/mriva/Recherche/nnUNet/preprocessed_feta_2.1/nnUNet_raw_data_base/nnUNet_raw_data/Task001_Brain/imagesTs"
label_test_folder = "/home/mriva/Recherche/nnUNet/preprocessed_feta_2.1/nnUNet_raw_data_base/nnUNet_raw_data/Task001_Brain/labelsTs"
output_folder = "./data"

# Reading train images and labels
for i in range(60):
    image = nib.load(os.path.join(image_train_folder,"{:0>3}_0000.nii.gz".format(i+1))).get_fdata()
    label = nib.load(os.path.join(label_train_folder,"{:0>3}.nii.gz".format(i+1))).get_fdata()

    # Rolling over axes so we take coronal view
    image = np.moveaxis(image, [0,1,2], [2,1,0])
    label = np.moveaxis(label, [0,1,2], [2,1,0])

    # Getting middle 16 slices
    image_slices = image[image.shape[0]//2-8:image.shape[0]//2+8]
    label_slices = label[label.shape[0]//2-8:label.shape[0]//2+8]

    # Saving on output
    for index, (image_slice, label_slice) in enumerate(zip(image_slices,label_slices)):
        # Creating folders if needed
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)
        if not os.path.isdir(os.path.join(output_folder,"images")):
            os.mkdir(os.path.join(output_folder,"images"))
        if not os.path.isdir(os.path.join(output_folder,"labels")):
            os.mkdir(os.path.join(output_folder,"labels"))

        # Saving images as numpy 2D arrays
        np.save(os.path.join(output_folder,"images","{:0>3}_{:0>4}".format(i+1, index)),image_slice)
        plt.imsave(os.path.join(output_folder,"images","{:0>3}_{:0>4}".format(i+1, index)),image_slice,cmap="gray")
        np.save(os.path.join(output_folder,"labels","{:0>3}_{:0>4}".format(i+1, index)),label_slice)
        plt.imsave(os.path.join(output_folder,"labels","{:0>3}_{:0>4}".format(i+1, index)),label_slice)

# Doing the same for test images and labels
for i in range(60,80):
    image = nib.load(os.path.join(image_test_folder,"{:0>3}_0000.nii.gz".format(i+1))).get_fdata()
    label = nib.load(os.path.join(label_test_folder,"{:0>3}.nii.gz".format(i+1))).get_fdata()
    
    image = np.moveaxis(image, [0,1,2], [2,1,0])
    label = np.moveaxis(label, [0,1,2], [2,1,0])

    # Getting middle 16 slices
    image_slices = image[image.shape[0]//2-8:image.shape[0]//2+8]
    label_slices = label[label.shape[0]//2-8:label.shape[0]//2+8]

    # Saving on output
    for index, (image_slice, label_slice) in enumerate(zip(image_slices,label_slices)):
        # Creating folders if needed
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)
        if not os.path.isdir(os.path.join(output_folder,"images")):
            os.mkdir(os.path.join(output_folder,"images"))
        if not os.path.isdir(os.path.join(output_folder,"labels")):
            os.mkdir(os.path.join(output_folder,"labels"))

        # Saving images as numpy 2D arrays
        np.save(os.path.join(output_folder,"images","{:0>3}_{:0>4}".format(i+1, index)),image_slice)
        plt.imsave(os.path.join(output_folder,"images","{:0>3}_{:0>4}".format(i+1, index)),image_slice,cmap="gray")
        np.save(os.path.join(output_folder,"labels","{:0>3}_{:0>4}".format(i+1, index)),label_slice)
        plt.imsave(os.path.join(output_folder,"labels","{:0>3}_{:0>4}".format(i+1, index)),label_slice)
