"""Reads FeTA data and organizes it for use.
"""
import os
import csv

import numpy as np
import torch
import nibabel as nib

# Global variable for data path
data_path_3d = "/home/mriva/Recherche/feta_2.0"

class FetaDataset2D(torch.utils.data.Dataset):
    def __init__(self, data_path, transform = None, target_transform = None):
        self.data_path = data_path

        # Transforms to be applied to the data and to the labels
        self.transform = transform
        self.target_transform = target_transform
        # we have 80*16 subjects and 7 classes+bg
        self.size = 1280
        self.num_classes = 8
        # Storing gestational ages
        with open(os.path.join(self.data_path,"participants.tsv")) as f:
            read_tsv = csv.reader(f, delimiter="\t")
            # Skip header
            next(read_tsv)
            gas = [float(line[2]) for line in read_tsv]
        self.gestational_ages = gas

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        """Loads an item from the dataset and corresponding labelmap and GA"""
        # Computing item ID from index
        index = index % self.size
        subindex = index % 16
        index = index // 16

        # Loading slices
        image = np.load(os.path.join(self.data_path, "images","{:0>3}_{:0>4}.npy".format(index+1, subindex)))
        labelmap = np.load(os.path.join(self.data_path, "labels","{:0>3}_{:0>4}.npy".format(index+1, subindex)))
        # Applying transforms
        image, labelmap = self.apply_transforms(image, labelmap)

        # unsqueezing image channel
        image = np.expand_dims(image,0)

        return {"image": image, "labelmap": labelmap, "ga":self.gestational_ages[index]}

    def apply_transforms(self, image, labelmap):
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            labelmap = self.target_transform(labelmap)
        return image, labelmap

class FetaDataset3D(torch.utils.data.Dataset):
    def __init__(self, data_path, transform = None, target_transform = None):
        self.data_path = data_path

        # Transforms to be applied to the data and to the labels
        self.transform = transform
        self.target_transform = target_transform
        # we have 80 subjects and 7 classes+bg
        self.size = 80
        self.num_classes = 8
        # Storing gestational ages
        with open(os.path.join(self.data_path,"participants.tsv")) as f:
            read_tsv = csv.reader(f, delimiter="\t")
            # Skip header
            next(read_tsv)
            gas = [float(line[2]) for line in read_tsv]
        self.gestational_ages = gas

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        """Loads an item from the dataset and corresponding labelmap and GA"""
        # Preparing data item path
        item_path = "sub-{:>03}".format(index + 1)

        # Loading niftis
        try:
            image = nib.load(os.path.join(self.data_path, item_path, "anat","sub-{:>03}_rec-mial_T2w.nii.gz".format(index+1))).get_fdata()
            labelmap = nib.load(os.path.join(self.data_path, item_path, "anat","sub-{:>03}_rec-mial_dseg.nii.gz".format(index+1))).get_fdata()
        except :
            image = nib.load(os.path.join(self.data_path, item_path, "anat","sub-{:>03}_rec-irtk_T2w.nii.gz".format(index+1))).get_fdata()
            labelmap = nib.load(os.path.join(self.data_path, item_path, "anat","sub-{:>03}_rec-irtk_dseg.nii.gz".format(index+1))).get_fdata()
        # Applying transforms
        image, labelmap = self.apply_transforms(image, labelmap)

        # unsqueezing image channel
        image = np.expand_dims(image,0)

        return {"image": image, "labelmap": labelmap, "ga":self.gestational_ages[index]}

    def apply_transforms(self, image, labelmap):
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            labelmap = self.target_transform(labelmap)
        return image, labelmap

if __name__ == '__main__':
    #set = FetaDataset(data_path_3d)
    #print(len(set))
    #print(set[0])
    #import matplotlib.pyplot as plt
    #plt.imshow(set[0][0][128],cmap="gray")
    #plt.show()
    #plt.imshow(set[0][1][128])
    #plt.show()
    set = FetaDataset2D("./data")
    print(len(set))
    print(set[0])
    print(set[-1])
    print(set[0]["image"].shape)
    print(set[0]["labelmap"].shape)
    import matplotlib.pyplot as plt
    plt.subplot(121); plt.imshow(set[0]["image"][0],cmap="gray")
    plt.subplot(122); plt.imshow(set[0]["labelmap"])
    plt.show()
    plt.subplot(121); plt.imshow(set[-1]["image"][0],cmap="gray")
    plt.subplot(122); plt.imshow(set[-1]["labelmap"])
    plt.show()
