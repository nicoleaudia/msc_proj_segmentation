# This script was adapted from the micro_sam sam_finetuning.ipynb notebook available here: https://github.com/computational-cell-analytics/micro-sam

# micro_sam: 
# Segment Anything for Microscopy
# Anwai Archit, Sushmita Nair, Nabeel Khalid, Paul Hilt, Vikas Rajashekar, Marei Freitag, Sagnik Gupta, Andreas Dengel, Sheraz Ahmed, Constantin Pape
# bioRxiv 2023.08.21.554208; doi: https://doi.org/10.1101/2023.08.21.554208

# SAM:
# A. Kirillov et al., "Segment Anything," 2023 IEEE/CVF International Conference on Computer Vision (ICCV), Paris, France, 2023, pp. 3992-4003, 
# doi: 10.1109/ICCV51070.2023.00371.

import os
from glob import glob
from IPython.display import FileLink

import os
import numpy as np
import imageio.v3 as imageio
from matplotlib import pyplot as plt
from skimage.measure import label as connected_components

import torch
import pandas as pd
import torch

import torch_em
from torch_em.model import UNETR
from torch_em.util.debug import check_loader
from torch_em.loss import DiceBasedDistanceLoss
from torch_em.util.util import get_random_colors
from torch_em.transform.label import PerObjectDistanceTransform

from micro_sam import util
import micro_sam.training as sam_training
from micro_sam.sample_data import fetch_tracking_example_data, fetch_tracking_segmentation_data
from micro_sam.instance_segmentation import (
    InstanceSegmentationWithDecoder,
    get_predictor_and_decoder,
    mask_data_to_segmentation
)

import tifffile as tiff
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

from utils.stack_manipulation_utils import save_indv_images


DATA_FOLDER = "/vol/biomedic3/bglocker/mscproj24/nma23/data/testing_directory/multi_model/finetuning"
# os.makedirs(DATA_FOLDER, exist_ok=True)

# Get image and label data
image_dir = os.path.join(DATA_FOLDER, "ft_750_images")
label_dir = os.path.join(DATA_FOLDER, "ft_750_labels")

image_paths = sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.tif') or fname.endswith('.tiff')])
label_paths = sorted([os.path.join(label_dir, fname) for fname in os.listdir(label_dir) if fname.endswith('.tif') or fname.endswith('.tiff')])

assert len(image_paths) == len(label_paths), "Number of images and labels does not match"
# print(f"Number of images: {len(image_paths)}")
# print(f"Number of labels: {len(label_paths)}")

# # The script below returns the train or val data loader for finetuning SAM.
# # The data loader must be a torch data loader that returns `x, y` tensors,
# # where `x` is the image data and `y` are the labels.
# # The labels have to be in a label mask instance segmentation format.
# # i.e. a tensor of the same spatial shape as `x`, with each object mask having its own ID.
# # Important: the ID 0 is reseved for background, and the IDs must be consecutive

class ConvertToUint8:
    def __call__(self, img):
        """ Convert a uint16 NumPy array to uint8 """
        img = (img / 65535.0) * 255.0  # Normalize to [0, 255]
        return img.astype(np.uint8)

class LabelTransform:
    def __init__(self, train_instance_segmentation):
        self.train_instance_segmentation = train_instance_segmentation
        
    def __call__(self, labels):
        if self.train_instance_segmentation:
            # Computes the distance transform for objects to jointly perform the additional decoder-based automatic instance segmentation (AIS) and finetune Segment Anything.
            label_transform = PerObjectDistanceTransform(
                distances=True,
                boundary_distances=True,
                directed_distances=False,
                foreground=True,
                instances=True,
                min_size=25
            )
        else:
            # Ensures the individual object instances.to finetune the clasiscal Segment Anything.
            label_transform = torch_em.transform.label.connected_components

        labels = label_transform(labels)
        return labels
    

class BrightFieldsDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, label_paths, do_transform=None, do_target_transform=None, train_instance_segmentation=False, train = False):
        
        self.raw_paths = image_paths
        self.label_paths = label_paths
        self.do_transform = do_transform
        self.do_target_transform = do_target_transform
        self.train = train
        self.train_instance_segmentation = train_instance_segmentation

        self.image_transform = transforms.Compose([
            ConvertToUint8(),               
            transforms.ToPILImage(),        
            transforms.ToTensor()   
        ])

        if train_instance_segmentation:
            self.label_transform = LabelTransform(train_instance_segmentation)
        else:
            self.label_transform = transforms.Compose([    
                transforms.ToTensor()
            ])
    
    def __len__(self):
        return len(self.label_paths)

    def __getitem__(self, idx: int):
        image = tiff.imread(self.raw_paths[idx])
        label = tiff.imread(self.label_paths[idx])

        if self.do_transform:
            image = self.image_transform(image)
            image = image * 255

        if self.train_instance_segmentation:
            label = self.label_transform(label)
        else:
            if self.do_target_transform:
                label = self.label_transform(label)
                label = label * 255  # Necessary because of transforms.ToTensor (input data had max 0-1, so got divided unnecessarily)
                label = label.to(torch.uint8)

        if self.train:
            patch_size = 256
            contains_segmentation = False
            attempts = 0
            max_attempts = 2000
            original_image = image  
            original_label = label  

            while not contains_segmentation and attempts < max_attempts:
                x = np.random.randint(0, original_image.shape[1] - patch_size + 1) # image = image[:, x : x + patch_size, y : y + patch_size]
                y = np.random.randint(0, original_image.shape[2] - patch_size + 1) # label = label[:, x : x + patch_size, y : y + patch_size]
                
                image_patch = original_image[:, x : x + patch_size, y : y + patch_size]
                label_patch = original_label[:, x : x + patch_size, y : y + patch_size]
                
                if label_patch.numel() > 0 and label_patch.max() >= 1:
                    contains_segmentation = True
                attempts += 1

            if not contains_segmentation:
                print(f"Warning: Could not find a patch with segmentation after {max_attempts} attempts.")
                print(f"Image path: {self.raw_paths[idx]}")
                print(f"Label path: {self.label_paths[idx]}")
            else:
                image = image_patch
                label = label_patch
  

        # (1, 1024, 1024), (1, 1024, 1024)
        # option: return {'image': image, 'label': label}
        return image, label
    
class DataLoaderWrapper:
    def __init__(self, dataloader, shuffle=False):
        self.dataloader = dataloader
        self.shuffle = shuffle
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)

    def __getattr__(self, attr):
        """Forward any attribute lookups to the underlying DataLoader"""
        return getattr(self.dataloader, attr)

def main():
    batch_size = 6
    train_instance_segmentation = False # Set false for finetuned models

    # Split the data - 80% dev (training+val), 20% test, then further split as 5% of dev set for validation
    train_raw_paths, test_raw_paths, train_label_paths, test_label_paths = train_test_split(
        image_paths, label_paths, test_size=0.2, random_state=42
    )
    train_raw_paths, val_raw_paths, train_label_paths, val_label_paths = train_test_split(
        train_raw_paths, train_label_paths, test_size=0.05, random_state=42
    )

    train_dataset = BrightFieldsDataset(image_paths=train_raw_paths, label_paths=train_label_paths, do_transform=True, do_target_transform=True, train_instance_segmentation=train_instance_segmentation, train=True)
    val_dataset = BrightFieldsDataset(image_paths=val_raw_paths, label_paths=val_label_paths, do_transform=True, do_target_transform=True, train_instance_segmentation=train_instance_segmentation, train=True)
    test_dataset = BrightFieldsDataset(image_paths=test_raw_paths, label_paths=test_label_paths, do_transform=True, do_target_transform=True, train_instance_segmentation=train_instance_segmentation, train=False)

    # DataLoader creation
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=True)
    
    train_loader = DataLoaderWrapper(train_loader)
    val_loader = DataLoaderWrapper(val_loader)
    test_loader = DataLoaderWrapper(test_loader)


    ################### All hyperparameters for training. ###################

    n_objects_per_batch = 1  # the number of objects per batch that will be sampled
    # device = "cuda" if torch.cuda.is_available() else "cpu" # the device/GPU used for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_epochs = 100  # how long we train (in epochs)

    # The model_type determines which base model is used to initialize the weights that are finetuned.
    # We use vit_b here because it can be trained faster. Note that vit_h usually yields higher quality results.
    model_type = "vit_b"

    # The name of the checkpoint
    checkpoint_name = "1_256_vit_b_patch"

    lr = 1e-4  # the learning rate for the training


    # Save the paths to a CSV file via pandas DataFrame for use inference, etc
    data = {
        "train_raw_paths": train_raw_paths,
        "train_label_paths": train_label_paths,
        "val_raw_paths": val_raw_paths,
        "val_label_paths": val_label_paths,
        "test_raw_paths": test_raw_paths,
        "test_label_paths": test_label_paths,
    }

    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data.items()]))

    csv_path = os.path.join(DATA_FOLDER, "filepaths", model_type, "data_paths.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)

    print("#########################################################################################################")
    print(f"Training {model_type} for {n_epochs} epochs with a batch size of {batch_size}, learning rate of {lr}. saving to {checkpoint_name}")
    print("#########################################################################################################")

    model, optimizer = sam_training.train_sam(
        name=checkpoint_name,
        save_root=os.path.join(DATA_FOLDER, "models_1_256_vit_b_patch"),
        model_type=model_type,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=n_epochs,
        n_objects_per_batch=n_objects_per_batch,
        with_segmentation_decoder=train_instance_segmentation,
        device=device,
        lr=lr # changed learning rate from default 1e-5
    )

    # Let's spot our best checkpoint and download it to get started with the annotation tool
    best_checkpoint = os.path.join(DATA_FOLDER, "models_1_256_vit_b_patch_other_folder", "checkpoints", checkpoint_name, "best.pt")
    os.makedirs(os.path.dirname(best_checkpoint), exist_ok=True)

    torch.save({
                'epoch': n_epochs,
                # 'model_state_dict': model.state_dict(),
                # 'optimizer_state_dict': optimizer.sam.state_dict(),
                'learning_rate': lr,
                }, best_checkpoint)

    # Free up memory
    print(f"Trained {model_type}")
    del model, optimizer
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
  