"""
This script contains functions inspired by or developed to work micro_sam. Credit to the original functionality that these functions build off of is below.

Micro_sam:
Segment Anything for Microscopy
Anwai Archit, Sushmita Nair, Nabeel Khalid, Paul Hilt, Vikas Rajashekar, Marei Freitag, Sagnik Gupta, Andreas Dengel, Sheraz Ahmed, Constantin Pape
bioRxiv 2023.08.21.554208; doi: https://doi.org/10.1101/2023.08.21.554208

github: https://github.com/computational-cell-analytics/micro-sam

SAM:
SAM: A. Kirillov et al., "Segment Anything," 2023 IEEE/CVF International Conference on Computer Vision (ICCV), Paris, France, 2023, pp. 3992-4003, doi: 10.1109/ICCV51070.2023.00371. 

github: https://github.com/facebookresearch/segment-anything

"""

import os
import tifffile as tiff
import numpy as np
import torch
import matplotlib.pyplot as plt
import json
import pickle

from utils.stack_manipulation_utils import normalise_ml_stack

from glob import glob
import h5py
from skimage.measure import label as connected_components
from torch_em.util.util import get_random_colors
from torch_em.data.datasets.covid_if import _download_covid_if
from micro_sam import util
from micro_sam.evaluation.model_comparison import _enhance_image
from micro_sam.instance_segmentation import (
    InstanceSegmentationWithDecoder,
    AutomaticMaskGenerator,
    get_predictor_and_decoder,
    mask_data_to_segmentation
)

from nma_finetune import BrightFieldsDataset, ConvertToUint8, LabelTransform, DataLoaderWrapper 

def preprocess_image(image):
    """
    Convert an image to float64 and normalize it to the range [0, 255].

    Args:
        image (np.ndarray): The input image to be processed.

    Returns:
        np.ndarray: The processed image, normalized to the range [0, 255].
    """
    im = image.astype(np.float64)
    im -= im.min()
    im /= (im.max() + 1e-6)  # Avoid division by zero
    im *= 255
    im = np.clip(im, 0, 255)
    return im

# This function is modified from the original micro_sam code
def run_automatic_instance_segmentation(image, model_type="vit_b_lm"):
    """Automatic Instance Segmentation by training an additional instance decoder in SAM.

    NOTE: It is supported only for `µsam` models.
    
    Args:
        image: The input image.
        model_type: The choice of the `µsam` model.
        
    Returns:
        The instance segmentation.
    """
    # Step 1: Initialize the model attributes using the pretrained µsam model weights.
    #   - the 'predictor' object for generating predictions using the Segment Anything model.
    #   - the 'decoder' backbone (for AIS).
    predictor, decoder = get_predictor_and_decoder(
        model_type=model_type,  # choice of the Segment Anything model
        checkpoint_path=None,  # overwrite to pass our own finetuned model
    )
    
    # Step 2: Computation of the image embeddings from the vision transformer-based image encoder.
    image_embeddings = util.precompute_image_embeddings(
        predictor=predictor,  # the predictor object responsible for generating predictions
        input_=image,  # the input image
        ndim=2,  # number of input dimensions
    )
    
    # Step 3: Combining the decoder with the Segment Anything backbone for automatic instance segmentation.
    ais = InstanceSegmentationWithDecoder(predictor, decoder)
    
    # Step 4: Initializing the precomputed image embeddings to perform faster automatic instance segmentation.
    ais.initialize(
        image=image,  # the input image
        image_embeddings=image_embeddings,  # precomputed image embeddings
    )

    # Step 5: Getting automatic instance segmentations for the given image and applying the relevant post-processing steps.
    prediction = ais.generate()
        
    # NMA: adding this in to account for images where no segmentation can be found
    if len(prediction):
        prediction = mask_data_to_segmentation(prediction, with_background=True)
    else:
        prediction = np.zeros_like(image, dtype=np.uint32)
        print(f"Processed {image} as an array of zeros")

    # prediction = mask_data_to_segmentation(prediction, with_background=True)
    
    return prediction


# This function is modified from the original micro_sam code
def run_automatic_mask_generation(image, model_type="vit_b", checkpoint_path=None, pred_iou_thresh=0.75, stability_score_thresh=0.75):
    """Automatic Mask Generation using a fine-tuned model.
    
    NOTE: It is supported for both Segment Anything models and µsam models.
    
    Args:
        image: The input image.
        model_type: The choice of the `SAM` / `µsam` model.
        checkpoint_path: Path to the fine-tuned model checkpoint.

        NMA NOTE: Added pred_iou_thresh and stability_score_thresh args
        
    Returns:
        The instance segmentation.
    """

    print(f"Running AMG with pred_iou_thresh: {pred_iou_thresh}, stability_score_thresh: {stability_score_thresh} via checkpoint_path: {checkpoint_path}")
    # Step 1: Initialize the model attributes using the pretrained or fine-tuned SAM / µsam model weights.
    predictor = util.get_sam_model(
        model_type=model_type, 
    )
    if hasattr(predictor, 'model'):
        # Load the fine-tuned model weights if a checkpoint path is provided
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            state_dict = checkpoint['model_state']
            # Strip the 'sam.' prefix from each key
            new_state_dict = {k.replace('sam.', ''): v for k, v in state_dict.items()}

            # Load the modified state_dict into the model
            predictor.model.load_state_dict(new_state_dict)
    else:
        print("No internal model found to load state_dict into.")

    # Step 2: Computation of the image embeddings from the vision transformer-based image encoder.
    image_embeddings = util.precompute_image_embeddings(
        predictor=predictor,  
        input_=image,  
        ndim=2,  
    )
    
    # Step 3: Initializing the predictor for automatic mask generation.
    amg = AutomaticMaskGenerator(predictor)
    
    # Step 4: Initializing the precomputed image embeddings to perform automatic segmentation using automatic mask generation.
    amg.initialize(
        image=image,  
        image_embeddings=image_embeddings, 
    )
    
    # NMA NOTE: assuming everything before this works. amg = segmenter in the original fxn

    # Step 5: Getting automatic instance segmentations for the given image and applying the relevant post-processing steps.
    print(f"RUN_AUTOMATIC_MASK_GENERATION: Generating list of instance segmentations (masks) for image with shape: {image.shape}. dtype: {image.dtype}, max: {image.max()}")
    prediction = amg.generate( 
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh 
    )
    
    # NMA NOTE: similiarly converts to np zeros uint32
    # NMA: adding this in to account for images where no segmentation can be found
    if len(prediction):
        prediction = mask_data_to_segmentation(prediction, with_background=False)
    else:
        prediction = np.zeros_like(image, dtype=np.uint32)

    # NMA NOTE: confirmed the segmentations are correct, just in the wrong order for finetuned only how weird
    # # NMA NOTE: Plotting
    # plt.imshow(prediction, cmap='gray') 
    # plt.title("(^Prediction outputted by mask_data_to_segmentation)")
    # plt.show()
    return prediction


def process_images_in_batches(filenames, batch_size, bf_dir, model_choice, usam_lm_algorithm=None, checkpoint_path=None, pred_iou_thresh=0.75, stability_score_thresh=0.75):
    """
    Process images in batches, applying automatic instance segmentation or mask generation. Only mask generation is suppotred for finetuned models.

    Args:
        filenames (list): List of image filenames.
        batch_size (int): Number of images to process per batch.
        bf_dir (str): Directory where the images are stored.
        model_choice (str): The model to use for segmentation.
        usam_lm_algorithm (str, optional): Algorithm to use for segmentation. Defaults to None.
        checkpoint_path (str, optional): Path to fine-tuned model checkpoint. Defaults to None.
        pred_iou_thresh (float, optional): IOU threshold for predictions. Defaults to 0.75.
        stability_score_thresh (float, optional): Stability score threshold. Defaults to 0.75.

    Returns:
        list: List of normalized segmentations for the processed images.
    """
    
    
    segmentations = []
    for batch_start in range(0, len(filenames), batch_size):
        batch_end = min(batch_start + batch_size, len(filenames))
        batch_filenames = filenames[batch_start:batch_end]

        for filename in batch_filenames:
            filepath = os.path.join(bf_dir, filename)
            tiff_img = tiff.imread(filepath)
            
            print(f"Processing image with dtype: {tiff_img.dtype}, max: {tiff_img.max()}")
            raw = preprocess_image(tiff_img)

            if usam_lm_algorithm == 'amg' or usam_lm_algorithm == None:

                # If finetuned model, temporarily rename in order to pass in base model
                if model_choice.startswith("finetuned_"):
                    model_choice = model_choice.replace("finetuned_", "")
                    print(f"Temporarily renaming the finetuned model to {model_choice}")

                # Make prediction
                prediction = run_automatic_mask_generation(raw, model_type=model_choice, checkpoint_path=checkpoint_path, pred_iou_thresh=pred_iou_thresh, stability_score_thresh=stability_score_thresh)

            else:
                # Make prediction
                prediction = run_automatic_instance_segmentation(raw, model_type=model_choice)

            print(f"Processed {filename} via {model_choice}, {usam_lm_algorithm}.\nPrediction shape: {prediction.shape}. dtype: {prediction.dtype}, max: {prediction.max()}")

            print(f'Length of segmentations list: {len(segmentations)}')

            segmentations.append(prediction)
        
    # Normalise stack
    normalised_segmentations = normalise_ml_stack(segmentations)
    print(f"PROCESS_IMAGES: nomrmalised segmentations have dtype: {normalised_segmentations[0].dtype}, max: {normalised_segmentations[0].max()}")

    # Return as list
    return normalised_segmentations


def plot_and_process_images_in_batches(filenames, batch_size, bf_dir, model_choice, usam_lm_algorithm=None, checkpoint_path=None, pred_iou_thresh=0.75, stability_score_thresh=0.75):
    """
    Process images in batches and plot them, applying automatic instance segmentation or mask generation. Only mask generation is suppotred for finetuned models.

    Args:
        filenames (list): List of image filenames.
        batch_size (int): Number of images to process per batch.
        bf_dir (str): Directory where the images are stored.
        model_choice (str): The model to use for segmentation.
        usam_lm_algorithm (str, optional): Algorithm to use for segmentation. Defaults to None.
        checkpoint_path (str, optional): Path to fine-tuned model checkpoint. Defaults to None.
        pred_iou_thresh (float, optional): IOU threshold for predictions. Defaults to 0.75.
        stability_score_thresh (float, optional): Stability score threshold. Defaults to 0.75.

    Returns:
        list: List of normalized segmentations for the processed images.
    """
    
    segmentations = []
    images_batch = []
    predictions_batch = []
    filenames_batch = []
    
    for idx, filename in enumerate(filenames):
        filepath = os.path.join(bf_dir, filename)
        tiff_img = tiff.imread(filepath)
        
        # Preprocess image
        raw = preprocess_image(tiff_img)

        if usam_lm_algorithm == 'amg' or usam_lm_algorithm is None:
            # Handle finetuned model name change
            if model_choice.startswith("finetuned_"):
                model_choice = model_choice.replace("finetuned_", "")
                print(f"PLOT_AND_PROCESS: Temporarily renaming the finetuned model to {model_choice}")

            # Make prediction
            print(f"PLOT_AND_PROCESS: Calling run_automatic_mask_generation using checkpoint: {checkpoint_path}")
            prediction = run_automatic_mask_generation(raw, model_type=model_choice, checkpoint_path=checkpoint_path, pred_iou_thresh=pred_iou_thresh, stability_score_thresh=stability_score_thresh)
        else:
            # Make prediction using a different algorithm
            prediction = run_automatic_instance_segmentation(raw, model_type=model_choice)

        print(f"PLOT_AND_PROCESS: run_automatic_mask_generation returned {filename} with shape: {prediction.shape}. dtype: {prediction.dtype}, max: {prediction.max()}")
        
        # Store the image and its prediction for visualization
        images_batch.append(raw)
        predictions_batch.append(prediction)
        filenames_batch.append(filename)

        segmentations.append(prediction)
        
        # Visualize every 5 images or at the end of the list
        if (idx + 1) % 5 == 0 or (idx + 1) == len(filenames):
            visualize_batch(images_batch, predictions_batch, filenames_batch)
            # Clear the batches after plotting
            images_batch.clear()
            predictions_batch.clear()
            filenames_batch.clear()
        
        print(f'PLOT_AND_PROCESS: Length of segmentations list being appended: {len(segmentations)}')
        print("PLOT_AND_PROCESS: End of inner loop")
        print("")

    
    # Normalize stack
    print("PLOT_AND_PROCESS: Calling normalise_ml_stack")
    normalised_segmentations = normalise_ml_stack(segmentations)

    return normalised_segmentations


def visualize_batch(images, predictions, filenames):
    """
    Visualise a batch of images and their corresponding predictions.

    Args:
        images (list): List of input images.
        predictions (list): List of predicted segmentations.
        filenames (list): List of image filenames.

    Returns:
        None. Displays the images and predictions in a plot.
    """    
    num_images = len(images)
    
    fig, axs = plt.subplots(num_images, 2, figsize=(6, 3 * num_images))
    
    for i in range(num_images):
        axs[i, 0].imshow(images[i], cmap='gray')
        axs[i, 0].set_title(f'Original Image: {filenames[i]}')
        axs[i, 0].axis('off')
        
        axs[i, 1].imshow(predictions[i], cmap='gray')
        axs[i, 1].set_title('Prediction')
        axs[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    

def process_single_patch(raw, model_choice, checkpoint_path=None, pred_iou_thresh=0.75, stability_score_thresh=0.75):
    """
    Load threshold values based on the model type.

    Args:
        model_id (str): The type of model ('finetuned_vit_b', 'finetuned_vit_l_lm', etc).
        thresholds_dir (str, optional): Directory where threshold files are stored. Defaults to 'thresholds'.

    Returns:
        tuple: A tuple containing pred_iou_thresh and stability_score_thresh values.
    """
    
    patch_size = 512 
    x, y = 0, 0  # Location of the patch in the image (this is top-left corner)

    raw = preprocess_image(raw)
    print(f"Average pixel value in the preprocessed image: {raw.mean()}")
    print(f"Max pixel value in the preprocessed image: {raw.max()}")
    print(f"Preprocessed image dtype: {raw.dtype}")


    # Extract a single patch
    image_patch = raw[x:x + patch_size, y:y + patch_size]

    # Ensure the patch size matches the desired size
    if image_patch.shape[0] != patch_size or image_patch.shape[1] != patch_size:
        pad_x = patch_size - image_patch.shape[0]
        pad_y = patch_size - image_patch.shape[1]
        image_patch = np.pad(image_patch, ((0, pad_x), (0, pad_y)), mode='constant', constant_values=0)

    print(f"Patch dtype: {image_patch.dtype}")
    print(f"Max value in the patch: {image_patch.max()}")

    if model_choice.startswith("finetuned_"):
        model_choice = model_choice.replace("finetuned_", "")
        print(f"Temporarily renaming the finetuned model to {model_choice}")
    # Make prediction for the patch
    prediction_patch = run_automatic_mask_generation(image_patch, model_type=model_choice, checkpoint_path=checkpoint_path, pred_iou_thresh=pred_iou_thresh, stability_score_thresh=stability_score_thresh)

    print(f"Prediction patch max: {prediction_patch.max()}, min: {prediction_patch.min()}, shape: {prediction_patch.shape}")

    # Convert prediction_patch to uint32 if necessary
    prediction_patch = prediction_patch.astype(np.uint32)

    return prediction_patch


def load_thresholds(model_id, thresholds_dir='thresholds'):
    """
    Load threshold values based on the model type.

    Args:
        model_type (str): The type of model ('finetuned_vit_b', 'finetuned_vit_l_lm', etc).
        thresholds_dir (str): Directory where threshold files are stored.

    Returns:
        tuple: A tuple containing thresh1 and thresh2 values.
    """

    # Check model is valid
    finetuned_models = ['finetuned_vit_b', 'finetuned_vit_b_lm', 'finetuned_vit_l', 'finetuned_vit_l_lm']
    if model_id not in finetuned_models:
        raise ValueError(f"Error: '{model_id}' is not a valid model type. Please choose from {finetuned_models}.")

    # Get the filename
    threshold_file = os.path.join(thresholds_dir, f"{model_id}_thresholds.json")
    print(f"Looking for threshold file at: {os.path.abspath(threshold_file)}")

    if not os.path.exists(threshold_file):
        raise FileNotFoundError(f"Threshold file for model type '{model_id}' not found at {threshold_file}.")

    # Load the thresholds from the file
    with open(threshold_file, 'r') as f:
        thresholds = json.load(f)

    pred_iou_thresh = thresholds.get('pred_iou_thresh', 0.75)  # Default to 0.75 if not found
    stability_score_thresh = thresholds.get('stability_score_thresh', 0.75)  # Default to 0.75 if not found

    return pred_iou_thresh, stability_score_thresh


def save_filepaths(file_paths, directory='.', filename='file_paths.pkl'):
    """
    Save file paths to a pickle file.

    Args:
        file_paths (list): List of file paths to save.
        directory (str, optional): Directory where the pickle file will be saved. Defaults to current directory.
        filename (str, optional): Name of the pickle file. Defaults to 'file_paths.pkl'.

    Returns:
        None. Saves the file paths to a pickle file.
    """
  
    os.makedirs(directory, exist_ok=True)
    
    full_path = os.path.join(directory, filename)
    
    # Save the file paths to the specified directory as a pickle file
    with open(full_path, 'wb') as file:
        pickle.dump(file_paths, file)
    
    print(f"File paths successfully saved to {full_path}")


def load_filepaths(directory='.', filename='file_paths.pkl'):
    """
    Load file paths from a pickle file.

    Args:
        directory (str, optional): Directory where the pickle file is located. Defaults to current directory.
        filename (str, optional): Name of the pickle file. Defaults to 'file_paths.pkl'.

    Returns:
        list: Loaded file paths from the pickle file.
    """
    full_path = os.path.join(directory, filename)
    
    # Load the file paths from the specified directory
    with open(full_path, 'rb') as file:
        file_paths = pickle.load(file)
    
    print(f"File paths successfully loaded from {full_path}")
    return file_paths