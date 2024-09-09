# python_script.py

# Get the bf_dir from command-line arguments
# bf_dir = sys.argv[1]

# # Ensure that the directory exists
# if not os.path.exists(bf_dir):
#     raise FileNotFoundError(f"Directory  does not exist")

# def my_python_function():
#     print(f"Hello from Python SCRIPT! bf_dir is :)")




# if __name__ == "__main__":
#     my_python_function()

#     from py4j.java_gateway import JavaGateway
#     gateway = JavaGateway()
#     gateway.entry_point.invokePythonFunction()


# import matplotlib.pyplot as plt
# import pandas as pd
# from skimage import exposure, measure, img_as_float
# from basicpy import BaSiC
# from basicpy import datasets as bdata
# import imagej
# import scyjava as sj
# import imagej
# from scyjava import jimport
# from pathlib import Path
# import sys
# from enum import Enum
# import math

# import shutil
# import cv2
#import skimage.segmentation
# import time # for basic testing

# for microsam:
# from glob import glob
# import h5py
# from skimage.measure import label as connected_components
# from torch_em.util.util import get_random_colors
# from torch_em.data.datasets.covid_if import _download_covid_if
# from micro_sam.evaluation.model_comparison import _enhance_image

print ('hello.........')
import os
print ('os')

import tifffile as tiff
print ('tiff')
import numpy as np
print ('np')

from natsort import natsorted
print ('natsorted')

from micro_sam import util
print ('util')

from micro_sam.instance_segmentation import (
    InstanceSegmentationWithDecoder,
    AutomaticMaskGenerator,
    get_predictor_and_decoder,
    mask_data_to_segmentation
)
print ('microsam...')

from utils.microsam_utils import run_automatic_mask_generation

# Not using this function
# def run_automatic_instance_segmentation(image, model_type="vit_b_lm"):
    
#     """Automatic Instance Segmentation by training an additional instance decoder in SAM.

#     NOTE: It is supported only for `µsam` models.
    
#     Args:
#         image: The input image.
#         model_type: The choice of the `µsam` model.
        
#     Returns:
#         The instance segmentation.
#     """
#     # Step 1: Initialize the model attributes using the pretrained µsam model weights.
#     #   - the 'predictor' object for generating predictions using the Segment Anything model.
#     #   - the 'decoder' backbone (for AIS).
#     predictor, decoder = get_predictor_and_decoder(
#         model_type=model_type,  # choice of the Segment Anything model
#         checkpoint_path=None,  # overwrite to pass our own finetuned model
#     )
    
#     # Step 2: Computation of the image embeddings from the vision transformer-based image encoder.
#     image_embeddings = util.precompute_image_embeddings(
#         predictor=predictor,  # the predictor object responsible for generating predictions
#         input_=image,  # the input image
#         ndim=2,  # number of input dimensions
#     )
    
#     # Step 3: Combining the decoder with the Segment Anything backbone for automatic instance segmentation.
#     ais = InstanceSegmentationWithDecoder(predictor, decoder)
    
#     # Step 4: Initializing the precomputed image embeddings to perform faster automatic instance segmentation.
#     ais.initialize(
#         image=image,  # the input image
#         image_embeddings=image_embeddings,  # precomputed image embeddings
#     )

#     # Step 5: Getting automatic instance segmentations for the given image and applying the relevant post-processing steps.
#     prediction = ais.generate()
#     prediction = mask_data_to_segmentation(prediction, with_background=True)
    
#     print("in run_ais fxn")

#     return prediction

# Importing
# def run_automatic_mask_generation(image, model_type="vit_b"):
#     """Automatic Mask Generation.
    
#     NOTE: It is supported for both Segment Anything models and µsam models.
    
#     Args:
#         image: The input image.
#         model_type: The choice of the `SAM` / `µsam` model.
        
#     Returns:
#         The instance segmentation.
#     """
#     # Step 1: Initialize the model attributes using the pretrained SAM / µsam model weights.
#     #   - the 'predictor' object for generating predictions using the Segment Anything model.
#     print("point 0 - in amg")

#     predictor = util.get_sam_model(
#         model_type=model_type,  # choice of the Segment Anything model
#     )
#     print("point 1")
#     # Step 2: Computation of the image embeddings from the vision transformer-based image encoder.
#     image_embeddings = util.precompute_image_embeddings(
#         predictor=predictor,  # the predictor object responsible for generating predictions
#         input_=image,  # the input image
#         ndim=2,  # number of input dimensions
#     )
#     print("point 2")

#     # Step 3: Initializing the predictor for automatic mask generation.
#     amg = AutomaticMaskGenerator(predictor)
    
#     print("point 3")

#     # Step 4: Initializing the precomputed image embeddings to perform automatic segmentation using automatic mask generation.
#     amg.initialize(
#         image=image,  # the input image
#         image_embeddings=image_embeddings,  # precomputed image embeddings
#     )
#     print("point 4")

#     # Step 5: Getting automatic instance segmentations for the given image and applying the relevant post-processing steps.
#     #  - the parameters for `pred_iou_thresh` and `stability_score_thresh` are lowered (w.r.t the defaults) to observe the AMG outputs for the microscopy domain.
#     prediction = amg.generate(
#         pred_iou_thresh=0.75,
#         stability_score_thresh=0.75
#     )

#     print("point 5")

#     prediction = mask_data_to_segmentation(prediction, with_background=True)
    
#     print("point 6")

#     return prediction


def preprocess_image(image):
    """Convert image to float64 and normalise to [0, 255] range."""
    im = image.astype(np.float64)
    im -= im.min()
    im /= (im.max() + 1e-6)  # Avoid division by zero
    im *= 255
    im = np.clip(im, 0, 255)
    return im


def process_images_in_batches(filenames, batch_size, bf_dir, model_choice, usam_lm_algorithm=None):
    segmentations = []
    for batch_start in range(0, len(filenames), batch_size):
        batch_end = min(batch_start + batch_size, len(filenames))
        batch_filenames = filenames[batch_start:batch_end]
        
        for filename in batch_filenames:
            filepath = os.path.join(bf_dir, filename)
            tiff_img = tiff.imread(filepath)
            
            raw = preprocess_image(tiff_img)

            if usam_lm_algorithm == 'amg' or usam_lm_algorithm == None:
                prediction = run_automatic_mask_generation(raw, model_type=model_choice)
            else:
                prediction = run_automatic_instance_segmentation(raw, model_type=model_choice)

            print(f"Processed {filename} via {model_choice}, {usam_lm_algorithm} (amg/ais/None=amg) - Prediction shape: {prediction.shape}")

            print(f'length of segmentation INSIDE loop: {len(segmentations)}')

            segmentations.append(prediction)

            print(f"vit_ Prediction dtype: {prediction.dtype}, bit depth: {np.iinfo(prediction.dtype).bits}")
            print(prediction)

    # Normalise stack
    normalised_segmentations = normalise_ml_stack(segmentations)
    for i, segmentation in enumerate(segmentations):
        print(f"Image {i} dtype: {segmentation.dtype}")
            
    # Return as list
    return normalised_segmentations
        
def save_indv_segmentations(segmentation_stack, directory, base_filename):
    os.makedirs(directory, exist_ok=True)
    for idx, image in enumerate(segmentation_stack):
        filename = os.path.join(directory, f"{base_filename}_{idx:04d}.tiff")
        bit_depth = get_bit_depth_image(image)
        print(f"save_indv_segmentations fxn: Image {filename} has a bit depth of: {bit_depth}")
        tiff.imwrite(filename, image)
        print("")

def normalise_ml_stack(segmentation_stack):
    normalised_stack = []
    for image in segmentation_stack:
        # Normalise and convert the image to binary
        normalised_image = (image / np.max(image) > 0).astype(np.uint8)
        normalised_stack.append(normalised_image)
    return normalised_stack # return in list form

def get_bit_depth_image(image):
    return image.dtype



# Get the bf_dir from command-line arguments
# bf_dir = sys.argv[1]

# Ensure that the directory exists
# if not os.path.exists(bf_dir):
#     raise FileNotFoundError(f"Directory {bf_dir} does not exist")

if __name__ == "__main__":

    acceptable_file_types = (".tif", ".tiff", ".TIF", ".TIFF")
    model_id = "vit_b"
    usam_lm_algorithm = None 
    bf_dir = "/vol/biomedic3/bglocker/mscproj24/nma23/data/testing_directory/sample3/exp1/Brightfield_Stack"
    segmentation_dir = "/vol/biomedic3/bglocker/mscproj24/nma23/data/testing_directory/sample3/exp1/Segmentation_Output"

    print ('in python')
    if model_id.startswith("vit_"):
        # filenames = natsorted([f for f in os.listdir(bf_dir) if f.endswith('.tif') or f.endswith('.tiff')])
        filenames = natsorted([f for f in os.listdir(bf_dir) if f.endswith(acceptable_file_types)])
        batch_size = 1 
        normalised_segmentations = process_images_in_batches(filenames=filenames, batch_size=batch_size, bf_dir=bf_dir, model_choice=model_id, usam_lm_algorithm=usam_lm_algorithm)
        print(len(normalised_segmentations))
        
        save_indv_segmentations(normalised_segmentations, segmentation_dir, "image")
    
