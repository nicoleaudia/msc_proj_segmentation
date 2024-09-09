
# This script will recreate the Receptor Expression MESNA macro for FIJI (ImageJ) found here: https://github.com/engpol/JonesLabFIJIScripts

########## Imports ##########
import os
import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
import scyjava as sj
import imagej
from pathlib import Path
import sys
from natsort import natsorted
from cellpose import denoise
from utils.exp_manipulation_utils import *
from utils.plot_and_debug_utils import *
from utils.stack_manipulation_utils import * 
from utils.imagej_and_basic_utils import *
from utils.phantast_utils import *
from utils.microsam_utils import *
from utils.measurement_utils import *
from utils.ensemble_utils import *

from nma_finetune import BrightFieldsDataset, ConvertToUint8, LabelTransform, DataLoaderWrapper 

########## Workflow ##########

############################## User specifies experiment details here ##############################
mode = "segmentation_and_fluor_analysis" # "segmentation_and_fluor_analysis" or "segmentation_only"
exfolder = "C:/"
experiment_folder_path = "/vol/biomedic3/bglocker/mscproj24/nma23/data/testing_directory/multi_model/toy_fullmesna/" # Path to experiment folder
model_id = "finetuned_vit_l_lm" # Best choices: "finetuned_vit_l", "finetuned_vit_l_lm", "finetuned_vit_b", "finetuned_vit_b_lm", "PHANTAST" 
# Other choices (low performance): "vit_b", "vit_l", "vit_b_lm", "vit_l_lm", "ensemble_1", "ensemble_2", "ensemble_3", "ensemble_4_w_PHANTAST", "ensemble_5_w_PHANTAST", "cellpose3"
usam_lm_algorithm = None # "amg", "ais", or None. Only chose amg or ais if using a micro_sam LM model (vit_b_lm, vit_l_lm, etc)

# If using PHANTAST (alone or in ensemble) in headless mode, define the source and destination directories
phantast_headless_source_dir = '.../Segmentation_Output'
phantast_headless_destination_dir = '.../Segmentation_Output'
basic_headless_source_dir = '.../BaSiC_Image'
basic_headless_destination_dir = '.../BaSiC_Image'
######################################### End of user input #########################################

acceptable_file_types = (".tif", ".tiff", ".TIF", ".TIFF")

# Load checkpoint path if model is a finetuned model
if model_id == "finetuned_vit_b":
    checkpoint_path = '/vol/biomedic3/bglocker/mscproj24/nma23/data/testing_directory/multi_model/finetuning/models_vit_b_patch/checkpoints/vit_b_patch_checkpoint/best.pt' 
elif model_id == "finetuned_vit_b_lm":
    checkpoint_path = '/vol/biomedic3/bglocker/mscproj24/nma23/data/testing_directory/multi_model/finetuning/models_vit_b_lm_patch/checkpoints/vit_b_lm_patch_checkpoint/best.pt'
elif model_id == "finetuned_vit_l":
    checkpoint_path = '/vol/biomedic3/bglocker/mscproj24/nma23/data/testing_directory/multi_model/finetuning/models_vit_l_patch/checkpoints/vit_l_patch_checkpoint/best.pt'
elif model_id == "finetuned_vit_l_lm":
    checkpoint_path = '/vol/biomedic3/bglocker/mscproj24/nma23/data/testing_directory/multi_model/finetuning/models_vit_l_lm_patch/checkpoints/vit_l_lm_patch_checkpoint/best.pt'
else:
    checkpoint_path = None

# Load thresholds if checkpoint path is not None
if checkpoint_path is not None:
    pred_iou_thresh, stability_score_thresh = load_thresholds(model_id)
else:
    pred_iou_thresh = 0.75 
    stability_score_thresh = 0.75 

# Headless and interactive functinons
def is_headless_environment():
    # Check if DISPLAY is not set
    if 'DISPLAY' not in os.environ:
        return True
    # Check if running in IPython or Jupyter
    try:
        if 'get_ipython' in globals():
            return False
    except NameError:
        pass
    # Check if stdin is not a terminal (for SSH and other non-interactive environments)
    if not sys.stdin.isatty():
        return True
    # Additional check for running as a main script
    if __name__ == '__main__':
        return False
    return True

# Necessary for interactive mode
def initialise_ij():
    sj.config.add_option('-Xmx12g')
    plugins_dir = Path('/data2/nma23/fiji-linux64/Fiji.app/plugins')
    sj.config.add_option(f'-Dplugins.dir={str(plugins_dir)}')
    ij_path = Path('/data2/nma23/fiji-linux64/Fiji.app')
    ij = imagej.init(str(ij_path), mode='interactive') # ij can either be an instance of ImageJ or False
    # ij.ui().showUI()
    return ij


##### Do shared steps (per experiment) ######

print(f"##################### EXPERIMENT DETAILS ########################")
print("")
print(f"Model: {model_id} with '{usam_lm_algorithm}' algorithm (amg, ais, or None)")
if checkpoint_path is not None:
    print(f"Model loaded from checkpoint path: {checkpoint_path}")
    print(f"Checkpoint using thresholds: pred_iou_thresh = {pred_iou_thresh} and stability_score_thresh = {stability_score_thresh}")
else:
    print(f"Default {model_id} selected. Using default thresholds pred_iou_thresh = {pred_iou_thresh} and stability_score_thresh = {stability_score_thresh}")
print(f"Experiment folder: {experiment_folder_path}")
print("")
print(f"#################################################################")

# Check if headless - if yes, ensure copy paths are set
if is_headless_environment():
    running_in_headless_mode = True
    ij = False
    print("Running in headless mode (laptop, SSH).")
else:
    running_in_headless_mode = False
    ij = initialise_ij()
    print("Running in interactive mode (will use FIJI GUI). Initialised IJ.")

# Make shared directories / check if they exist
# NOTE: if running in segmentation_only mode, ensure bf_dir is already populated
bf_dir, basic_dir = make_shared_dirs(experiment_folder_path) 

if mode == "segmentation_and_fluor_analysis":

    # Check if bf and basic folders are already populated (to prevent repopulating them downstream)
    bf_dir_populated = is_directory_populated(bf_dir)
    basic_dir_populated = is_directory_populated(basic_dir)

    # Deinterleave stack into 4 channels
    deint_images = deinterleave_cv2(experiment_folder_path, acceptable_file_types)

    # Assign channels to variables
    flochannelbefore = deint_images[0]
    bfchannelbefore = deint_images[1]
    flochannelafter = deint_images[2]
    bfchannelafter = deint_images[3]

    # Save brightfield images in directory for later segmentation
    if not bf_dir_populated:
        save_bf(bfchannelbefore, bf_dir)
    else:
        print("Brightfield folder already exists and is populated, not re-saving.")

    # Interleave fluorescent channels
    interleaved_images = interleave_cv2(channel1=flochannelbefore, channel2=flochannelafter)

    # Perform BaSiC correction; if in headless mode, ensure source and destination directories are specified w/in function
    if not basic_dir_populated: 
        if running_in_headless_mode:
            basic_correction_headless(basic_headless_source_dir, basic_headless_destination_dir, running_in_headless_mode)
        else:
            ij.ui().showUI()
            basic_correction_interactive(interleaved_images, basic_dir, ij, running_in_headless_mode)
    else:
        print("BaSiC folder already exists and is populated, not re-calculating.")


if mode == "segmentation_only":
    # Check if brightfields are saved as a single image stack; if yes, deinterleave them and save to be individual image files
    separate_bf_stack(bf_dir, acceptable_file_types)

###### Do model-specific steps (per model) ######

# Make model-specific directory & subdirectories
model_dir, segmentation_dir, results_dir, multip_dir, thresh_dir, ensemble_dir = make_model_dirs(experiment_folder_path, model_id, usam_lm_algorithm)

# Segment according to model specified, result = normalised_stack
if model_id == "PHANTAST":
    if running_in_headless_mode:
        run_phantast_headless(phantast_headless_source_dir, phantast_headless_destination_dir)
    else:
        run_phantast_interactive(running_in_headless_mode, bf_dir=bf_dir, phantast_dir=segmentation_dir, ij=ij)

    # Normalise PHANTAST images (/255), invert, replace in directory, and return as stack
    normalised_stack = normalise_phantast_save_images_in_dir(segmentation_dir)

# Includes finetuned models
if "vit_" in model_id:
    filenames = natsorted([f for f in os.listdir(bf_dir) if f.endswith(acceptable_file_types)]) # filenames is a list
    batch_size = 5 

    normalised_stack = process_images_in_batches(filenames=filenames, batch_size=batch_size, bf_dir=bf_dir, model_choice=model_id, usam_lm_algorithm=usam_lm_algorithm, checkpoint_path=checkpoint_path, pred_iou_thresh=pred_iou_thresh, stability_score_thresh=stability_score_thresh)
    if mode == "segmentation_and_fluor_analysis": # Necessary to save indv images for duplication
        save_indv_images(normalised_stack, segmentation_dir, "image")

if model_id == "cellpose3":  
    # model_type="cyto3" or "nuclei", or other model
    # restore_type: "denoise_cyto3", "deblur_cyto3", "upsample_cyto3" (NMA: upsample up to diam=30), "denoise_nuclei", "deblur_nuclei", "upsample_nuclei"
    imgs = read_tiff_images_from_dir(bf_dir)
    print(f"Cellpose3 input images max: {np.max(imgs)} and images min: {np.min(imgs)}")
    channels = [0,0] # Grayscale = 0; first channel is cyto, second is nuclei; if no nuclei set 2nd channel to 0
    diams = 50 # Cell diameter
    cellpose3_model = denoise.CellposeDenoiseModel(gpu=True, model_type="cyto3", restore_type="denoise_cyto3")
    masks, flows, styles, imgs_dn = cellpose3_model.eval(imgs, diameter=diams, channels=channels)
    print(f"Cellpose3 masks max: {np.max(masks)} and masks min: {np.min(masks)}")

    normalised_stack = normalise_ml_stack(masks)
    if mode == "segmentation_and_fluor_analysis": # Necessary to save indv images for duplication
        save_indv_images(normalised_stack, segmentation_dir, "image")

if model_id.startswith("ensemble_"):

    # Determine which combination of models to run
    if model_id == "ensemble_1":
        # 3 highest IoU scores against PHANTAST, indicating they had the most overlap with PHANTAST. Also lowest 3 RMSE/MSE/MAE scores
        models = ['vit_l_lm_ais', 'vit_b_lm_ais', 'vit_l']
        print(f"Segmenting and assessing ensemble_1 : {models}")
    elif model_id == "ensemble_2":
        # 2 non-specialised models plus the highest performing specialised model
        models = ['vit_l', 'vit_b', 'vit_l_lm_ais']
        print(f"Segmenting and assessing ensemble_3 : {models}")
    elif model_id == "ensemble_3":
        # Variety of architectures (exclusing PHANTAST) - cellpose3 vs micro_sam with and without extra decoder
        models = ['cellpose3', 'vit_l_lm_ais', 'vit_b']
        print(f"Segmenting and assessing ensemble_3 : {models}")
    elif model_id == "ensemble_4_w_PHANTAST":
        # Low combination of correlation coefficients (including PHANTAST) -- vit_b_lm_amg consistently scores low (vit_b and vit_b_lm_amg is 2nd lowest combo), and vit_b and PHANTAST is vit_b's next lowest combo  
        models = ['vit_b', 'vit_b_lm_amg', 'PHANTAST']
        print(f"Segmenting and assessing ensemble_4_w_PHANTAST : {models}")
    elif model_id == "ensemble_5_w_PHANTAST":
        # Ignoring the 2 worst performing (the amgs), choosing the 2 middle performers in correlation coeff against PHANTAST to pair with PHANTAST
        models = ['PHANTAST', 'vit_b_lm_ais', 'vit_b']
        print(f"Segmenting and assessing ensemble_5_w_PHANTAST : {models}")
    else:
        raise ValueError(f"Please check which ensemble model is being used, {model_id} is not one of the options.")

    # Run models and ensemble via simple average per pixel -- above 0.5 average pixel value becomes 1
    normalised_stack = ensemble_process(models, running_in_headless_mode, bf_dir, segmentation_dir, ensemble_dir, acceptable_file_types, ij, phantast_headless_source_dir=phantast_headless_source_dir, phantast_headless_destination_dir=phantast_headless_destination_dir)  


###### Do post model steps (per model) ######

if mode == "segmentation_and_fluor_analysis":
    # Ensure length of segmentation stack is equal to length of fluorescent stack
    original_images, copied_images = duplicate_segmentations(segmentation_dir)
    interleaved_stack = interleave_cv2(original_images, copied_images)
    print(f"Dtype of interleaved stack: {interleaved_stack.dtype}")

# Save interleaved segmentations as single stack, Mask_Stack.tiff
if mode == "segmentation_and_fluor_analysis":
    save_images_as_stack(directory=segmentation_dir, input_images=interleaved_stack, filename="Mask_Stack.tiff")
if mode == "segmentation_only":
    normalised_stack = np.array(normalised_stack) 
    save_images_as_stack(directory=segmentation_dir, input_images=normalised_stack, filename="Mask_Stack.tiff")

if mode == "segmentation_and_fluor_analysis":
    # Load and multiply stacks
    flo_stack = tiff.imread(os.path.join(basic_dir, 'Corrected_Flo_Image.tiff'))
    mask_stack = tiff.imread(os.path.join(segmentation_dir, 'Mask_Stack.tiff'))
    result_stack = multiply_image_stacks(flo_stack, mask_stack)
  
    # DEBUGGING: save stack
    stack_output_path = os.path.join(multip_dir, 'Multiplied_Image.tiff') 
    tiff.imwrite(stack_output_path, result_stack)
    print(f"Saving entire multiplied image stack to: {stack_output_path}")

    # Threshold stack and collect measurements (cell area and mean intensity)
    thresholded_stack = threshold_stack(result_stack, thresh_dir) # Thresholded stack is saved in function
    collect_measurements(result_stack, thresholded_stack, results_dir)

if mode == "segmentation_only":
    collect_area_measurement(normalised_stack, results_dir)

