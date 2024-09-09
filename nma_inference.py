# Imports
import os
from glob import glob
import tempfile
import imageio.v3 as imageio
from matplotlib import pyplot as plt
from micro_sam.evaluation import inference
from micro_sam.evaluation.evaluation import run_dice_evaluation
import numpy as np
import tifffile as tiff
import torch
from nma_finetune import BrightFieldsDataset, ConvertToUint8, LabelTransform, DataLoaderWrapper
from utils.filepaths import grab_paths

# Functions

# Goal: Preprocess the images (to uint32 0-255) and labels (to uint8 0-255 instances) before grid search 
def preprocess_filepaths_for_grid_search(image_paths, label_paths, temp_dir):
    processed_image_paths = []
    processed_label_paths = []

    for img_path, lbl_path in zip(image_paths, label_paths):
        
        image = imageio.imread(img_path).astype(np.float32) 
        label = imageio.imread(lbl_path)

        # Normalise image values to ensure they are in the 0-255 range
        min_val, max_val = image.min(), image.max()
        if max_val > 255 or min_val < 0:
            processed_image = 255 * (image - min_val) / (max_val - min_val)
        else:
            processed_image = image  # Already in the correct range

        # Convert the label to binary mask with uint32 type and scale to 0-255
        processed_label = np.where(label > 0, 255, 0).astype(np.uint32)

        # Define new paths in the temporary directory with distinct filenames
        # NOTE: Need to append filenames in order to properly save in intended temp directory
        img_filename = os.path.basename(img_path).replace('.tiff', '_img.tiff')
        lbl_filename = os.path.basename(lbl_path).replace('.tiff', '_label.tiff')

        processed_img_path = os.path.join(temp_dir, img_filename)
        processed_lbl_path = os.path.join(temp_dir, lbl_filename)

        # Save the processed image to temp dir as float32 tiff
        imageio.imwrite(processed_img_path, processed_image.astype(np.float32))

        # Save the processed label to temp dir as uint8 tiff
        imageio.imwrite(processed_lbl_path, processed_label.astype(np.uint8))

        # Store the paths
        processed_image_paths.append(processed_img_path)
        processed_label_paths.append(processed_lbl_path)

    return processed_image_paths, processed_label_paths

# Goal: Preprocess the predictions and ground truth images (to uint8 0-1) for evaluation
def preprocess_filepaths_for_eval(prediction_folder, gt_paths, temp_dir, acceptable_file_types):
    # Normalise predictions
    prediction_files = [f for f in os.listdir(prediction_folder) if f.endswith(acceptable_file_types)]
    normalised_prediction_paths = []
    
    for file_name in prediction_files:
        image_path = os.path.join(prediction_folder, file_name)
        image = imageio.imread(image_path)

        # Normalise the image to uint8 0-1
        max_value = np.max(image)
        if max_value > 0:  # Ensure max value is not zero
            normalised_image = (image / max_value > 0).astype(np.uint8)
        else:
            normalised_image = np.zeros_like(image, dtype=np.uint8)  
        

        # Modify the filename to append '_norm_pred'
        norm_file_name = file_name.replace('.tif', '_norm_pred.tif').replace('.tiff', '_norm_pred.tiff')
        normalised_image_path = os.path.join(temp_dir, norm_file_name)
        imageio.imwrite(normalised_image_path, normalised_image)

        # Append the normalised image path to the list
        normalised_prediction_paths.append(normalised_image_path)

    # Normalise ground truth images
    normalised_gt_paths = []
    
    for gt_file in gt_paths:
        gt_img = tiff.imread(gt_file)

        # Normalise the GT image to uint8 0-1
        max_value = np.max(gt_img)
        if max_value > 0:  # Ensure max value is not zero
            normalised_gt_img = (gt_img / max_value > 0).astype(np.uint8)
        else:
            normalised_gt_img = np.zeros_like(gt_img, dtype=np.uint8)
        
        # Modify the filename to append '_norm_gt'
        gt_file_name = os.path.basename(gt_file).replace('.tif', '_norm_gt.tif').replace('.tiff', '_norm_gt.tiff')
        normalised_gt_path = os.path.join(temp_dir, gt_file_name)
        tiff.imwrite(normalised_gt_path, normalised_gt_img)

        # Append the normalised GT path to the list
        normalised_gt_paths.append(normalised_gt_path)

    return normalised_prediction_paths, normalised_gt_paths

def main():

    # Define settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DATA_FOLDER = "/vol/biomedic3/bglocker/mscproj24/nma23/data/testing_directory/multi_model/finetuning"
    os.makedirs(DATA_FOLDER, exist_ok=True)

    train_image_paths, val_image_paths, test_image_paths, train_gt_paths, val_gt_paths, test_gt_paths = grab_paths()

    # # DEBUGGING: Create toy set
    # toy_train_image_paths = train_image_paths[:2]
    # toy_val_image_paths = val_image_paths[:2]
    # toy_test_image_paths = test_image_paths[:2]
    # toy_train_gt_paths = train_gt_paths[:2]
    # toy_val_gt_paths = val_gt_paths[:2]
    # toy_test_gt_paths = test_gt_paths[:2]

    print("########################################## USER INPUT ##########################################")

    checkpoint='/vol/biomedic3/bglocker/mscproj24/nma23/data/testing_directory/multi_model/finetuning/models_vit_b_lm_patch/checkpoints/vit_b_lm_patch_checkpoint/best.pt'
    assert os.path.exists(checkpoint), f"Checkpoint does not exist: {checkpoint}"

    model_type = "vit_b_lm"

    experiment_folder = os.path.join(DATA_FOLDER, "inference", "v5_b_models_redo", model_type)
    os.makedirs(experiment_folder, exist_ok=True) 

    eval_save_path = os.path.join(experiment_folder, "v5_eval", "v5_diceresults_eval.csv")
    eval_save_dir = os.path.dirname(eval_save_path)
    os.makedirs(eval_save_dir, exist_ok=True) 

    acceptable_file_types = (".tif", ".tiff", ".TIF", ".TIFF")

    print(f"Using checkpoint: {checkpoint}")    
    print(f"Saving everything to directory: {experiment_folder}") 
    print(f"Saving evaluation results to: {eval_save_path}")
    print("########################################## USER INPUT ^ ##########################################")


    # Run grid search and inference (via run_amg)

    with tempfile.TemporaryDirectory() as temp_dir:

        # Preprocess validation images and labels, then save to temporary directories
        preprocessed_val_image_paths, preprocessed_val_gt_paths = preprocess_filepaths_for_grid_search(val_image_paths, val_gt_paths, temp_dir)

        # Run grid search (on val set) and inference (on test set)
        prediction_folder = inference.run_amg(
            checkpoint=checkpoint,
            model_type=model_type,
            experiment_folder=experiment_folder,
            val_image_paths=preprocessed_val_image_paths,
            val_gt_paths=preprocessed_val_gt_paths,
            test_image_paths=test_image_paths,
            iou_thresh_values=[0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7, 0.725, 0.75, 0.775, 0.8],
            stability_score_values=[0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7, 0.725, 0.75, 0.775, 0.8]
        )

    # Run evaluation on predictions with best parameters found during grid search

    # Get (non-normalised) prediction paths
    prediction_paths = sorted(glob(os.path.join(prediction_folder, "*")))

    with tempfile.TemporaryDirectory() as temp_dir:
        # Call the function to normalise and save images
        normalised_prediction_paths, normalised_gt_paths = preprocess_filepaths_for_eval(
            prediction_folder=prediction_folder,
            gt_paths=test_gt_paths,
            temp_dir=temp_dir,
            acceptable_file_types=acceptable_file_types
        )

        # Run dice evaluation on the normalised predictions and ground truths
        results = run_dice_evaluation(
            gt_paths=normalised_gt_paths,
            prediction_paths=normalised_prediction_paths,
            save_path=eval_save_path,
        )

    print(f"Dice score for evaluation: {results}")


if __name__ == "__main__":
    main()