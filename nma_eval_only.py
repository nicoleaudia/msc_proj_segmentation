from glob import glob
import os
import tempfile
import numpy as np
import tifffile as tiff
from utils.filepaths import grab_paths
from micro_sam.evaluation.evaluation import run_dice_evaluation
from natsort import natsorted
import matplotlib.pyplot as plt

from nma_inference import preprocess_filepaths_for_eval

def main():
    print("########################################## USER INPUT ##########################################")
    model_type = "finetuned_vit_l_lm"

    # DATA_FOLDER = "/vol/biomedic3/bglocker/mscproj24/nma23/data/testing_directory/multi_model/finetuning"
    # experiment_folder = os.path.join(DATA_FOLDER, "inference", "v4_all_models_full_inference", model_type)
    experiment_folder = "/vol/biomedic3/bglocker/mscproj24/nma23/data/testing_directory/multi_model/eval_test_data/inhomogeneous_light_exps/disk10_finetuned_vit_l_lm"
    os.makedirs(experiment_folder, exist_ok=True) 

    prediction_folder = '/vol/biomedic3/bglocker/mscproj24/nma23/data/testing_directory/multi_model/eval_test_data/inhomogeneous_light_exps/disk10_finetuned_vit_l_lm/Segmentation_Output'
    eval_save_path = os.path.join(experiment_folder, "dice_eval", "eval.csv")
    eval_save_dir = os.path.dirname(eval_save_path)
    os.makedirs(eval_save_dir, exist_ok=True) 

    # train_image_paths, val_image_paths, test_image_paths, train_gt_paths, val_gt_paths, test_gt_paths = grab_paths()
    test_gt_dir = '/vol/biomedic3/bglocker/mscproj24/nma23/data/testing_directory/multi_model/eval_test_data/inhomogeneous_light_exps/inhomogeneous_light_norm_labels'
    test_gt_paths = natsorted(glob(os.path.join(test_gt_dir, "*")))


    acceptable_file_types = (".tif", ".tiff", ".TIF", ".TIFF")

    print(f"Model: {model_type}")
    print(f"Running evaluation only on predictions in: {prediction_folder}")    
    print(f"Saving evaluation results to: {eval_save_path}") 
    # print("Corresponding inference log:'/vol/biomedic3/bglocker/mscproj24/nma23/nma23_code/keeping_slurm_outputs/DONE_VITLLM_inference.mira09.72034.log'")
    print("########################################## USER INPUT ##########################################")


    # Run evaluation on predictions with best parameters found during grid search

    # Get (non-normalised) prediction paths
    prediction_paths = natsorted(glob(os.path.join(prediction_folder, "*")))

    with tempfile.TemporaryDirectory() as temp_dir:

        # Normalise predictions and ground truths and save to temporary directory
        normalised_prediction_paths, normalised_gt_paths = preprocess_filepaths_for_eval(
            prediction_folder=prediction_folder,
            gt_paths=test_gt_paths,
            temp_dir=temp_dir,
            acceptable_file_types=acceptable_file_types
        )

        print(f"Normalised predictions length: {len(normalised_prediction_paths)}")
        print(f"Normalised GT length: {len(normalised_gt_paths)}")
        assert len(normalised_prediction_paths) == len(normalised_gt_paths), "Mismatch between prediction and GT paths"

        # Run dice evaluation on the normalised predictions and ground truths
        results = run_dice_evaluation(
            gt_paths=normalised_gt_paths,
            prediction_paths=normalised_prediction_paths,
            save_path=eval_save_path,
        )

    print(f"Dice score for evaluation: {results}")

if __name__ == "__main__":
    main()

