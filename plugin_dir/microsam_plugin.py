# python_script.py

# Get the bf_dir from command-line arguments
# bf_dir = sys.argv[1]

# # Ensure that the directory exists
# if not os.path.exists(bf_dir):
#     raise FileNotFoundError(f"Directory  does not exist")



def my_python_function():
    print(f"Testing from Python script.")


# if __name__ == "__main__":
    # my_python_function()

    # from py4j.java_gateway import JavaGateway
    # gateway = JavaGateway()
    # gateway.entry_point.invokePythonFunction()



import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
import numpy as np
from natsort import natsorted
from plugin_dir.microsam_plugin_helper import *
print('Imported all functions.')

# For MacOS
device = torch.device("mps")

# # Get the bf_dir from command-line arguments
# # bf_dir = sys.argv[1]

# # Ensure that the directory exists
# # if not os.path.exists(bf_dir):
# #     raise FileNotFoundError(f"Directory {bf_dir} does not exist")

def main():
    print ('Starting micro_sam plugin.')

    ########## USER INPUT ##########
    model_id = "finetuned_vit_l"
    usam_lm_algorithm = None # 'amg', 'ais', or None. Should be None for finetuned models

    # This should be automatic / hardcoded based on git project location

    # bf_dir = "/vol/biomedic3/bglocker/mscproj24/nma23/data/testing_directory/multi_model/gt5samples/Brightfield_Stack"
    # segmentation_dir = os.path.join("/vol/biomedic3/bglocker/mscproj24/nma23/data/testing_directory/multi_model/gt5samples", model_id, "Segmentation_Output")
    # thresholds_dir = '/vol/biomedic3/bglocker/mscproj24/nma23/nma23_code/thresholds'

    bf_dir = '/Users/nicoleaudia/full_workflow/exp1/Brightfield_Stack'
    segmentation_dir = '/Users/nicoleaudia/full_workflow/exp1/Segmentation_Output'
    # Create segmentation_dir if it doesn't exist - eventually these filepaths should be passed in...
    os.makedirs(segmentation_dir, exist_ok=True)
    thresholds_dir = '/Users/nicoleaudia/thresholds'

    

    acceptable_file_types = (".tif", ".tiff", ".TIF", ".TIFF")

    ########## ^USER INPUT ##########

    # Load checkpoint based on model_id
    if model_id == "finetuned_vit_b":
        checkpoint_path = '/vol/biomedic3/bglocker/mscproj24/nma23/data/testing_directory/multi_model/finetuning/models_vit_b_patch/checkpoints/vit_b_patch_checkpoint/best.pt' 
        # checkpoint_path = '/Users/nicoleaudia/checkpoints/finetuned_vit_b/best.pt'
    elif model_id == "finetuned_vit_b_lm":
        checkpoint_path = '/vol/biomedic3/bglocker/mscproj24/nma23/data/testing_directory/multi_model/finetuning/models_vit_b_lm_patch/checkpoints/vit_b_lm_patch_checkpoint/best.pt'
        # checkpoint_path = '/Users/nicoleaudia/checkpoints/finetuned_vit_b_lm/best.pt'
    elif model_id == "finetuned_vit_l":
        # checkpoint_path = '/vol/biomedic3/bglocker/mscproj24/nma23/data/testing_directory/multi_model/finetuning/models_vit_l_patch/checkpoints/vit_l_patch_checkpoint/best.pt'
        # checkpoint_path = '/vol/biomedic3/bglocker/mscproj24/nma23/models_vit_l_patch_other_folder/checkpoints/vit_l_patch_checkpoint/best.pt'
        checkpoint_path = '/Users/nicoleaudia/final_weightsonly.pt'
        # checkpoint_path = '/vol/biomedic3/bglocker/mscproj24/nma23/data/testing_directory/multi_model/finetuning/models_vit_l_patch/checkpoints/vit_l_patch_checkpoint/final_weightsonly.pt'
    elif model_id == "finetuned_vit_l_lm":
        checkpoint_path = '/vol/biomedic3/bglocker/mscproj24/nma23/data/testing_directory/multi_model/finetuning/models_vit_l_lm_patch/checkpoints/vit_l_lm_patch_checkpoint/best.pt'
        # checkpoint_path = '/Users/nicoleaudia/checkpoints/finetuned_vit_l_lm/best.pt'
    else:
        checkpoint_path = None   
        
    # Load thresholds if checkpoint path is not None
    if checkpoint_path is not None:
        pred_iou_thresh, stability_score_thresh = load_thresholds(model_id, thresholds_dir=thresholds_dir)
    else:
        pred_iou_thresh = 0.75 
        stability_score_thresh = 0.75

    # Only need option for finetuned ViT model - PHANTAST will be handled by macro, other models not available
    if "vit_" in model_id:
        filenames = natsorted([f for f in os.listdir(bf_dir) if f.endswith(acceptable_file_types)])
        batch_size = 3
        
        # Segment the images
        normalised_segmentations = process_images_in_batches(filenames=filenames, batch_size=batch_size, bf_dir=bf_dir, model_choice=model_id, usam_lm_algorithm=usam_lm_algorithm, checkpoint_path=checkpoint_path, pred_iou_thresh=pred_iou_thresh, stability_score_thresh=stability_score_thresh)
      
        print(f"Normalised segmentations stack has length: {len(normalised_segmentations)}.")
        normalised_segmentations = np.array(normalised_segmentations) 
    
        # save_images_as_stack(directory=segmentation_dir, input_images=normalised_segmentations, filename="Mask_Stack.tiff")
        save_indv_images(normalised_segmentations, segmentation_dir, "segmentation")

        print(f"Saved individual segmentations to {segmentation_dir}.")

    else:
        # Throw an error
        raise ValueError(f"Model {model_id} not supported.")
    
    print ('Micro_sam plugin closing.')

if __name__ == "__main__":
    main()
