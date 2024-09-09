import os
from natsort import natsorted
import sys
import tifffile as tiff
from glob import glob


def image_files_only_array(arr):
    """
    Filters a list of file paths, returning only those that match supported image file types. e.g. NEWARRAY = ImageFilesOnlyArray(NEWARRAY). Uses a python list comprehension.
    
    Supported file types: '.tif', '.tiff', '.nd2', '.LSM', '.czi', '.jpg'.
    
    Parameters:
    arr (list): List of file paths to filter.
    
    Returns:
    list: A sorted list of file paths that correspond to supported image types.
    """
    supported_filetypes = ['.tif', '.tiff', '.nd2', '.LSM', '.czi', '.jpg']
    files = [file for file in arr if os.path.splitext(file)[1] in supported_filetypes]
    files.natsort()
    return files


def is_headless_environment():
    """
    Determines whether the script is running in a headless environment (remote server / without a display).
    
    Checks for common signs of a headless environment:
    - No 'DISPLAY' environment variable set (common in Linux environments).
    - Running in a non-interactive terminal or SSH.
    
    Returns:
    bool: True if running in a headless environment, False otherwise.
    """
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


def make_shared_dirs(experiment_folder_path):    
    """
    Creates shared directories (Brightfield_Stack and BaSiC_Image) in the specified experiment folder.
    
    Parameters:
    experiment_folder_path (str): Path to the experiment folder where directories should be created.
    
    Returns:
    tuple: A tuple containing the paths to the created directories (Brightfield_Stack, BaSiC_Image).
    """
    bf_dir = os.path.join(experiment_folder_path, "Brightfield_Stack")
    if not os.path.exists(bf_dir):
        os.makedirs(bf_dir)

    basic_dir = os.path.join(experiment_folder_path, "BaSiC_Image")
    if not os.path.exists(basic_dir):
        os.makedirs(basic_dir)

    return bf_dir, basic_dir


def make_model_dirs(experiment_folder_path, model_id, usam_lm_algorithm=None):
    """
    Creates directories for a specific model in the experiment folder. If the model is associated 
    with a micro_sam LM algorithm (amg or ais), it creates subdirectories for segmentation, 
    results, multiplied image, threshold stack, and ensemble intermediates.

    Parameters:
    experiment_folder_path (str): Path to the experiment folder.
    model_id (str): Model identifier (used in the directory name).
    usam_lm_algorithm (str, optional): Micro_sam LM algorithm type ('amg', 'ais', or None).

    Returns:
    tuple: A tuple containing paths to the created directories.
    
    Raises:
    ValueError: If an invalid micro_sam LM algorithm is passed.
    """

    # If running a micro_sam LM model, specify either amg (automatic mask generation) or ais (automatic instance segmentation; uses additional decoder)
    valid_functions = ["amg", "ais", None]
    if usam_lm_algorithm not in valid_functions:
        raise ValueError(f"Invalid micro_sam LM function: {function}. Expected one of: {valid_functions}")
    
    # Tack function on to name
    function_name = f"_{usam_lm_algorithm}" if usam_lm_algorithm in ["amg", "ais"] else ""

    # Create model directory
    model_dir = os.path.join(experiment_folder_path, f"Model_{model_id}{function_name}")
    if os.path.exists(model_dir):
        print(f"Directory {model_dir} already exists. Stopping the script.")
        sys.exit(1)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Create subdirectories
    segmentation_dir = os.path.join(model_dir, "Segmentation_Output")
    if not os.path.exists(segmentation_dir):
        os.makedirs(segmentation_dir)

    results_dir = os.path.join(model_dir, "Results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    multip_dir = os.path.join(model_dir, "Multiplied_Image")
    if not os.path.exists(multip_dir):
        os.makedirs(multip_dir)

    thresh_dir = os.path.join(model_dir, "Threshold_Final_Stack")
    if not os.path.exists(thresh_dir):
        os.makedirs(thresh_dir)

    ensemble_dir = os.path.join(model_dir, "Ensemble_Intermediate_Segmentations")
    if not os.path.exists(ensemble_dir):
        os.makedirs(ensemble_dir)
    
    return model_dir, segmentation_dir, results_dir, multip_dir, thresh_dir, ensemble_dir


def read_tiff_images_from_dir(directory):
    """
    Reads all .tif and .tiff images from the specified directory and loads them into memory.
    
    Parameters:
    directory (str): Path to the directory containing .tif and .tiff images.
    
    Returns:
    list: A list of numpy arrays representing the loaded images.
    """
    image_paths = natsorted(glob(os.path.join(directory, "*.tif"))) + natsorted(glob(os.path.join(directory, "*.tiff")))
    images = []
    
    for image_path in image_paths:
        try:
            image = tiff.imread(image_path)
            images.append(image)
        except Exception as e:
            print(f"Error reading {image_path}: {e}")
    
    return images