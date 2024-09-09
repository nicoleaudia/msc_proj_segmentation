"""
This script contains utility functions developed to run BaSiC image correction in FIJI/ImageJ via Python, either through interactive mode or headless mode. Credit to the original functionality that these functions build off of is below.

BaSiC correction:
Peng, T., Thorn, K., Schroeder, T. et al. A BaSiC tool for background and shading correction of optical microscopy images. Nat Commun 8, 14836 (2017). https://doi.org/10.1038/ncomms14836

FIJI:
Schindelin, J., Arganda-Carreras, I., Frise, E. et al. Fiji: an open-source platform for biological-image analysis. Nat Methods 9, 676–682 (2012). https://doi.org/10.1038/nmeth.2019

ImageJ:
Schneider, C., Rasband, W. & Eliceiri, K. NIH Image to ImageJ: 25 years of image analysis. Nat Methods 9, 671–675 (2012). https://doi.org/10.1038/nmeth.2089

"""

from pathlib import Path
import scyjava as sj
from scyjava import jimport
import imagej
import tifffile as tiff
import numpy as np
import os
import shutil

def initialise_ij():
    """
    Initializes an instance of ImageJ with specified memory (12GB) and plugin configurations. It is used for running ImageJ in an interactive mode.
    
    Returns:
        ij (imagej.ImageJ): An initialized instance of ImageJ.
    """
    sj.config.add_option('-Xmx12g')
    plugins_dir = Path('/vol/biomedic3/bglocker/mscproj24/nma23/miniforge3/envs/micro-sam/Fiji.app/plugins')
    sj.config.add_option(f'-Dplugins.dir={str(plugins_dir)}')
    ij_path = Path('/vol/biomedic3/bglocker/mscproj24/nma23/miniforge3/envs/micro-sam/Fiji.app')
    ij = imagej.init(str(ij_path), mode='interactive')
    # ij.ui().showUI()
    return ij


# Will run if script is run in interactive mode
def basic_correction_interactive(interleaved_images, basic_dir, ij, running_in_headless_mode):
    """
    Applies BaSiC correction to interleaved images interactively using ImageJ, following the same settings and pattern as the Jones Lab MESNA macro for their workflows..

    This function performs flat-field and shading correction using the BaSiC plugin in 
    interactive mode. Corrected images are saved in the specified directory.

    Args:
        interleaved_images (numpy.ndarray): The input interleaved images as a 3D numpy array.
        basic_dir (str): The directory where the corrected images will be saved.
        ij (imagej.ImageJ): The initialized instance of ImageJ.
        running_in_headless_mode (bool): Flag to check if the function is running in headless mode. 
                                         If True, the function will not execute.

    Raises:
        RuntimeError: If the BaSiC plugin fails to return a valid corrected image stack.
    """
    if not running_in_headless_mode:
        print(f"BaSiC section: is legacy active: {ij.legacy.isActive()}")
        
        image = interleaved_images
        number_of_images = interleaved_images.shape[0]

        if number_of_images > 200:
            subset_image = []
            # Select every 10th image, starting w 1st, of the first HALF (in macro, count is set before interleaving)
            for i in range(0, len(interleaved_images)//2, 10):
                subset_image.append(interleaved_images[i])

            # Convert the subset images to a 3D numpy array
            subset_image = np.stack(subset_image, axis=0)

        # Convert dataset from numpy -> java
        if number_of_images > 200:
            image_iterable = ij.op().transform().flatIterableView(ij.py.to_java(subset_image))
        else:
            image_iterable = ij.op().transform().flatIterableView(ij.py.to_java(image))

        # Show image in imagej since BaSiC plugin cannot be run headless
        ij.ui().show(image_iterable)
        WindowManager = jimport('ij.WindowManager')
        current_image = WindowManager.getCurrentImage()
    

        # Macros and plug-ins get called  like functions 
        # Macro 1 converts virtual stack to real stack and reorder for BaSiC
        macro1 = """
        rename("active")
        run("Duplicate...", "duplicate")
        selectWindow("active")
        run("Close")
        selectWindow("active-1")
        run("Re-order Hyperstack ...", "channels=[Slices (z)] slices=[Channels (c)] frames=[Frames (t)]")
        """

        macro2 = """
        selectWindow("active-1")
        run("Close")
        """

        macro3 = """
        rename("active")
        """
            
        # Run BaSiC plugin
        plugin = 'BaSiC '
        
        args_small = {
            'processing_stack': 'active-1',
            'flat-field': 'None',
            'dark-field': 'None',
            'shading_estimation': '[Estimate shading profiles]',
            'shading_model': '[Estimate flat-field only (ignore dark-field)]',
            'setting_regularisationparametes': 'Manual',
            'temporal_drift': '[Replace with zero]',
            'correction_options': '[Compute shading and correct images]',
            'lambda_flat': 5,
            'lambda_dark': 0.5
        }

        args_big_first = {
            'processing_stack': 'active-1',
            'flat-field': 'None',
            'dark-field': 'None',
            'shading_estimation': '[Estimate shading profiles]',
            'shading_model': '[Estimate flat-field only (ignore dark-field)]',
            'setting_regularisationparametes': 'Manual',
            'temporal_drift': 'Ignore',
            'correction_options': '[Compute shading only]',
            'lambda_flat': 4,
            'lambda_dark': 0.5
        }

        args_big_second = {
            'processing_stack': 'active-1',
            'flat-field': '[Flat-field:active-1]',
            'dark-field': 'None',
            'shading_estimation': '[Skip estimation and use predefined shading profiles]',
            'shading_model': '[Estimate flat-field only (ignore dark-field)]',
            'setting_regularisationparametes': 'Automatic',
            'temporal_drift': '[Replace with zero]',
            'correction_options': '[Compute shading and correct images]',
            'lambda_flat': 0.5,
            'lambda_dark': 0.5
        }

        # Run macro1 regardless of dataset size
        ij.py.run_macro(macro1)

        # Run BaSiC accordingly depending on dataset size
        if number_of_images > 200:
            # Run BaSiC on subset, compute shading profile
            ij.py.run_plugin(plugin, args_big_first)
            print("Completed first round of BaSiC")
        
            # Close subset (but keep flatfield)
            ij.py.run_macro(macro2)
            print("Completed macro2")

            # Convert full dataset from numpy -> java
            image_iterable = ij.op().transform().flatIterableView(ij.py.to_java(image))

            # Open full dataset in ImageJ
            ij.ui().show(image_iterable)
            WindowManager.getImage('active (V)')
            ij.py.run_macro(macro1)
            print("Completed macro1 again")

            # Run BaSiC on full dataset using shading profile calculated above
            ij.py.run_plugin(plugin, args_big_second)
            print("Completed second round of BaSiC")
        else:
            print("Running BaSiC for 200 or fewer images")
            ij.py.run_plugin(plugin, args_small)

        # Retrieve the corrected image stack directly from ImageJ
        corrected_image_stack = WindowManager.getImage('Corrected:active-1')
        if corrected_image_stack is None:
            raise RuntimeError("Image correction failed: BaSiC didn't return a valid image stack")
        print(corrected_image_stack)
        n_slices = corrected_image_stack.getStackSize()
        height = corrected_image_stack.getHeight()
        width = corrected_image_stack.getWidth()

        # Initialize a NumPy array to store the entire stack
        corrected_image_array = np.zeros((n_slices, height, width), dtype=np.uint16)

        # Iterate through each slice and retrieve the pixel data
        for i in range(1, n_slices + 1):
            corrected_image_stack.setSlice(i)
            slice_processor = corrected_image_stack.getProcessor()
            slice_array = np.array(slice_processor.getPixels(), dtype=np.uint16)
            corrected_image_array[i - 1, :, :] = slice_array.reshape(height, width)

        # Save BaSiC-corrected image stack
        output_path = os.path.join(basic_dir, "Corrected_Flo_Image_Minus1.tiff")
        tiff.imwrite(output_path, corrected_image_array)
        print(f"Original corrected image saved at: {output_path}")

        # Post-processing: Add 1 to every pixel, then save stack
        corrected_image_array += 1
        output_path = os.path.join(basic_dir, "Corrected_Flo_Image.tiff")
        tiff.imwrite(output_path, corrected_image_array)
        print(f"Modified corrected image saved at: {output_path}")
    else:
        print(f"Error; basic_correction_interactive function should only be called in interactive mode")

# Will run if script is run in headless mode; Use for debugging; Ensure filepaths are correct
def basic_correction_headless(source_dir, destination_dir, running_in_headless_mode):
    """
    A workaround function that copies files from the source directory to the destination directory in headless mode.

    This function is used to copy BaSiC-corrected images and related files in a headless mode. 
    It copies both files and directories from the source directory to the destination directory, 
    ensuring that all items are transferred. This can be helpful for minimising interactive mode use, as it is slower and less
    streamlined;

    Args:
        source_dir (str): The path to the source directory containing the files to copy.
        destination_dir (str): The path to the destination directory where the files will be copied.
        running_in_headless_mode (bool): Flag indicating if the function is being run in headless mode.

    Raises:
        FileNotFoundError: If the source directory does not exist.
    """
    if running_in_headless_mode:
        # source_dir = '/vol/biomedic3/bglocker/mscproj24/nma23/data/testing_directory/multi_model/exp5/BaSiC_Image'
        # destination_dir = '/vol/biomedic3/bglocker/mscproj24/nma23/data/testing_directory/multi_model/exp5copy/BaSiC_Image'

        if not os.path.exists(source_dir):
            raise FileNotFoundError(f"Source directory '{source_dir}' does not exist.")

        os.makedirs(destination_dir, exist_ok=True)

        # Iterate over all the files and directories in the source directory
        for item in os.listdir(source_dir):
            source_path = os.path.join(source_dir, item)
            destination_path = os.path.join(destination_dir, item)
            
            # Copy files or directories
            if os.path.isdir(source_path):
                shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
            else:
                shutil.copy2(source_path, destination_path)

        print(f"All items from '{source_dir}' have been copied to '{destination_dir}'")
    else:
        print(f"Error; basic_correction_headless function should only be called in headless mode")