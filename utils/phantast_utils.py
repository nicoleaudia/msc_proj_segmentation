"""
This script contains helper functions to run PHANTAST segmentation. Credit to the original functionality that these functions build off of is below.

PHANTAST:
Jaccard, N., Griffin, L.D., Keser, A., Macown, R.J., Super, A., Veraitch, F.S. and Szita, N. (2014), Automated method for the rapid and precise estimation of adherent cell culture characteristics from phase contrast microscopy images. Biotechnol. Bioeng., 111: 504-517. https://doi.org/10.1002/bit.25115

github: https://github.com/nicjac/PHANTAST-FIJI
"""

import os
import numpy as np
import scyjava as sj
import tifffile as tiff
import shutil
from natsort import natsorted


# Will be called if script is run in interactive mode
def run_phantast_interactive(running_in_headless_mode, bf_dir, phantast_dir, ij):
    """
    Run the PHANTAST plugin for ImageJ in interactive mode for processing images.

    Args:
        running_in_headless_mode (bool): Whether the script is running in headless mode. 
                                         This function should only run in interactive mode.
        bf_dir (str): Directory containing input images (in .tiff format).
        phantast_dir (str): Directory where the PHANTAST processed output images will be saved.
        ij: ImageJ (Fiji) instance used for running PHANTAST plugin operations.
    
    Returns:
        None. Processes images and saves the results in the specified output directory.
    
    Raises:
        ValueError: If the result image is not created or unsupported bit depth is encountered.
        FileNotFoundError: If the input file is not found.
        Exception: For any errors during image processing or saving.
    
    Note:
        This function should only be called when running in interactive mode.
    """
    if not running_in_headless_mode:
        # If running BaSiC, don't need this. Still check for legacy though
        # # Initialize ImageJ with explicit legacy support
        # fiji_path = '/vol/biomedic3/bglocker/mscproj24/nma23/miniforge3/envs/micro-sam/Fiji.app'
        # ij = imagej.init(fiji_path, mode='interactive')
        # #ij = imagej.init(fiji_path, mode='headless')

        print(f"PHANTAST section: is legacy active: {ij.legacy.isActive()}")

        sigmaint = 4.0
        epsiint = 0.05

        # Prepare the options for PHANTAST
        options = {
            'sigma': sigmaint,
            'epsilon': epsiint,
            'do new': True
        }

        # Iterate over each image file in the directory
        for filename in os.listdir(bf_dir):
            if filename.endswith(".tiff") or filename.endswith(".tif"):
                
                image_path = os.path.join(bf_dir, filename)

                # Print the name of the image being processed
                print(f"PHANTAST section: Processing image: {filename}")

                # Open the image
                dataset = ij.io().open(image_path)
                print(f"PHANTAST section: Opened image type: {type(dataset)}")

                # Convert the image to an ImagePlus
                ConvertService = ij.get('org.scijava.convert.ConvertService')
                ImagePlus = sj.jimport('ij.ImagePlus')
                imp = ConvertService.convert(dataset, ImagePlus)
                print(f"PHANTAST section: Converted image type: {type(imp)}")

                # Convert the image to uint32 using ImageJ
                ImageConverter = sj.jimport('ij.process.ImageConverter')
                ic = ImageConverter(imp)
                ic.convertToGray32()  # Convert to 32-bit

                # Confirm conversion to 32-bit
                converted_bit_depth = imp.getBitDepth()
                print(f"PHANTAST section: Converted bit depth: {converted_bit_depth}")

                print(f"PHANTAST section: Running PHANTAST with options: {options}")
                result = ij.py.run_plugin("PHANTAST", options, ij1_style=True, imp=imp)

                # Check if the result image is created
                result_image = ij.WindowManager.getCurrentImage()
                if result_image is None:
                    print("PHANTAST section: Error: No image produced by the PHANTAST plugin for {filename}.")
                else:
                    output_path = os.path.join(phantast_dir, f"PHANTAST_Image_{filename}")
                    try:
                        # Map bit depth to numpy data type
                        bit_depth_to_dtype = {8: np.uint8, 16: np.uint16, 32: np.float32}
                        
                        # Get the bit depth of the result image
                        bit_depth = result_image.getBitDepth()
                        print(f"PHANTAST section: Result image bit depth: {bit_depth} bits")
                        
                        if bit_depth not in bit_depth_to_dtype:
                            raise ValueError(f"Unsupported bit depth: {bit_depth}")
                        
                        # Convert the ImagePlus to a NumPy array with the appropriate dtype
                        dtype = bit_depth_to_dtype[bit_depth]
                        result_image_data = np.array(result_image.getProcessor().getPixels(), dtype=dtype)
                        result_image_data_reshaped = result_image_data.reshape((result_image.getHeight(), result_image.getWidth()))
                        
                        # Save the PHANTAST result image with its original bit depth
                        tiff.imwrite(output_path, result_image_data_reshaped)
                        print(f"PHANTAST section: Saved the result image to {output_path}")
                        # print(f"PHANTAST section: Final image bit depth after saving: {get_bit_depth_image(result_image_data_reshaped)}")
                        
                    except Exception as e:
                        print(f"PHANTAST section: Error saving the image: {e}")

                    # Close the current image to free up memory
                    result_image.close()

        print(f"PHANTAST section: Processing complete!")

    else:
        print("Error; run_phantast_interactive function should only be called in interactive mode.")


# Will be called if script is run in headless mode; For debugging; Ensure filepaths are correct
def run_phantast_headless(source_dir, destination_dir, running_in_headless_mode):
    """
    Run PHANTAST in headless mode, a workaround that copies (PHANTAST segmentation) files from a source directory to a destination directory.

    Args:
        source_dir (str): The source directory containing images to copy.
        destination_dir (str): The destination directory where files will be copied.
        running_in_headless_mode (bool): Whether the script is running in headless mode.
    
    Returns:
        None. Copies files and directories from source to destination, skipping Mask_Stack.tiff.
    
    Raises:
        FileNotFoundError: If the source directory does not exist.
        Exception: For any errors during file copying.
    
    Note:
        This function should only be called when running in headless mode.
    """
    if running_in_headless_mode:
        # Check if the source directory exists
        if not os.path.exists(source_dir):
            raise FileNotFoundError(f"Source directory '{source_dir}' does not exist.")

        # Create the destination directory if it does not exist
        os.makedirs(destination_dir, exist_ok=True)

        # Iterate over all the files and directories in the source directory
        for item in os.listdir(source_dir):
            print(f"Processing file: {item}")

            # Skip over multi-image mask stack from previous experiment
            if item == "Mask_Stack.tiff":
                print(f"Skipping file: {item}")
                continue

            source_path = os.path.join(source_dir, item)
            destination_path = os.path.join(destination_dir, item)
            
            # Copy files
            if os.path.isdir(source_path):
                shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
            else:
                shutil.copy2(source_path, destination_path)

        print(f"All items from '{source_dir}' have been copied to '{destination_dir}'")
    else:
        print("Error; run_phantast_headless should only be called in headless mode.")

def normalise_image(image):
    """
    Normalise an image by dividing pixel values by 255 and inverting the pixel intensities (inversion necessary with PHANTAST).

    Args:
        image (np.ndarray): The input image to normalize.
    
    Returns:
        np.ndarray: The normalised image, with values scaled between 0 and 1 and inverted.
    
    Note:
        This function assumes that the PHANTAST output has cells as 0 and background as 1. It should only be used for PHANTAST images.
    """
    # Divide by 255 to normalise
    normalised_image = image / 255.0

    # Invert image (PHANTAST seems to make cell = 0, background = 1)
    normalised_image = 1.0 - normalised_image
    normalised_image = normalised_image.astype(np.uint8)

    return normalised_image

def normalise_phantast_save_images_in_dir(directory):
    """
    Normalize and save all images in the given directory after processing with PHANTAST.

    Args:
        directory (str): The directory containing images to normalize.
    
    Returns:
        np.ndarray: A 3D stack of normalized images if successful, or None if no images were processed.
    
    Raises:
        Exception: For any errors in reading or writing image files.
    
    Note:
        This function processes and normalizes all TIFF files in the specified directory and saves them with the same filenames. This function should only be used with PHANTAST images.
    """
    normalised_stack = []
    for filename in natsorted(os.listdir(directory)):
        print(f"Processing PHANTAST file {filename}")
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            filepath = os.path.join(directory, filename)
            print(f"Created PHANTAST filepath {filepath}")
            
            # Read the TIFF image
            try:
                image = tiff.imread(filepath)
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
                continue
            
            # Normalise the image
            normalised_image = normalise_image(image)
            
            # Save the normalised image, replacing the original
            try:
                tiff.imwrite(filepath, normalised_image)
                print(f"Normalised and saved {filename}")
            except Exception as e:
                print(f"Error writing {filepath}: {e}")
                continue
            
            # Append the normalised image to the stack
            normalised_stack.append(normalised_image)
    
    if not normalised_stack:
        print("No images were normalised. Please check the directory and file extensions.")
        return None
    
    # Convert list to numpy array
    normalised_stack = np.stack(normalised_stack, axis=0)
    
    return normalised_stack
