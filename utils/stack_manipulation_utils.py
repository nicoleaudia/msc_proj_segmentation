from natsort import natsorted
import os
import cv2
import numpy as np
import tifffile as tiff
import math


# terminology:
    # channel = list of image stacks
    # img_stack = image stack = 3D np array where d1 -> no. of slices
    # slice = 1 image = 2D np array

def separate_bf_stack(bf_dir, acceptable_file_types):
    """
    Check if the brightfield images (phase contrast for MESNA) are stored as a single stack; if so, deinterleave them into individual files.
    
    Args:
    bf_dir (str): Directory containing brightfield images.
    acceptable_file_types (tuple): Acceptable file types for the images.
    
    Returns:
    None
    """
    # Check if brightfields are saved as a single image stack; if yes, deinterleave them and save to be individual image files
    files = [f for f in os.listdir(bf_dir) if f.endswith(acceptable_file_types)]
    if len(files) != 1:
        print("Found multiple individual Brightfield images in directory, proceeding to segmentation.")
    else:   
        image_path = os.path.join(bf_dir, files[0])
        image = tiff.imread(image_path)
        image_shape = image.shape
        # print(f"image shape: {image_shape}")

        # Confirm it's a stack (first dimension is greater than 1)
        if len(image_shape) > 2 and image_shape[0] > 1:
            print("This is a stack of images. Shape:", image_shape)
            print("Now saving the stack as individual images instead, and deleting the stack.")
            save_bf(image, bf_dir)
            os.remove(os.path.join(bf_dir, image_path)) # Delete stack from directory, because it's saved as indv imges now
            del(image) # Delete stack from memory
        else:
            raise ValueError("Check that your Brightfield data is either individual TIFF files or a stack of TIFF files.")

def deinterleave_cv2(folder_path, acceptable_file_types):
    """
    Deinterleave a stack of images into four separate lists of images.
    
    Args:
    folder_path (str): Directory containing image stacks.
    acceptable_file_types (tuple): Acceptable file types for the images.
    
    Returns:
    list: A list of four lists, each containing images from one channel.
    """
    tiff_files = [f for f in os.listdir(folder_path) if f.endswith(acceptable_file_types)]

    tiff_files = natsorted(tiff_files)

    # Create list of 4 empty lists; _ indicates loop var won't be used in loop body
    deint_images = [[] for _ in range(4)]

    for tiff_file in tiff_files:
        
        print(f'Deinterleave function: Processing: {tiff_file}')
        tiff_path = os.path.join(folder_path, tiff_file)

        # Flag indicates 16/32 bit image will be returned when input has corresponding depth; otherwise convert to 8 bit
        success, images = cv2.imreadmulti(tiff_path, flags=cv2.IMREAD_ANYDEPTH) 
 
        # Check loading was successful
        if not success:
            raise ValueError(f"Error: Failed to load images from {tiff_file}")
        if len(images) != 4:
            raise ValueError(f"Error: Expected 4 images in stack, but got {len(images)}") 

        # Deinterleave the image stack by appending images to corresponding lists
        for i in range(4):
            if isinstance(images[i], np.ndarray):
                deint_images[i].append(images[i])
            else:
                raise TypeError(f"Expected frames to be np.ndarray, but got {type(images[i])}")

    return deint_images

def interleave_cv2(channel1, channel2):
    """
    Interleave two channels of images to form a single stack.
    
    Args:
    channel1 (list): List of images for channel 1.
    channel2 (list): List of images for channel 2.
    
    Returns:
    numpy.ndarray: Interleaved stack of images.
    """
    # Verify lengths are the same
    if len(channel1) != len(channel2):
        raise ValueError("Interleave function: The two channels do not have the same number of images and cannot be interleaved properly") 
    
    print(f"Interleave function: Length of each channel: {len(channel1)}")

    # Create empty list to store final interleaved images
    interleaved_images = []

    # For each pair of images, interleave them
    for img1, img2 in zip(channel1, channel2):
        # Verify image shapes match
        if img1.shape != img2.shape:
            raise ValueError(f"Interleave section: Shapes do not match between img1 and img2")

        # Interleave images
        interleaved_images.append(img1)
        interleaved_images.append(img2)

    # Convert list of interleaved images to numpy array
    interleaved_image_stack = np.stack(interleaved_images, axis=0)

    return interleaved_image_stack

def is_directory_populated(directory, extension=(".tif", ".tiff")):
    """
    Check if a directory contains files with the given extension(s).
    
    Args:
    directory (str): The directory to check.
    extension (tuple): Tuple of file extensions to look for.
    
    Returns:
    bool: True if the directory contains files with the given extension, False otherwise.
    """
    return any(fname.endswith(extension) for fname in os.listdir(directory))


def multiply_image_stacks(stack1, stack2):
    """
    Multiply two 3D image stacks element-wise.
    
    Args:
    stack1 (numpy.ndarray): The first image stack.
    stack2 (numpy.ndarray): The second image stack.
    
    Returns:
    numpy.ndarray: The resulting stack after multiplication.
    """
    # Perform element-wise multiplication
    multiplied_stack = stack1 * stack2
    
    return multiplied_stack


def apply_threshold_clip(image_stack, low=1, high=65535):
    """
    Apply a threshold to an image stack, clipping values outside the specified range.
    
    Args:
    image_stack (numpy.ndarray): The input image stack.
    low (int): The lower threshold value.
    high (int): The upper threshold value.
    
    Returns:
    numpy.ndarray: The thresholded image stack as uint16.
    """
    # Create a copy of the image stack to avoid modifying the original data
    thresholded_stack = image_stack.copy()
    # Set values below/above the thresholds to 0
    thresholded_stack[thresholded_stack < low] = 0
    thresholded_stack[thresholded_stack > high] = 0
   
    return thresholded_stack.astype(np.uint16)


def duplicate_segmentations(segmentation_dir):
    """
    Create copies of segmentation images in a given directory.
    
    Args:
    segmentation_dir (str): The directory containing segmentation images.
    
    Returns:
    tuple: Two lists - the original images and their copies.
    """
    print(f"Processing directory: {segmentation_dir}")

    original_images = []
    copied_images = []

    for filename in natsorted(os.listdir(segmentation_dir)):
        if filename.endswith(".tiff") or filename.endswith(".tif"):
            print(f"duplicate_segmentations fxn: processing file {filename}")
            image_path = os.path.join(segmentation_dir, filename)
            img = tiff.imread(image_path)
            if img is not None:
                original_images.append(img)
                # Create a copy of the image
                copied_images.append(np.copy(img))
                # bit_depth = get_bit_depth_image(img)
                # print(f"duplicate_segmentations fxn: Image {filename} has a bit depth of: {bit_depth}")
            else:
                print(f"Warning: Failed to read image {filename}")

    return original_images, copied_images


def save_images_as_stack(directory, input_images, filename):
    """
    Save a stack of images as a single TIFF file.
    
    Args:
    directory (str): Directory to save the file.
    input_images (numpy.ndarray): Stack of images to save.
    filename (str): The name of the output file.
    
    Returns:
    None
    """
    output_path = os.path.join(directory, filename)
    tiff.imwrite(output_path, input_images)


def threshold_stack(result_stack, thresh_dir):
    """
    Apply a threshold to each image in the stack and save the final result.
    
    Args:
    result_stack (numpy.ndarray): The stack of images to threshold.
    thresh_dir (str): Directory to save the thresholded stack.
    
    Returns:
    numpy.ndarray: The thresholded image stack.
    """
    thresholded_stack = np.array([apply_threshold_clip(image) for image in result_stack])
    #display_first_4_images(thresholded_stack, title_prefix='Thresholded Stack')

    # Save thresholded final stack
    stack_output_path = os.path.join(thresh_dir, 'Threshold_Final_Stack.tif')
    tiff.imwrite(stack_output_path, thresholded_stack.astype(np.float32))

    return thresholded_stack.astype(np.float32)

def save_bf(bfchannelbefore, bf_dir):
    """
    Save individual brightfield images with sequential filenames.
    
    Args:
    bfchannelbefore (list): List of brightfield images to save.
    bf_dir (str): Directory to save the images.
    
    Returns:
    None
    """
    # Determine the number of digits in the largest index
    max_index = len(bfchannelbefore) - 1
    num_digits = math.ceil(math.log10(max_index + 1)) if max_index > 0 else 1

    # Format string for leading zeros
    format_str = f"{{:0{num_digits}d}}"

    # Save bfchannelbefore to folder
    for idx, img in enumerate(bfchannelbefore):
        filename = f"bf_before_{format_str.format(idx)}.tiff"
        filepath = os.path.join(bf_dir, filename)
        cv2.imwrite(filepath, img)

def normalise_ml_stack(segmentation_stack):
    """
    Normalize each image in the segmentation stack and convert to binary.
    
    Parameters:
    segmentation_stack (list): List of images in the segmentation stack.
    
    Returns:
    list: A list of normalized binary images.
    """
    normalised_stack = []
    for image in segmentation_stack:
        # Normalise and convert the image to binary
        normalised_image = (image / np.max(image) > 0).astype(np.uint8)
        normalised_stack.append(normalised_image)
    return normalised_stack # return in list form

def save_indv_images(image_stack, directory, base_filename):
    """
    Save individual images from a stack as separate files.
    
    Parameters:
    image_stack (numpy.ndarray): Stack of images to save.
    directory (str): Directory to save the images.
    base_filename (str): Base filename for the saved images.
    
    Returns:
    None
    """
    os.makedirs(directory, exist_ok=True)
    for idx, image in enumerate(image_stack):
        filename = os.path.join(directory, f"{base_filename}_{idx:04d}.tiff")
        tiff.imwrite(filename, image)
    