import cv2
import tifffile as tiff
import numpy as np
import os

stack_path = '/vol/biomedic3/bglocker/mscproj24/nma23/data/testing_directory/multi_model/eval_test_data/Model_vit_l_lm_ais/Segmentation_Output/Mask_Stack.tiff'
stack = tiff.imread(stack_path)

# Get the number of slices in the stack
num_slices = stack.shape[0]

# Create dilation kernel (7x7)
kernel = np.ones((7, 7), np.uint8)

# Specify save directory
output_directory = '/vol/biomedic3/bglocker/mscproj24/nma23/data/testing_directory/multi_model/eval_test_data/Model_dilated_vit_l_lm_ais/'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Dilate each slice and save it as an individual tiff
for i in range(num_slices):
    # Get slice
    slice_2d = stack[i, :, :]
    
    # Apply dilation
    dilated_slice = cv2.dilate(slice_2d, kernel, iterations=1)
    
    # Save 
    output_file = os.path.join(output_directory, f'dilated_mask_{i:03d}.tiff')
    cv2.imwrite(output_file, dilated_slice)

print("Dilation and saving of slices complete.")
