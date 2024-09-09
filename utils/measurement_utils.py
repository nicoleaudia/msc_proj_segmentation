import os
import numpy as np
import pandas as pd

def calc_area(image, low=1, high=65535):
    """
    Math function that calculates the total area of pixels within a specified intensity range for an image.

    Args:
        image (np.ndarray): The image array to be processed.
        low (int, optional): The lower threshold for pixel values to be included. Defaults to 1.
        high (int, optional): The upper threshold for pixel values to be included. Defaults to 65535.

    Returns:
        tuple: A tuple containing:
            - total_area (int): The total area (number of pixels) that meet the threshold condition.
            - low (int): The lower threshold used for the calculation.
            - high (int): The upper threshold used for the calculation.
    """
    binary = (image >= low) & (image <= high)
    total_area = np.count_nonzero(binary)

    return total_area, low, high

# Collect measurements for mean cell area and mean fluorescent intensity
def collect_measurements(result_stack, thresholded_stack, results_dir):
    """
    Collects measurements for the total cell area and mean fluorescent intensity for each slice in the result stack.

    Args:
        result_stack (np.ndarray): A 3D array (stack) of image slices to be measured.
        thresholded_stack (np.ndarray): A 3D array of the same dimensions as result_stack, where pixel intensities above a threshold 
                                        are used to calculate mean fluorescent intensity.
        results_dir (str): Directory where the results will be saved as a CSV file.

    Returns:
        None. The results (area, intensity, etc.) are printed to the console and saved to a CSV file in the specified directory.
    """
    # Measure properties for each slice in the stack
    results = []

    for i, image in enumerate(result_stack):
        total_area, min_thr, max_thr = calc_area(image)
        mean_intensity = np.mean(result_stack[i][thresholded_stack[i] > 0])
        
        # Print the results to screen in three columns
        print(f"Slice {i}: Total area = {total_area}, Mean intensity = {mean_intensity:.2f}")
        
        # Label and append the measurements for this slice
        label = f"Result of Corrected_Flo_Image.tiff:{i+1}"
        results.append({
            'Label': label,
            'Area': total_area,
            'Mean': mean_intensity,
            'Slice': i+1,
            'MinThr': min_thr,
            'MaxThr': max_thr 
        })

    # Save results as DataFrame
    results_df = pd.DataFrame(results)

    # Round intensity to 2 decimal points
    results_df['Mean'] = results_df['Mean'].round(2)

    # Save the results to a CSV file
    csv_path = os.path.join(results_dir, 'Results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Contrast and thresh section: Results saved to {csv_path}")


# Modified from collect_measurements, does not measure fluorescence
def collect_area_measurement(result_stack, results_dir):
    """
    Collects measurements for the total cell area for each slice in the result stack. Unlike collect_measurements, this function 
    does not calculate fluorescent intensity.

    Args:
        result_stack (np.ndarray): A 3D array (stack) of image slices to be measured.
        results_dir (str): Directory where the results will be saved as a CSV file.

    Returns:
        None. The results (area, thresholds, etc.) are printed to the console and saved to a CSV file in the specified directory.
    """
    results = []

    for i, image in enumerate(result_stack):
        total_area, min_thr, max_thr = calc_area(image)

        # Print the results to screen in three columns
        print(f"Slice {i}: Total area = {total_area}")
        
        # Label and append the measurements for this slice
        label = f"Result of ground truth tiff:{i+1}"
        results.append({
            'Label': label,
            'Area': total_area,
            'Mean': 'NA',
            'Slice': i+1,
            'MinThr': min_thr,
            'MaxThr': max_thr 
        })

    # Save results as DataFrame
    results_df = pd.DataFrame(results)

    # Round intensity to 2 decimal points
    results_df['Mean'] = results_df['Mean'].round(2)

    # Save the results to a CSV file
    csv_path = os.path.join(results_dir, 'Results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")