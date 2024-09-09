import matplotlib as plt
import numpy as np

from glob import glob
import h5py
from skimage.measure import label as connected_components
from torch_em.util.util import get_random_colors
from torch_em.data.datasets.covid_if import _download_covid_if
from micro_sam import util
from micro_sam.evaluation.model_comparison import _enhance_image
from micro_sam.instance_segmentation import (
    InstanceSegmentationWithDecoder,
    AutomaticMaskGenerator,
    get_predictor_and_decoder,
    mask_data_to_segmentation
)
def get_bit_depth_stack(stack):
    """
    Determine the bit depth of a given image stack based on its data type.

    Parameters:
    stack (numpy.ndarray): The input image stack.

    Returns:
    str: The bit depth as a string description.
    """
    print(f"Debugging: dtype of the stack is {stack.dtype}")

    if stack.dtype == np.uint8:
        return '8-bit unsigned integer'
    elif stack.dtype == np.uint16:
        return '16-bit unsigned integer'
    elif stack.dtype == np.uint32:
        return '32-bit unsigned integer'
    elif stack.dtype == np.float16:
        return '16-bit floating point'
    elif stack.dtype == np.float32:
        return '32-bit floating point'
    elif stack.dtype == np.float64:
        return '64-bit floating point'
    else:
        return 'Unknown'
    

def get_bit_depth_image(image):
    """
    Get the data type (bit depth) of a single image.

    Args:
    image (numpy.ndarray): The input image.

    Returns:
    dtype: The data type of the image.
    """
    return image.dtype

def display_first_4_images(image_stack, title_prefix='Image Stack', cmap='gray'):
    """
    Display the first 4 images from an image stack in a 2x2 grid.

    Args:
    image_stack (numpy.ndarray): The stack of images.
    title_prefix (str): The prefix for image titles.
    cmap (str): The color map to use for displaying the images.

    Returns:
    None
    """
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    fig.suptitle(title_prefix)
    
    for i, ax in enumerate(axes.flat):
        if i < len(image_stack):
            ax.imshow(image_stack[i], cmap=cmap)
            ax.set_title(f'{title_prefix} - Image {i}')
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def display_all_images(image_stack, title_prefix='Image Stack', cmap='gray'):
    """
    Display all images from a stack in a grid layout.

    Args:
    image_stack (numpy.ndarray): The stack of images.
    title_prefix (str): The prefix for the image titles.
    cmap (str): The color map to use for displaying the images.

    Returns:
    None
    """
    n_images = len(image_stack)
    n_cols = 4  # Number of columns
    n_rows = int(np.ceil(n_images / n_cols))  # Calculate rows needed

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    fig.suptitle(title_prefix)
    
    for i, image in enumerate(image_stack):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.imshow(image, cmap=cmap)
        ax.set_title(f'Slice {i+1}')
        ax.axis('off')
    
    # Hide any remaining empty subplots
    for j in range(i + 1, n_rows * n_cols):
        row = j // n_cols
        col = j % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# Convenience function to plot images side-by-side
def plot_samples(image, gt=None, segmentation=None):
    """
    Plot an image, ground truth, and segmentation side by side for comparison.

    Args:
    image (numpy.ndarray): The original input image.
    gt (numpy.ndarray, optional): The ground truth image.
    segmentation (numpy.ndarray, optional): The predicted segmentation.

    Returns:
    None
    """
    if gt is None:
        n_images = 1 if segmentation is None else 2
    else:
        n_images = 2 if segmentation is None else 3
    
    fig, ax = plt.subplots(1, n_images, figsize=(10, 10))
    if n_images == 1:
        ax = [ax]
    ax[0].imshow(_enhance_image(image, do_norm=True), cmap="gray")
    ax[0].axis("off")
    ax[0].set_title("Image")

    if gt is not None:
        gt = connected_components(gt)
        ax[1].imshow(gt, cmap=get_random_colors(gt), interpolation="nearest")
        ax[1].axis("off")
        ax[1].set_title("Ground Truth")    
    
    if segmentation is not None:
        if gt is None:
            ax[1].imshow(segmentation, cmap=get_random_colors(segmentation), interpolation="nearest")
            ax[1].axis("off")
            ax[1].set_title("Prediction")
        else:
            ax[2].imshow(segmentation, cmap=get_random_colors(segmentation), interpolation="nearest")
            ax[2].axis("off")
            ax[2].set_title("Prediction")

    plt.show()
    plt.close()

def print_segmentation_info(all_segmentations):
    """
    Print detailed information about segmentations from different models.

    Args:
    all_segmentations (dict): A dictionary where keys are model names and values are lists of segmentations.

    Returns:
    None
    """
    print(f"Overall type: {type(all_segmentations)}")
    for model, seg_list in all_segmentations.items():
        print(f"Model: {model}")
        print(f"  Number of segmentations: {len(seg_list)}")
        for idx, segmentation in enumerate(seg_list):
            print(f"  Segmentation {idx}:")
            print(f"    Type: {type(segmentation)}")
            print(f"    Dtype: {segmentation.dtype}")
            print(f"    Shape: {segmentation.shape}")



def plot_masks(imgs, imgs_dn, masks, indices, model, model_type, restore_type):
    """
    Plot noisy images, denoised images, and their corresponding segmentation masks.

    Args:
    imgs (numpy.ndarray): Original noisy images.
    imgs_dn (numpy.ndarray): Denoised images.
    masks (numpy.ndarray): Segmentation masks.
    indices (list): List of indices for images to be plotted.
    model (str): The model used for segmentation.
    model_type (str): Type of the model (e.g., denoising, segmentation).
    restore_type (str): The restoration method used.

    Returns:
    None
    """
    print(f"Model: {model}")
    plt.figure(figsize=(12, 12))

    for i, idx in enumerate(indices):
        # Noisy image
        img = imgs[idx].squeeze()
        plt.subplot(3, len(indices), 1 + i)
        plt.imshow(img, cmap="gray")
        plt.axis('off')
        plt.title(f"Noisy {idx}")

        # Denoised image
        img_dn = imgs_dn[idx].squeeze()
        plt.subplot(3, len(indices), len(indices) + 1 + i)
        plt.imshow(img_dn, cmap="gray")
        plt.axis('off')
        plt.title(f"Denoised {idx} via {restore_type}")

        # Segmented image
        plt.subplot(3, len(indices), 2 * len(indices) + 1 + i)
        plt.imshow(masks[idx], cmap="gray")
        plt.axis('off')
        plt.title(f"Segmentation {idx} via {model_type}")

    plt.tight_layout()
    plt.show()