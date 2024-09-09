import os
import random
import shutil

def distribute_images(source_folder, destination_folders, num_images_per_folder):
    # Get all image filenames in the source folder
    all_images = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    
    # Check if there are enough images
    total_images_needed = len(destination_folders) * num_images_per_folder
    if len(all_images) < total_images_needed:
        raise ValueError(f"Not enough images in the source folder. Needed: {total_images_needed}, Available: {len(all_images)}")
    
    # Randomly select the required number of unique images
    selected_images = random.sample(all_images, total_images_needed)
    
    # Distribute images to the destination folders
    for i, folder in enumerate(destination_folders):
        if not os.path.exists(folder):
            os.makedirs(folder)
        images_to_copy = selected_images[i*num_images_per_folder:(i+1)*num_images_per_folder]
        for image in images_to_copy:
            shutil.copy(os.path.join(source_folder, image), os.path.join(folder, image))
        print(f"Copied {len(images_to_copy)} images to {folder}")

        
source_folder = '/vol/biomedic3/bglocker/mscproj24/nma23/data/Jones_data/16FOV_1_tifs'
destination_folders = ['/vol/biomedic3/bglocker/mscproj24/nma23/data/testing_directory/multi_model/exp1', '/vol/biomedic3/bglocker/mscproj24/nma23/data/testing_directory/multi_model/exp2', '/vol/biomedic3/bglocker/mscproj24/nma23/data/testing_directory/multi_model/exp3', '/vol/biomedic3/bglocker/mscproj24/nma23/data/testing_directory/multi_model/exp4', '/vol/biomedic3/bglocker/mscproj24/nma23/data/testing_directory/multi_model/exp5']
num_images_per_folder = 50

distribute_images(source_folder, destination_folders, num_images_per_folder)
