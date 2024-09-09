# msc_prj_jones_workflow

This project addresses whole cell segmentation in phase contrast microscopy images. Specifically, this project was designed to segment HEK293 cells used by a research lab at Imperial College London. This project is submitted in partial fulfillment of the requirements for the MSc degree in Computing of Imperial College London.

The contents are as follows:
- analysis_notebooks: Contains Jupyter notebooks for creating plots, tables, etc.
- joneslab_segmentation_plugin: Contains the code needed to install and run the Jones Lab Segmentation plugin for FIJI/ImageJ, developed as part of this project. See the README in the joneslab_segmentation_plugin folder for me details on contents and installation.
- plugin_dir: Contains the python script and helper script to be called by the Jones Lab Segmentation plugin.
- thresholds: Contains JSON files with the best performining thresholds for each finetuned model in this project.
- utils: Contains utility functions used throughout the project, organised into general use categories.
- adapted_macro.ijm: The ImageJ macro used by the Jones Lab, modified to call the Jones Lab Segmentation plugin for segmentation and integrate the results back into the macro. This is modified from the Jones Lab Receptor Expression MESNA macro.*
- combined_jones_script.py: The Python recreation of the Jones Lab Receptor Expression MESNA macro.
- image_compare.ipynb, image_sort.py: Contains files to assist in testing.
- local_histogram_eq.ipynb: The file used for generating segmentations and analysis of local histogram equalisation.
- nma_dilate.py: The file used for adding a morphological dilation to model predictions.
- nma_eval_only.py: The evaluation script fot the finetuned models.
- nma_finetune.py: The training script for the SAM and micro_sam finetuning procedure.
- nma_inference.py: The inference script for the finetuned models.


* The full Receptor Expression MESNA macro can be found here: https://github.com/engpol/JonesLabFIJIScripts/blob/main/Receptor_Expression_Macro_MESNA_MAC.ijm