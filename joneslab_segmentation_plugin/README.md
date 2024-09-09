This is a Maven project implementing an ImageJ2 plugin called "Jones Lab Segmentation".

The purpose of the plugin is to perform whole cell segmentation on phase contrast microscopy images; specifically, HEK293 cells from the Jones lab.

To use this plugin:
1. Clone the repo
2. Build a conda environment from the appropriate yaml file.
3. Install any other necessary dependencies (tifffile, natsort, etc) to the virtual environment.
4. Install micro-sam to the virtual environment. It is recommended by the authors to use mamba. Micromamba in the base environment (alongside conda) is usually appropriate, if your system does not have an existing mamba installation. Instructions on installing micro-sam here: https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#installation
5. Place the JAR file in the plugins directory for the FIJI application.
6. Place the Python directory (script and helper file) and model checkpoints in the desired location. The application's base directory (e.g. 'Fiji.app') is usually appropriate.
7. Ensure all filepaths in the Python and Java scripts are correct for your workflow.
8. If anything was changed in the Java project, run a clean install (mvn clean install -X). Once complete, copy the JAR file to the FIJI plugins directory.
9. Set the environment variable to your Python installation. This works best if it is the Python installed in the virtual environment.
10. **Launch FIJI from the terminal** by navigating to the directory and running the executable.
11. Find the plugin in the Plugins menu, and go!


This plugin implements finetuned SAM and micro_sam models as FIJI/ImageJ plugins. Credit to the original is as follows:

SAM: A. Kirillov et al., "Segment Anything," 2023 IEEE/CVF International Conference on Computer Vision (ICCV), Paris, France, 2023, pp. 3992-4003, doi: 10.1109/ICCV51070.2023.00371.
SAM github: https://github.com/facebookresearch/segment-anything

micro_sam: Segment Anything for Microscopy
Anwai Archit, Sushmita Nair, Nabeel Khalid, Paul Hilt, Vikas Rajashekar, Marei Freitag, Sagnik Gupta, Andreas Dengel, Sheraz Ahmed, Constantin Pape
bioRxiv 2023.08.21.554208; doi: https://doi.org/10.1101/2023.08.21.554208

micro_sam github: https://github.com/computational-cell-analytics/micro-sam