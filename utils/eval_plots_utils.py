import numpy as np
import pandas as pd
import tifffile as tiff
from skimage.io import imread
from skimage.color import rgb2gray
import seaborn as sns
import matplotlib.pyplot as plt

def process_data(file_paths_by_exp):
    """
    Processes data from multiple experiments by reading CSV files, calculating the mean and standard deviation for cell area and intensity, 
    and organizing the results into a DataFrame.

    Args:
        file_paths_by_exp (dict): Dictionary where keys are experiment names and values are dictionaries of model names and their file paths.

    Returns:
        pd.DataFrame: A DataFrame containing the experiment, model, mean cell area, mean intensity, and their standard deviations.
    
    Raises:
        ValueError: If the file does not contain the required number of columns.
    """

    results = []

    for exp, models in file_paths_by_exp.items():
        for model, file_path in models.items():
            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)
            
            if df.shape[1] > 2:  
                mean_area = round(df.iloc[:, 1].mean(), 2)
                mean_fluor = round(df.iloc[:, 2].mean(), 2)
                std_area = round(df.iloc[:, 1].std(), 2)
                std_fluor = round(df.iloc[:, 2].std(), 2)
                
                results.append({
                    'Experiment': exp,
                    'Model': model,
                    'Mean Cell Area': mean_area,
                    'Mean Intensity': mean_fluor,
                    'Std Cell Area': std_area,
                    'Std Intensity': std_fluor
                })
            else:
                print(f"File {file_path} does not have enough columns")

    return pd.DataFrame(results)

def w_gt_process_data(file_paths_by_model):
    """
    Processes data from multiple models by reading CSV files, calculating the mean and standard deviation for cell area and intensity, 
    and organizing the results into a DataFrame. Data must all be in folders in one directory, unlike the above function process_data() with experiments.

    Args:
        file_paths_by_model (dict): Dictionary where keys are model names and values are their respective file paths.

    Returns:
        pd.DataFrame: A DataFrame containing the model, mean cell area, mean intensity, and their standard deviations.
    
    Raises:
        ValueError: If the file does not contain the required number of columns.
    """

    results = []

    for model, file_path in file_paths_by_model.items():
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        
        if df.shape[1] > 2:  # Ensure the file has enough columns
            mean_area = round(df.iloc[:, 1].mean(), 2)
            mean_fluor = round(df.iloc[:, 2].mean(), 2)
            std_area = round(df.iloc[:, 1].std(), 2)
            std_fluor = round(df.iloc[:, 2].std(), 2)
            
            results.append({
                'Model': model,
                'Mean Cell Area': mean_area,
                'Mean Intensity': mean_fluor,
                'Std Cell Area': std_area,
                'Std Intensity': std_fluor
            })
        else:
            print(f"File {file_path} does not have enough columns")

    return pd.DataFrame(results)

def calculate_iou(segmentation1, segmentation2):
    """
    Calculates the Intersection over Union (IoU) score between two binary segmentation masks.

    Args:
        segmentation1 (np.ndarray): First segmentation mask (binary).
        segmentation2 (np.ndarray): Second segmentation mask (binary).

    Returns:
        float: IoU score representing the overlap between the two segmentation masks.
    """
    # Ensure the images are binary
    segmentation1 = segmentation1 > 0.5
    segmentation2 = segmentation2 > 0.5
    
    intersection = np.logical_and(segmentation1, segmentation2)
    union = np.logical_or(segmentation1, segmentation2)
    
    iou = np.sum(intersection) / np.sum(union)
    return iou


def read_stack(path):
    """
    Reads a stack of images from a TIFF file and converts them to grayscale if necessary.

    Args:
        path (str): Path to the TIFF file.

    Returns:
        np.ndarray: Image stack.
    """
    stack = imread(path, plugin='tifffile')
    # Convert slices to greyscale
    if stack.ndim == 4 and stack.shape[3] == 3:
        stack = np.array([rgb2gray(stack[i]) for i in range(stack.shape[0])])
    elif stack.ndim == 4 and stack.shape[3] != 3:
        stack = np.array([stack[i].mean(axis=2) for i in range(stack.shape[0])])
    return stack


def compute_iou_dataframe_phantast(phantast_stack_paths_by_exp):
    """
    Computes the mean IoU scores for segmentation masks produced by the PHANTAST model and other models.

    Args:
        phantast_stack_paths_by_exp (dict): Dictionary of experiment names and file paths to segmentation stacks for each model, 
                                            including PHANTAST .

    Returns:
        pd.DataFrame: A DataFrame with the experiment name, model name, mean IoU score, and individual IoU scores for each image slice.
    """
    results = []  # Initialize results inside the function
    for exp, model_paths in phantast_stack_paths_by_exp.items():
        phantast_stack = read_stack(model_paths['PHANTAST'])
        
        # Loop through each model in the experiment
        for model, path in model_paths.items():
            if model == 'Model_PHANTAST_':
                continue
            
            # Read the model stack
            model_stack = read_stack(path)
            
            assert phantast_stack.shape == model_stack.shape, "Stacks must have the same shape"
            
            # Calculate IoU for each slice
            iou_scores = [calculate_iou(phantast_stack[i], model_stack[i]) for i in range(phantast_stack.shape[0])]
            
            mean_iou = np.mean(iou_scores)
            results.append({
                'Experiment': exp,
                'Model': model,
                'Mean_IoU': mean_iou,
                'IoU_Scores': iou_scores
            })

    return pd.DataFrame(results)


def compute_iou_dataframe_gt(gt_stack_paths_by_exp):
    """
    Computes the Intersection over Union (IoU) scores for a set of ground truth and predicted image stacks 
    across multiple experiments and models. The function calculates the IoU for each slice in the stack 
    and returns a DataFrame with the mean IoU score per model and experiment.

    Parameters:
    gt_stack_paths_by_exp (dict): A dictionary where the keys are experiment names, and the values are 
                                  dictionaries with model names as keys and file paths as values. 
                          
    Returns:
    pandas.DataFrame: A DataFrame containing the mean IoU and individual IoU scores for each model and experiment. 
                      Columns:
                      - 'Experiment': Name of the experiment.
                      - 'Model': Name of the model.
                      - 'Mean_IoU': Mean IoU score across all slices in the stack.
                      - 'IoU_Scores': List of IoU scores for each slice in the stack.
    """
    results = []  # Initialize results inside the function
    
    for exp, model_paths in gt_stack_paths_by_exp.items():
        gt_stack = read_stack(model_paths['Ground_truth'])
        
        # Loop through each model in  experiment
        for model, path in model_paths.items():
            if model == 'Ground_truth':
                continue
            
            model_stack = read_stack(path)
            
            assert gt_stack.shape == model_stack.shape, "Stacks must have the same shape"
            
            # Calculate IoU for each slice
            iou_scores = [calculate_iou(gt_stack[i], model_stack[i]) for i in range(gt_stack.shape[0])]
            mean_iou = np.mean(iou_scores)
        
            results.append({
                'Experiment': exp,
                'Model': model,
                'Mean_IoU': mean_iou,
                'IoU_Scores': iou_scores
            })

    return pd.DataFrame(results)

# Recall = TP / (TP + FN)
def recall_score_(gt_mask, pred_mask):
    """
    Computes the recall score between a predicted mask and a ground truth mask.

    Parameters:
    gt_mask (numpy array): Ground truth mask where positive pixels are 1 and others are 0.
    pred_mask (numpy array): Predicted mask where positive pixels are 1 and others are 0.
    
    Returns:
    float: Recall score, rounded to 6 decimal places.
    """
    intersect = np.sum(pred_mask*gt_mask)
    total_pixel_truth = np.sum(gt_mask)
    recall = np.mean(intersect/total_pixel_truth)
    return round(recall, 6)

# Precision = TP / (TP + FP)
def precision_score_(gt_mask, pred_mask):
    """
    Computes the precision score between a predicted mask and a ground truth mask.
    
    Parameters:
    gt_mask (numpy array): Ground truth mask where positive pixels are 1 and others are 0.
    pred_mask (numpy array): Predicted mask where positive pixels are 1 and others are 0.
    
    Returns:
    float: Precision score, rounded to 6 decimal places.
    """
    intersect = np.sum(pred_mask*gt_mask)
    total_pixel_pred = np.sum(pred_mask)
    precision = np.mean(intersect/total_pixel_pred)
    return round(precision, 6)

# Dice = 2TP / (2TP + FP + FN)
def dice_coef(gt_mask, pred_mask):
    """
    Computes the Dice Coefficient between a predicted mask and a ground truth mask.
    
    Parameters:
    gt_mask (numpy array): Ground truth mask where positive pixels are 1 and others are 0.
    pred_mask (numpy array): Predicted mask where positive pixels are 1 and others are 0.
    
    Returns:
    float: Dice coefficient, rounded to 6 decimal places.
    """
    intersect = np.sum(pred_mask*gt_mask)
    total_sum = np.sum(pred_mask) + np.sum(gt_mask)
    dice = np.mean(2*intersect/total_sum)
    return round(dice, 6)

# can use read_stack instead 
def load_mask(file_path):
    """
    Loads a binary mask from a given file path.

    Parameters:
    file_path (str): Path to the mask file.

    Returns:
    numpy array: The binary mask loaded from the file.
    """
    return imread(file_path)


def plot_scatter_plots(data, compare_choice_data, models_choice, data_to_plot, y_axis_label):
    """
    Plots scatter plots for comparing ground truth data with model predictions. Each model is compared 
    against the ground truth for the selected data, and all plots are displayed as subplots.

    Parameters:
    data (dict): Dictionary containing model data where keys are model names and values are DataFrames with data.
    compare_choice_data (dict): Dictionary containing ground truth data (key: 'Ground_truth') and values as DataFrames.
    models_choice (list): List of model names to compare against ground truth.
    data_to_plot (str): Column name representing the data to be compared between ground truth and model predictions.
    y_axis_label (str): Label for the Y-axis of the scatter plots.

    Returns:
    None: Displays a grid of scatter plots comparing ground truth and model predictions.
    """
    num_models = len(models_choice)
    num_cols = 3  
    num_rows = (num_models + num_cols - 1) // num_cols 

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(5 * num_cols, 6 * num_rows), sharex=True, sharey=True)
    sns.despine(left=True, bottom=True)

    palettes = sns.color_palette("bright", len(models_choice))

    min_val = float('inf')
    max_val = float('-inf')

    for m_choice in models_choice:
        m_data = data.get(m_choice)
        gt_data = compare_choice_data.get('Ground_truth')

        if m_data is not None and gt_data is not None:
            if data_to_plot in m_data.columns and data_to_plot in gt_data.columns:
                min_val = min(min_val, gt_data[data_to_plot].min(), m_data[data_to_plot].min())
                max_val = max(max_val, gt_data[data_to_plot].max(), m_data[data_to_plot].max())
            else:
                print(f"Column {data_to_plot} is missing in the data for {m_choice}.")
        else:
            print(f"Data for model {m_choice} or ground truth is missing.")

    padding = (max_val - min_val) * 0.05
    min_val -= padding
    max_val += padding

    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    axes = axes.flatten()

    for i, m_choice in enumerate(models_choice):
        m_data = data.get(m_choice)
        if m_data is not None and data_to_plot in m_data.columns:
            sns.scatterplot(x=compare_choice_data['Ground_truth'][data_to_plot], y=m_data[data_to_plot], ax=axes[i], hue=m_data['Model'], palette=[palettes[i]])
            axes[i].set_title(f'Model: {m_choice}')
            axes[i].set_xlabel(f"{y_axis_label} (Ground Truth)")
            axes[i].set_ylabel(f'{y_axis_label} (Model)')
            axes[i].set_xlim(min_val, max_val)
            axes[i].set_ylim(min_val, max_val)
            axes[i].set_aspect('equal', 'box')
        else:
            print(f"Skipping plot for {m_choice} as data is missing.")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def load_nogt_data(model_name, file_paths_by_exp): # Passing in each model in a loop, so ground truth is automatically excluded
    """
    Loads data for a specific model across different experiments (excluding ground truth data, based on the filepath setup).
    Reads the data from CSV files and combines them into a DataFrame.

    Parameters:
    model_name (str): Name of the model to load data for.
    file_paths_by_exp (dict): Dictionary of file paths where keys are experiment names, and values are paths 
                              to the model's result CSV file for each experiment.

    Returns:
    pandas.DataFrame: A DataFrame containing data from all experiments for the specified model.
    If no data is found, an empty DataFrame is returned.
    """
    data_list = []
    for exp in file_paths_by_exp:
        if model_name in file_paths_by_exp[exp]:  
            file_path = file_paths_by_exp[exp][model_name]
            print(f"Loading data for model {model_name} from file: {file_path}")
            try:
                df = pd.read_csv(file_path)  # Read in Results.csv data
                df['Experiment'] = exp
                df['Model'] = model_name
                data_list.append(df)
            except Exception as e:
                print(f"Failed to load data for {model_name} from {file_path}: {e}")
        else:
            print(f"Model {model_name} not found in experiment {exp}")
    
    if data_list:  # Concatenate data (if available)
        return pd.concat(data_list, ignore_index=True)
    else:
        print(f"No data found for model {model_name}. Returning empty DataFrame.")
        return pd.DataFrame()  # Return empty DataFrame if no data


def load_gt_data(gt_results_csv_file_paths):
    """
    Loads ground truth data from CSV files for multiple experiments and models.
    Combines the data into a dictionary with model names as keys.

    Parameters:
    gt_results_csv_file_paths (dict): A dictionary where keys are experiment names, and values are dictionaries 
                                      mapping model names to file paths for the ground truth CSV data.

    Returns:
    dict: A dictionary where keys are model names, and values are DataFrames containing the ground truth data.
    """
    gt_data = {}
    for exp, models in gt_results_csv_file_paths.items():
        for model, path in models.items():
            # Load the data from the CSV file
            df = pd.read_csv(path)
            
            # Add the 'Model' column
            df['Model'] = model
            
            if model not in gt_data:
                gt_data[model] = df
            else:
                # If already exists, concatenate or otherwise handle the new data
                gt_data[model] = pd.concat([gt_data[model], df], ignore_index=True)
    
    return gt_data


def plot_box_plots(all_data, x_data, x_axis_label, ground_truth_df=None, stripplot=False):
    """
    Plots box plots for data across multiple models, with an optional strip plot overlay.

    Parameters:
    all_data (pandas.DataFrame): DataFrame containing the data to be plotted, with 'Model' as one of the columns.
    x_data (str): Column name representing the data to be plotted on the x-axis.
    x_axis_label (str): Label for the x-axis.
    ground_truth_df (pandas.DataFrame, optional): DataFrame containing ground truth data for overlay (default: None).
    stripplot (bool, optional): Whether to overlay a strip plot on top of the box plot (default: False).

    Returns:
    None: Displays the box plots and optional strip plot.
    """
    # Initialize the figure with a logarithmic x axis
    sns.set_theme(style="ticks")
    f, ax = plt.subplots(figsize=(7, 6))
    ax.set_xscale("log")

    # Plot the box plot for all models in the DataFrame
    sns.boxplot(
        data=all_data, x=x_data, y="Model", hue="Model",
        whis=[0, 100], width=.6, palette="pastel", ax=ax
    )

    if stripplot:
        sns.stripplot(data=all_data, x=x_data, y="Model", size=4, color=".3", jitter=True, ax=ax)

    # Overlay the Ground_truth data if provided
    if ground_truth_df is not None:
        sns.boxplot(
            data=ground_truth_df, x=x_data, y=['Ground_truth']*len(ground_truth_df), 
            width=.6, palette={'Ground_truth': 'blue'}, ax=ax, boxprops=dict(alpha=.5), showfliers=False, hue=['Ground_truth']*len(ground_truth_df), legend=False
        )


    ax.xaxis.grid(True)
    ax.set(ylabel="")
    ax.set(xlabel=x_axis_label)
    sns.despine(trim=True, left=True)


def plot_scatter_phantast_vs_models(phantast_data, data_against_phantast, ml_models, data_to_plot, y_axis_label):
    """
    Plots scatter plots comparing the predictions of various models to the PHANTAST model.
    Each scatter plot compares PHANTAST's predictions against the predictions of a selected model, 
    with plots displayed in subplots.

    Parameters:
    phantast_data (pandas.DataFrame): DataFrame containing PHANTAST model data.
    data_against_phantast (dict): Dictionary containing model data where keys are model names and values are DataFrames.
    ml_models (list): List of machine learning models to compare against PHANTAST.
    data_to_plot (str): Column name representing the data to be compared.
    y_axis_label (str): Label for the y-axis of the scatter plots.

    Returns:
    None: Displays scatter plots comparing PHANTAST and each selected model.
    """
    num_models = len(ml_models)
    num_cols = 3  
    num_rows = (num_models + num_cols - 1) // num_cols 

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(5 * num_cols, 6 * num_rows), sharex=True, sharey=True)
    sns.despine(left=True, bottom=True)

    palettes = sns.color_palette("bright", len(ml_models))

    min_val = float('inf')
    max_val = float('-inf')

    for model in ml_models:
        model_data = data_against_phantast[model]

        if data_to_plot in model_data.columns:
            phantast_vals = model_data[model_data['Model'] == 'PHANTAST'][data_to_plot]
            model_vals = model_data[model_data['Model'] == model][data_to_plot]
            
            min_val = min(min_val, phantast_vals.min(), model_vals.min())
            max_val = max(max_val, phantast_vals.max(), model_vals.max())
        else:
            print(f"Column {data_to_plot} is missing in the data for {model}.")
            continue

    padding = (max_val - min_val) * 0.05
    min_val -= padding
    max_val += padding

    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    axes = axes.flatten()

    for i, model in enumerate(ml_models):
        model_data = data_against_phantast[model]
        if data_to_plot in model_data.columns:
            phantast_vals = model_data[model_data['Model'] == 'PHANTAST'][data_to_plot]
            model_vals = model_data[model_data['Model'] == model][data_to_plot]
            # Reset indices
            phantast_vals = phantast_vals.reset_index(drop=True)
            model_vals = model_vals.reset_index(drop=True)
            
            color = sns.color_palette("bright", len(ml_models))[i]
            sns.scatterplot(x=phantast_vals, y=model_vals, ax=axes[i], color=color)
            axes[i].set_title(f'PHANTAST vs {model}')
            axes[i].set_xlabel(f"{y_axis_label} (PHANTAST)")
            axes[i].set_ylabel(f"{y_axis_label} ({model})")
            axes[i].set_xlim(min_val, max_val)
            axes[i].set_ylim(min_val, max_val)
            axes[i].set_aspect('equal', 'box')
        else:
            print(f"Skipping plot for {model} as data is missing.")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()