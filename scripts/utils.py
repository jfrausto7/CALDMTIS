import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def generate_stripplot(scores, metric):
    """
    Generate a strip plot to visualize the distribution of metric scores across models.

    Args:
        scores (list of arrays): Metric scores for different models.
        metric (str): The name of the metric being visualized.
    """
    data = np.concatenate(scores)
    length = len(scores[0])
    labels = ['SDXL 1.0 base'] * length + ['SDXL 1.0 base+refiner'] * length + ['SDXL 0.9 base'] * length + ['SDXL 0.9 base+refiner'] * length + ['SD 1.5'] * length + ['SD 2.1'] * length

    plt.figure(figsize=(10, 6))

    sns.stripplot(x=labels, y=data, jitter=True, dodge=True, palette='Set2')

    plt.xlabel('Model')
    plt.ylabel(f'{metric} Score')
    plt.title(f'Strip Plot of {metric} Scores')

    plt.tight_layout()

    plt.show()

def generate_violinplot(scores, metric):
    """
    Generate a violin plot to visualize the distribution of metric scores across models.

    Args:
        scores (list of arrays): Metric scores for different models.
        metric (str): The name of the metric being visualized.
    """
    data = scores
    labels = ["SDXL 1.0 base", "SDXL 1.0 base+refiner", "SDXL 0.9 base", "SDXL 0.9 base+refiner", "SD 1.5", "SD 2.1"]

    plt.figure(figsize=(10, 6))

    sns.violinplot(data=data, inner="quartile")
    plt.xticks(range(len(data)), labels)

    plt.xlabel('Model')
    plt.ylabel(f'{metric} Score')
    plt.title(f'Violin Plot of {metric} Scores')

    plt.tight_layout()

    plt.show()

def generate_heatmap(matrix_list, cmap='viridis'):
    """
    Generate a heatmap visualization of an averaged GMSD matrix.

    Args:
        matrix_list (list of np.ndarray): List of matrices to average and visualize.
        cmap (str): Colormap to use for visualization (default is 'viridis').

    Returns:
        None
    """
    if len(matrix_list) == 0:
        raise ValueError("Input matrix list is empty")

    # Calculate the element-wise average of the matrices
    matrix_shape = matrix_list[0].shape
    for matrix in matrix_list:
        if matrix.shape != matrix_shape:
            raise ValueError("Matrices in the list must have the same shape")

    # Average all matrices
    averaged_matrix = np.mean(matrix_list, axis=0)

    labels = ["SDXL 1.0 base", "SDXL 1.0 base+refiner", "SDXL 0.9 base", "SDXL 0.9 base+refiner", "SD 1.5", "SD 2.1"]

    # Generate heatmap visualization
    plt.figure(figsize=(8, 6))
    plt.imshow(averaged_matrix, cmap=cmap, origin='lower', aspect='auto')
    plt.colorbar(label='Averaged GMSD Value')
    plt.title('Averaged GMSD Matrix')
    plt.xlabel(labels)
    plt.ylabel(labels)
    plt.show()