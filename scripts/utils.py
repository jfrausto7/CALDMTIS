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
    labels = ['SDXL 1.0 base'] * length + ['SDXL 1.0 base+refiner'] * length + ['SDXL 0.9 base'] * length + ['SDXL 0.9 base+refiner'] * length + ['SD 2.1'] * length + ['SD 1.5'] * length

    plt.figure(figsize=(10, 6))

    sns.stripplot(x=labels, y=data, jitter=True, dodge=True, palette='Set2')

    plt.xlabel('Model')
    plt.ylabel(f'{metric} Score')
    plt.title(f'Strip Plot of {metric} Scores')

    plt.tight_layout()

    plt.savefig(f'figures/{metric}_stripplot.png')

    plt.show()

def generate_violinplot(scores, metric):
    """
    Generate a violin plot to visualize the distribution of metric scores across models.

    Args:
        scores (list of arrays): Metric scores for different models.
        metric (str): The name of the metric being visualized.
    """
    data = scores
    labels = ["SDXL 1.0 base", "SDXL 1.0 base+refiner", "SDXL 0.9 base", "SDXL 0.9 base+refiner", "SD 2.1", "SD 1.5"]

    plt.figure(figsize=(10, 6))

    sns.violinplot(data=data, inner="quartile")
    plt.xticks(range(len(data)), labels)

    plt.xlabel('Model')
    plt.ylabel(f'{metric} Score')
    plt.title(f'Violin Plot of {metric} Scores')

    plt.tight_layout()

    plt.savefig(f'figures/{metric}_violinplot.png')

    plt.show()

def generate_GMSD_heatmap(matrix_list, cmap='viridis'):
    if len(matrix_list) == 0:
        raise ValueError("Input matrix list is empty")

    matrix_shape = matrix_list[0].shape
    for matrix in matrix_list:
        if matrix.shape != matrix_shape:
            raise ValueError("Matrices in the list must have the same shape")

    averaged_matrix = np.mean(matrix_list, axis=0)
    num_rows, num_cols = averaged_matrix.shape

    triangular_matrix = np.zeros((num_rows, num_cols))
    for i in range(num_rows):
        for j in range(i, num_cols):
            triangular_matrix[i, j] = averaged_matrix[i, j]

    labels = ["SDXL 1.0 b", "SDXL 1.0 b+r", "SDXL 0.9 b", "SDXL 0.9 b+r", "SD 2.1", "SD 1.5"]

    plt.figure(figsize=(10, 8))
    plt.imshow(triangular_matrix, cmap=cmap, origin='lower', aspect='auto')
    plt.colorbar(label='Averaged GMSD Value')
    plt.title('Averaged GMSD Matrix (Triangular Heatmap)')
    plt.xlabel("Model")
    plt.ylabel("Model")
    plt.xticks(ticks=np.arange(len(labels)), labels=labels)
    plt.yticks(ticks=np.arange(len(labels)), labels=labels)
    plt.savefig('figures/GMSD_heatmap.png')
    plt.show()

def generate_correlation_heatmap(metric_scores, model_names, cmap='coolwarm'):
    """
    Generate a triangular heatmap to visualize the average correlation matrix between metrics for different models and scenarios.
    
    Args:
        metric_scores (numpy.ndarray): A 3D numpy array containing metric scores for different metrics, models, and scenarios.
                                      Shape: (num_metrics, num_models, num_scenarios)
        model_names (list): List of model names corresponding to the models' dimension of the metric_scores array.
        cmap (str, optional): Colormap for the heatmap. Default is 'coolwarm'.
        
    Returns:
        None
    """
    num_metrics, num_models, num_scenarios = metric_scores.shape
    
    # Calculate the average correlation matrix across all scenarios
    avg_correlation_matrix = np.mean([np.corrcoef(metric_scores[:, :, i], rowvar=False) for i in range(num_scenarios)], axis=0)
    
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(avg_correlation_matrix, dtype=bool))
    
    # Initialize a figure for the heatmap
    plt.figure(figsize=(10, 8))
    
    # Plot the triangular heatmap
    sns.heatmap(avg_correlation_matrix, mask=mask, annot=True, cmap=cmap, xticklabels=model_names, yticklabels=model_names)
    plt.title("Average Correlation of Metrics Heatmap")
    
    plt.tight_layout()
    plt.savefig('figures/metric_correlation.png')
    plt.show()