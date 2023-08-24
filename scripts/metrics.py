import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
import imquality.brisque as brisque

def calculate_clip_score(image, prompt, clip_score_fn):
    """
    Calculate the CLIP score for an image and a given prompt using a specified CLIP score function.

    This function takes an image, a prompt, and a CLIP score function as inputs. The CLIP score function should
    accept an image tensor and a list of prompts as arguments and return a CLIP score tensor. The function then
    calculates the CLIP score for the provided image and prompt using the specified CLIP score function.

    Low score = less similarity to caption; High score = high similarity to caption
    
    https://arxiv.org/abs/2104.08718

    Args:
        image (numpy.ndarray): The input image as a NumPy array.
        prompt (str): The prompt to be used for calculating the CLIP score.
        clip_score_fn (Callable): The CLIP score function that accepts an image tensor and a list of prompts
                                 and returns a CLIP score tensor.

    Returns:
        float: The calculated CLIP score rounded to four decimal places.
    """
    clip_score = clip_score_fn(
        torch.from_numpy(image).unsqueeze(0).permute(0, 3, 1, 2), [prompt]
    ).detach()
    return round(float(clip_score), 4)

def calculate_niqe(image):
    """
    Calculate the Naturalness Image Quality Evaluator (NIQE) score for a given image.

    The NIQE score is calculated based on local mean and standard deviation statistics of the image.

    Low score = higher quality; High score = more distortion, noise, artifacts, etc.

    https://live.ece.utexas.edu/research/quality/niqe_spl.pdf

    Args:
        image (numpy.ndarray): The input image as a NumPy array.

    Returns:
        float: The calculated NIQE score representing the image quality.
    """
    # Calculate local mean and standard deviation
    image = np.array(image, dtype=np.float32) / 255.0
    window_size = 7
    mean_map = cv2.boxFilter(image, -1, (window_size, window_size))
    squared_diff_map = (image - mean_map)**2
    var_map = cv2.boxFilter(squared_diff_map, -1, (window_size, window_size))
    std_map = np.sqrt(var_map)

    # Calculate overall mean and standard deviation of local means and standard deviations
    overall_mean_mean = np.mean(mean_map)
    overall_std_mean = np.std(mean_map)
    overall_mean_std = np.mean(std_map)
    overall_std_std = np.std(std_map)

    # Calculate simplified NIQE score
    niqe_score = overall_mean_std / overall_std_mean + overall_std_std / overall_mean_mean

    return niqe_score

def calculate_brisque(image):
    """
    Calculate the Blind/Referenceless Image Spatial Quality Evaluator (BRISQUE) score for a given image.

    The BRISQUE score quantifies the perceived quality of an image without requiring a reference image.

    Low score = higher quality; High score = more distortion, noise, artifacts, etc.

    https://live.ece.utexas.edu/publications/2012/TIP%20BRISQUE.pdf

    Args:
        image (numpy.ndarray): The input image as a NumPy array.

    Returns:
        float: The calculated BRISQUE score representing the image quality.
    """
    image = np.array(image, dtype=np.float32) / 255.0
    return brisque.score(image)

def calculate_teng(image):
    """
    Calculate the Tenengrad focus measure of an input image.

    The Tenengrad focus measure operator quantifies the sharpness of an image
    based on the gradient magnitude. It measures the average squared gradient
    magnitude of the image, which indicates the amount of detail and contrast
    present. Higher values suggest greater focus and sharper edges.

    Low score = few high-frequency details, less well-defined edges; High score = more high-frequency details/edges.

    https://arxiv.org/pdf/1903.02695.pdf

    Args:
        image (numpy.ndarray): The input image as a numpy array of dtype float32,
                              with pixel values in the range [0, 1].

    Returns:
        float: The calculated mean Tenengrad focus measure value.
    """
    image = np.array(image, dtype=np.float32) / 255.0

    # Calculate gradient in x and y directions
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate squared gradient magnitude
    teng = gradient_x**2 + gradient_y**2

    # Calculate the mean TENG value
    mean_teng = np.mean(teng)

    return mean_teng

def calculate_gmsd(image_list):
    """
    Calculate the Gradient Magnitude Structural Dissimilarity (GMSD) scores for a list of images.

    The GMSD score measures the structural dissimilarity between pairs of images based on gradient magnitudes.

    Low score = higher similarity of the structural features between two images; High score = less similarity.

    https://arxiv.org/ftp/arxiv/papers/1308/1308.3052.pdf

    Args:
        image_list (list): A list of input images, each represented as a NumPy array.

    Returns:
        numpy.ndarray: A 2D array of GMSD scores for all pairs of images in the input list.
    """
    num_images = len(image_list)
    gmsd_scores = np.zeros((num_images, num_images))

    for i in range(num_images):
        for j in range(i + 1, num_images):
            image1 = np.array(image_list[i], dtype=np.float32) / 255.0
            image2 = np.array(image_list[j], dtype=np.float32) / 255.0

            # Resize images to a common size
            common_size = (min(image1.shape[0], image2.shape[0]), min(image1.shape[1], image2.shape[1]))
            image1_resized = cv2.resize(image1, common_size)
            image2_resized = cv2.resize(image2, common_size)

            gradient_x1 = cv2.Sobel(image1_resized, cv2.CV_64F, 1, 0, ksize=3)
            gradient_y1 = cv2.Sobel(image1_resized, cv2.CV_64F, 0, 1, ksize=3)
            gradient_x2 = cv2.Sobel(image2_resized, cv2.CV_64F, 1, 0, ksize=3)
            gradient_y2 = cv2.Sobel(image2_resized, cv2.CV_64F, 0, 1, ksize=3)

            gradient_mag1 = np.sqrt(gradient_x1**2 + gradient_y1**2)
            gradient_mag2 = np.sqrt(gradient_x2**2 + gradient_y2**2)

            mse = np.mean((gradient_mag1 - gradient_mag2)**2)
            gmsd_score = 1.0 / (1.0 + mse)

            gmsd_scores[i, j] = gmsd_score
            gmsd_scores[j, i] = gmsd_score

    return gmsd_scores

def normalize_metric(scores):
    """
    Normalize metric scores using Min-Max Scaling.

    Args:
        scores (list): List of metric scores for each model.

    Returns:
        list: Normalized metric scores for each model.
    """
    min_score = np.min(scores)
    max_score = np.max(scores)
    normalized_scores = [(score - min_score) / (max_score - min_score) for score in scores]
    return normalized_scores

def aggregate_scores(metrics, model_names, scale_factor=100):
    """
    Aggregate metric scores for each model using weighted averaging.

    Args:
        metrics (tuple): Tuple of lists containing metric scores (CLIP, NIQE, BRISQUE, TENG) for each model.
        model_names (list): List of model names.

    Returns:
        list: A list of aggregate scores for each model.
    """
    # Unpack the metrics tuple
    CLIP_scores, NIQE_scores, BRISQUE_scores, TENG_scores = metrics

    # Define metric weights (adjust these based on your preferences)
    metric_weights = [0.8, -0.01, -0.04, -0.15]  # negative weights = inverse relationship

    # Normalize each metric
    CLIP_normalized = normalize_metric(CLIP_scores)
    NIQE_normalized = normalize_metric(NIQE_scores)
    BRISQUE_normalized = normalize_metric(BRISQUE_scores)
    TENG_normalized = normalize_metric(TENG_scores)

    # Initialize lists for storing weighted scores
    weighted_scores = [[] for _ in range(len(model_names))]

    # Calculate and store weighted scores for each model and metric
    for i in range(len(model_names)):
        for j in range(len(CLIP_normalized[0])):
            weighted_score = (
                CLIP_normalized[i][j] * metric_weights[0] +
                NIQE_normalized[i][j] * metric_weights[1] +
                BRISQUE_normalized[i][j] * metric_weights[2] +
                TENG_normalized[i][j] * metric_weights[3]
            )
            weighted_scores[i].append(weighted_score)

    # Calculate aggregated scores for each model
    aggregated_scores = [np.mean(scores) * scale_factor for scores in weighted_scores]

    # Print aggregated scores
    for i, model in enumerate(model_names):
        print(f"Aggregate score for {model}: {aggregated_scores[i]}")

    # Visualize with a bar plot
    # Create a bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, aggregated_scores)
    plt.xlabel('Model')
    plt.ylabel('Aggregate Score')
    plt.title('Aggregate Scores for Different Models')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('figures/aggregate_metric.png')
    plt.show()

    return aggregated_scores
