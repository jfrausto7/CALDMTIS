import cv2
import torch
import numpy as np
import imquality.brisque as brisque

def calculate_clip_score(image, prompt, clip_score_fn):
    clip_score = clip_score_fn(
        torch.from_numpy(image).unsqueeze(0).permute(0, 3, 1, 2), [prompt]
    ).detach()
    return round(float(clip_score), 4)

def calculate_niqe(image):
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
    image = np.array(image, dtype=np.float32) / 255.0
    return brisque.score(image)

def calculate_teng(image):
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
