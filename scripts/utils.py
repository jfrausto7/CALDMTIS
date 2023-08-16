import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def generate_stripplot(scores, metric):
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