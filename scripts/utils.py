import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def generate_stripplot(base, refined, one_five, two_one):
    data = np.concatenate((base, refined, one_five, two_one))
    labels = ['SDXL 1.0 base'] * len(base) + ['SDXL 1.0 base+refiner'] * len(refined) + ['SD 1.5'] * len(one_five) + ['SD 2.1'] * len(two_one)

    plt.figure(figsize=(10, 6))

    sns.stripplot(x=labels, y=data, jitter=True, dodge=True, palette='Set2')

    plt.xlabel('Model')
    plt.ylabel('CLIP Score')
    plt.title('Strip Plot of CLIP Scores')

    plt.tight_layout()

    plt.show()

def generate_violinplot(base, refined, one_five, two_one):
    data = [base, refined, one_five, two_one]
    labels = ["SDXL 1.0 base", "SDXL 1.0 base+refiner", "SD 1.5", "SD 2.1"]

    plt.figure(figsize=(10, 6))

    sns.violinplot(data=data, inner="quartile")
    plt.xticks(range(len(data)), labels)

    plt.xlabel('Model')
    plt.ylabel('CLIP Score')
    plt.title('Violin Plot of CLIP Scores')

    plt.tight_layout()

    plt.show()