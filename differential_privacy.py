import pandas as pd
import numpy as np

def apply_laplace_noise(data, epsilon):
    """
    Apply Laplace noise to the given data.
    :param data: The data to which the noise will be added.
    :param epsilon: The privacy parameter (higher values provide less privacy).
    :return: The data with added Laplace noise.
    """
    scale = 1 / epsilon
    noise = np.random.laplace(loc=0, scale=scale, size=data.shape)
    return data + noise

# Load the synthetic dataset (replace this with your own dataset file)
data = pd.read_csv("synthetic_dataset.csv")

# Set the privacy parameter epsilon (choose a value between 0.1 and 1)
epsilon = 0.5

# Apply Laplace noise to the Gameplay_Info column
data["Gameplay_Info_Noisy"] = apply_laplace_noise(data["Gameplay_Info"], epsilon)

# Save the differentially private dataset to a new CSV file
data.to_csv("synthetic_dataset_differentially_private.csv", index=False)
