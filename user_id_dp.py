import pandas as pd
import numpy as np
from math import exp

# Load the dataset
data = pd.read_csv('synthetic_dataset.csv')

# Define the utility function (score function)
def utility_function(data, output):
    return data['User_ID'].value_counts()[output]

# Define the sensitivity of the utility function
sensitivity = 1

# Set the privacy budget (epsilon)
epsilon = 0.1  # Adjust this value to your desired privacy level.

# Apply the exponential mechanism to the "User_ID" column
def exponential_mechanism(data, utility_function, epsilon, sensitivity):
    unique_user_ids = data['User_ID'].unique()
    probabilities = []

    for user_id in unique_user_ids:
        score = utility_function(data, user_id)
        probability = exp(epsilon * score / (2 * sensitivity))
        probabilities.append(probability)

    probabilities /= np.sum(probabilities)  # Normalize probabilities
    return np.random.choice(unique_user_ids, p=probabilities)

# Apply the exponential mechanism to the entire "User_ID" column
noisy_user_ids = [exponential_mechanism(data, utility_function, epsilon, sensitivity) for _ in range(len(data))]

# Create a new DataFrame with the noisy "User_ID" column
noisy_data = data.copy()
noisy_data['User_ID'] = noisy_user_ids

# Save the new DataFrame to a CSV file
noisy_data.to_csv('synthetic_dataset_noisy.csv', index=False)
