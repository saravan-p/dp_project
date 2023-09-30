#To create an automated iterative process for finding the maximum level of privacy protection 
#while maintaining an acceptable level of utility,
#you can use a grid search method. This method will test various epsilon values 
#and scoring functions to find the optimal balance between privacy and utility.

#In this example, we will use the Laplace mechanism for the Gameplay_Info numerical column and 
#the exponential mechanism for the Location categorical column. 
#We'll use the Mean Squared Error (MSE) and the F1-score as utility metrics for the numerical and categorical columns, respectively.

#you can install the required libraries using pip:

# pip install numpy pandas scikit-learn

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, f1_score
from scipy.special import expit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Define utility functions
def add_laplace_noise(data, epsilon):
    scale = 1 / epsilon
    noise = np.random.laplace(loc=0, scale=scale, size=data.shape)
    return data + noise

def apply_exponential_mechanism(data, epsilon):
    def scoring_function(x):
        return expit(x)

    probabilities = scoring_function(data)
    probabilities /= probabilities.sum()
    return np.random.choice(data, p=probabilities, replace=False, size=len(data))

# Load the synthetic dataset (replace this with your own dataset file)
data = pd.read_csv("synthetic_dataset.csv")

# Define the grid search parameters
epsilon_values = np.linspace(0.1, 1.0, 10)
scoring_functions = [expit]  # Add more scoring functions if desired
best_epsilon = None
best_scoring_function = None
best_utility = float("inf")

# Split the dataset into train and test sets
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

for epsilon in epsilon_values:
    for scoring_function in scoring_functions:
        # Apply Laplace mechanism to the numerical columns
        train_data_noisy = train_data.copy()
        train_data_noisy["Gameplay_Info_Noisy"] = add_laplace_noise(train_data["Gameplay_Info"], epsilon)

        # Calculate the utility for the numerical columns (using MSE)
        mse = mean_squared_error(train_data["Gameplay_Info"], train_data_noisy["Gameplay_Info_Noisy"])

        # Apply the exponential mechanism to the categorical columns
        train_data_noisy["Location_Noisy"] = apply_exponential_mechanism(train_data["Location"], epsilon)

        # Calculate the utility for the categorical columns (using F1-score)
        le = LabelEncoder()
        true_labels = le.fit_transform(train_data["Location"])
        noisy_labels = le.transform(train_data_noisy["Location_Noisy"])
        f1 = f1_score(true_labels, noisy_labels, average='weighted')

        # Combine the utility metrics for the numerical and categorical columns
        combined_utility = mse - f1

        # Update the best privacy settings if the utility is improved
        if combined_utility < best_utility:
            best_epsilon = epsilon
            best_scoring_function = scoring_function
            best_utility = combined_utility

print(f"Best privacy settings: epsilon = {best_epsilon}, scoring function = {best_scoring_function.__name__}")

# Apply the best privacy settings to the test dataset
test_data_noisy = test_data.copy()
test_data_noisy["Gameplay_Info_Noisy"] = add_laplace_noise(test_data["Gameplay_Info"], best_epsilon)
test_data_noisy["Location_Noisy"] = apply_exponential_mechanism(test_data["Location"],best_epsilon)

# Validate the model's performance using the test dataset
mse_test = mean_squared_error(test_data["Gameplay_Info"], test_data_noisy["Gameplay_Info_Noisy"])
true_labels_test = le.transform(test_data["Location"])
noisy_labels_test = le.transform(test_data_noisy["Location_Noisy"])
f1_test = f1_score(true_labels_test, noisy_labels_test, average='weighted')

print(f"Validation results (test set):")
print(f"MSE: {mse_test}")
print(f"F1-score: {f1_test}")

# Save the differentially private dataset to a new CSV file
test_data_noisy.to_csv("synthetic_dataset_differentially_private_optimized.csv", index=False)


#This code iterates over a range of epsilon values and scoring functions, 
# applying them to the training dataset, and calculating the utility using the Mean Squared Error (MSE) and F1-score. 
# The best privacy settings are those that minimize the combined utility metric.

#After finding the best privacy settings, 
# the code applies these settings to the test dataset and validates the model's performance using the MSE and F1-score. 
# Finally, the differentially private dataset with optimized privacy settings is saved to a new CSV file.

#Please note that the code provided here is a basic example. 
# You may need to customize the utility functions, scoring functions, 
# and grid search parameters to fit your specific needs and dataset.
