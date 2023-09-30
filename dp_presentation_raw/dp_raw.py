import pandas as pd
import numpy as np
from scipy.special import expit

def add_laplace_noise(data, epsilon):
    scale = 1 / epsilon
    noise = np.random.laplace(loc=0, scale=scale, size=data.shape)
    return data + noise

def apply_exponential_mechanism(data, epsilon):
    # Define a simple scoring function for this example
    def scoring_function(x):
        return expit(x)

    # Create a probability distribution based on the scoring function
    probabilities = scoring_function(data)
    probabilities /= probabilities.sum()

    # Apply the exponential mechanism
    return np.random.choice(data, p=probabilities, replace=False, size=len(data))

def anonymize_ip(ip):
    # Convert IP address to a less precise form (e.g., only use the first two octets)
    ip_parts = ip.split(".")
    return f"{ip_parts[0]}.{ip_parts[1]}.*.*"

# Load the synthetic dataset (replace this with your own dataset file)
data = pd.read_csv("synthetic_dataset.csv")

# Set the privacy parameter epsilon (choose a value between 0.1 and 1)
epsilon = 0.5

# Apply Laplace noise to the numerical columns
data["Gameplay_Info_Noisy"] = add_laplace_noise(data["Gameplay_Info"], epsilon)
data["Usage_Data_Noisy"] = add_laplace_noise(data["Usage_Data"].str.extract(r'(\d+)').astype(float), epsilon)

# Apply the exponential mechanism to categorical columns
data["Location_Noisy"] = apply_exponential_mechanism(data["Location"], epsilon)
data["Device_Info_Noisy"] = apply_exponential_mechanism(data["Device_Info"], epsilon)
data["Cookie_ID_Noisy"] = apply_exponential_mechanism(data["Cookie_ID"], epsilon)

# Anonymize the IP address
data["IP_Address_Noisy"] = data["IP_Address"].apply(anonymize_ip)

# Save the differentially private dataset to a new CSV file
data.to_csv("synthetic_dataset_differentially_private.csv", index=False)

#This script applies Laplace noise to the Gameplay_Info and Usage_Data columns, 
# the exponential mechanism to the Location, Device_Info, and Cookie_ID columns, 
# and anonymizes the IP address using a generalization technique.
#Keep in mind that this script is an example to demonstrate the application of various differential privacy techniques 
# to a synthetic dataset. You may need to adjust the privacy parameters, scoring functions, 
# and methods according to your specific privacy and utility requirements.