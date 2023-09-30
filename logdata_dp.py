
import pandas as pd
import numpy as np
from math import exp
from datetime import datetime

# Load the dataset
data = pd.read_csv('synthetic_dataset.csv')

# Define a function to parse log data
def parse_log_data(log_string):
    log_parts = log_string.split(';')
    login_data = log_parts[0].split(': ')
    logout_data = log_parts[1].split(': ')

    login_event = login_data[0].strip()
    login_timestamp = datetime.strptime(login_data[1].strip(), '%Y-%m-%d %H:%M')

    logout_event = logout_data[0].strip()
    logout_timestamp = datetime.strptime(logout_data[1].strip(), '%Y-%m-%d %H:%M')

    return login_event, login_timestamp, logout_event, logout_timestamp

# Parse the "Log_Data" column
parsed_log_data = data['Log_Data'].apply(parse_log_data)

# Separate the events and timestamps
login_events, login_timestamps, logout_events, logout_timestamps = zip(*parsed_log_data)

# Set the privacy budget (epsilon)
epsilon = 0.1  # Adjust this value to your desired privacy level.

# Apply the Laplace mechanism for numerical data (timestamps)
def laplace_mechanism(value, epsilon, sensitivity=1):
    scale = sensitivity / epsilon
    return value + np.random.laplace(0, scale)

# Convert timestamps to numerical data (e.g., POSIX timestamps)
login_timestamps_numerical = [timestamp.timestamp() for timestamp in login_timestamps]
logout_timestamps_numerical = [timestamp.timestamp() for timestamp in logout_timestamps]

# Apply the Laplace mechanism to the numerical timestamps
noisy_login_timestamps = [laplace_mechanism(ts, epsilon) for ts in login_timestamps_numerical]
noisy_logout_timestamps = [laplace_mechanism(ts, epsilon) for ts in logout_timestamps_numerical]

# Convert noisy numerical timestamps back to datetime objects
noisy_login_timestamps = [datetime.fromtimestamp(ts) for ts in noisy_login_timestamps]
noisy_logout_timestamps = [datetime.fromtimestamp(ts) for ts in noisy_logout_timestamps]

# Define the utility function (score function) for categorical data
def utility_function_categorical(data, output):
    return data.value_counts()[output]

# Apply the exponential mechanism to the categorical data
def exponential_mechanism_categorical(data, utility_function, epsilon, sensitivity):
    unique_values = data.unique()
    probabilities = []

    for value in unique_values:
        score = utility_function(data, value)
        probability = exp(epsilon * score / (2 * sensitivity))
        probabilities.append(probability)

    probabilities /= np.sum(probabilities)  # Normalize probabilities
    return np.random.choice(unique_values, p=probabilities)

event_sensitivity = 1

noisy_login_events = [exponential_mechanism_categorical(pd.Series(login_events), utility_function_categorical, epsilon, event_sensitivity) for _ in range(len(data))]
noisy_logout_events = [exponential_mechanism_categorical(pd.Series(logout_events), utility_function_categorical, epsilon, event_sensitivity) for _ in range(len(data))]

# Combine the noisy events and timestamps
noisy_log_data = [f"{login_event}: {login_timestamp}; {logout_event}: {logout_timestamp}"
                  for login_event, login_timestamp, logout_event, logout_timestamp in
                  zip(noisy_login_events, noisy_login_timestamps, noisy_logout_events, noisy_logout_timestamps)]

# Create a new DataFrame with the noisy "Log_Data" column
noisy_data = data.copy()
noisy_data['Log_Data'] =noisy_log_data

# Save the new DataFrame to a CSV file
noisy_data.to_csv('synthetic_data_noisy1.csv', index=False)

import matplotlib.pyplot as plt

def utility_metric(original_timestamps, noisy_timestamps):
    differences = [abs((ot - nt).total_seconds()) for ot, nt in zip(original_timestamps, noisy_timestamps)]
    return np.mean(differences)

# Generate a range of epsilon values to analyze
epsilon_values = np.logspace(-3, 0, 50)  # 50 points between 0.001 and 1

# Calculate the utility metric for each epsilon value
utility_metrics = []

for epsilon in epsilon_values:
    noisy_login_timestamps = [laplace_mechanism(ts, epsilon) for ts in login_timestamps_numerical]
    noisy_logout_timestamps = [laplace_mechanism(ts, epsilon) for ts in logout_timestamps_numerical]
    
    noisy_login_timestamps = [datetime.fromtimestamp(ts) for ts in noisy_login_timestamps]
    noisy_logout_timestamps = [datetime.fromtimestamp(ts) for ts in noisy_logout_timestamps]

    login_utility = utility_metric(login_timestamps, noisy_login_timestamps)
    logout_utility = utility_metric(logout_timestamps, noisy_logout_timestamps)
    average_utility = (login_utility + logout_utility) / 2
    
    utility_metrics.append(average_utility)

# Plot the utility vs privacy graph
plt.plot(epsilon_values, utility_metrics)
plt.xscale('log')
plt.xlabel('Privacy (epsilon)')
plt.ylabel('Utility (average absolute difference in seconds)')
plt.title('Utility vs Privacy for Log Data')
plt.grid(True)
plt.show()
