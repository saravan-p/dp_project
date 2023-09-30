import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from geopy.distance import great_circle
from ast import literal_eval

def clip_latitude(lat):
    return max(min(lat, 90), -90)


def planar_laplace_noise(location, epsilon, sensitivity):
    """
    Adds planar Laplace noise to a location tuple (latitude, longitude).
    """
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale, 2)
    noisy_location = np.add(location, noise)
    
    # Clip the latitude value to the valid range
    noisy_lat = clip_latitude(noisy_location[0])
    noisy_lon = noisy_location[1]
    
    return (noisy_lat, noisy_lon)


def utility_metric(original_locations, noisy_locations):
    errors = [great_circle(ol, nl).meters for ol, nl in zip(original_locations, noisy_locations)]
    return np.mean(errors)

def parse_location(location_str):
    location_tuple = literal_eval(location_str)
    return location_tuple[0], location_tuple[1]

# Load the dataset
data = pd.read_csv('synthetic_dataset.csv')
locations = data['Location']
original_locations = list(map(parse_location, locations))

# Define privacy parameters
epsilon_values = np.logspace(-3, 0, 50)  # 50 points between 0.001 and 1
sensitivity = 100  # Set this value based on your data and desired level of privacy

# Calculate the utility metric for each epsilon value
utility_metrics = []

for epsilon in epsilon_values:
    noisy_locations = [planar_laplace_noise(loc, epsilon, sensitivity) for loc in original_locations]
    average_error = utility_metric(original_locations, noisy_locations)
    utility_metrics.append(average_error)

# Plot the utility vs privacy graph
plt.plot(epsilon_values, utility_metrics)
plt.xscale('log')
plt.xlabel('Privacy (epsilon)')
plt.ylabel('Utility (average distance error in meters)')
plt.title('Utility vs Privacy for Location Data in synthetic_dataset.csv')
plt.grid(True)
plt.show()
