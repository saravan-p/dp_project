import pandas as pd

import numpy as np

def planar_laplace_noise(location, epsilon, sensitivity):
    """
    Adds planar Laplace noise to a location tuple (latitude, longitude).
    """
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale, 2)
    noisy_location = np.add(location, noise)
    return tuple(noisy_location)


data = pd.read_csv('synthetic_dataset.csv')
locations = data['Location']

import ast

def parse_location(location_str):
    location_tuple = ast.literal_eval(location_str)
    return location_tuple[0], location_tuple[1]

epsilon = 1.0
sensitivity = 100  # Set this value based on your data and desired level of privacy

noisy_locations = []

for location_str in locations:
    lat, lon = parse_location(location_str)
    noisy_lat, noisy_lon = planar_laplace_noise((lat, lon), epsilon, sensitivity)
    noisy_locations.append((noisy_lat, noisy_lon))

data['location'] = noisy_locations
data.to_csv('synthetic_dataset_noisy.csv', index=False)


import numpy as np

original_latitudes = [lat for lat, lon in map(parse_location, locations)]
original_longitudes = [lon for lat, lon in map(parse_location, locations)]

# Assuming original_latitudes and original_longitudes are lists with the original data
original_mean_lat = np.mean(original_latitudes)
original_mean_lon = np.mean(original_longitudes)
noisy_mean_lat = np.mean([loc[0] for loc in noisy_locations])
noisy_mean_lon = np.mean([loc[1] for loc in noisy_locations])

print("Original mean latitude:", original_mean_lat)
print("Noisy mean latitude:", noisy_mean_lat)
print("Original mean longitude:", original_mean_lon)
print("Noisy mean longitude:", noisy_mean_lon)

import folium

def create_map(data_points, map_title):
    # Create a map centered at the mean location of the data points
    map_center = (np.mean([point[0] for point in data_points]), np.mean([point[1] for point in data_points]))
    data_map = folium.Map(location=map_center, zoom_start=10, control_scale=True, tiles='cartodb positron')
    
    # Add each data point to the map
    for point in data_points:
        folium.Marker([point[0], point[1]]).add_to(data_map)
    
    # Add a title to the map
    title_html = f"""
                 <h3 align="center" style="font-size:20px">
                 <b>{map_title}</b></h3>
                 """
    data_map.get_root().html.add_child(folium.Element(title_html))

    return data_map

original_locations = list(zip(original_latitudes, original_longitudes))
original_map = create_map(original_locations, "Original Locations")
noisy_map = create_map(noisy_locations, "Noisy Locations with Differential Privacy")

original_map.save("original_map.html")
noisy_map.save("noisy_map.html")

from scipy.stats import pearsonr

# Assuming ref_point is the reference point (e.g., city center)
original_distances = [great_circle(ref_point, loc).meters for loc in original_locations]
noisy_distances = [great_circle(ref_point, loc).meters for loc in noisy_locations]

correlation, _ = pearsonr(original_distances, noisy_distances)
print("Correlation between original and noisy distances:", correlation)
