import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def laplace_mechanism(value, epsilon, sensitivity):
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale)
    return value + noise

def count_keywords(texts, keywords):
    counter = Counter()
    for text in texts:
        words = text.lower().split()
        for word in words:
            if word in keywords:
                counter[word] += 1
    return counter

def utility_metric(original_counts, noisy_counts):
    return np.mean(np.abs(np.array(list(original_counts.values())) - np.array(list(noisy_counts.values()))))

# Load the dataset
data = pd.read_csv('synthetic_dataset.csv')
data.SentimentText=data.SentimentText.astype(str)
communication_data = data['Communication_Data']

# Define keywords to count
keywords = {'good', 'luck', 'game', 'attack', 'defend', 'objective','going', 'in-game', 'items', 'stay', 'focused', 'wish', 'would', 'stop', 'putting', 'ads', 'in-game', 'purchases', 'everywhere',
            'stick', 'wait', 'virtual', 'currency','charging', 'god'
            }

# Count occurrences of keywords
original_counts = count_keywords(communication_data, keywords)

# Define privacy parameters
epsilon_values = np.logspace(-3, 0, 50)  # 50 points between 0.001 and 1
sensitivity = 1

# Calculate the utility metric for each epsilon value
utility_metrics = []

for epsilon in epsilon_values:
    noisy_counts = {word: laplace_mechanism(count, epsilon, sensitivity) for word, count in original_counts.items()}
    mae = utility_metric(original_counts, noisy_counts)
    utility_metrics.append(mae)

# Plot the utility vs privacy graph
plt.plot(epsilon_values, utility_metrics)
plt.xscale('log')
plt.xlabel('Privacy (epsilon)')
plt.ylabel('Utility (Mean Absolute Error)')
plt.title('Utility vs Privacy for Keyword Counts in Communication_Data')
plt.grid(True)
plt.show()
