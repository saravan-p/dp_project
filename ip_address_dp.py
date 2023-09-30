import pandas as pd
import matplotlib.pyplot as plt

def truncate_ipv4_address(ip_address, mask_bits=24):
    ip_parts = ip_address.split('.')
    masked_ip_parts = ip_parts[:mask_bits // 8]
    return '.'.join(masked_ip_parts + ['0'] * (4 - len(masked_ip_parts)))

def utility_metric(original_ips, truncated_ips):
    same_ips = sum(1 for o, t in zip(original_ips, truncated_ips) if o == t)
    return same_ips / len(original_ips)

# Load the dataset
data = pd.read_csv('synthetic_dataset.csv')
ip_addresses = data['IP_Address']

# Define the range of mask bits to analyze
mask_bits_values = list(range(0, 33))

# Calculate the utility metric for each mask bit value
utility_metrics = []

for mask_bits in mask_bits_values:
    truncated_ips = [truncate_ipv4_address(ip, mask_bits) for ip in ip_addresses]
    utility = utility_metric(ip_addresses, truncated_ips)
    utility_metrics.append(utility)

# Truncate the IP addresses using the desired mask bits value
mask_bits = 24
truncated_ips = [truncate_ipv4_address(ip, mask_bits) for ip in ip_addresses]

# Save the truncated IP addresses to a new file
data['Truncated_IP_Address'] = truncated_ips
data.to_csv('synthetic_dataset_truncated.csv', index=False)


# Plot the utility vs privacy graph
plt.plot(mask_bits_values, utility_metrics)
plt.xlabel('Privacy (Mask bits)')
plt.ylabel('Utility (Proportion of unchanged IP addresses)')
plt.title('Utility vs Privacy for IP Addresses in synthetic_dataset.csv')
plt.grid(True)
plt.show()
