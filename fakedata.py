import pandas as pd
from faker import Faker
import random
import csv

fake = Faker()

import random

# Define the bounding box coordinates (latitude and longitude)
# In this example, the bounding box is roughly around the United States
MIN_LAT = 24.0
MAX_LAT = 49.0
MIN_LONG = -125.0
MAX_LONG = -67.0

def random_location():
    # Generate a random latitude and longitude within the bounding box
    lat = round(random.uniform(MIN_LAT, MAX_LAT), 6)
    long = round(random.uniform(MIN_LONG, MAX_LONG), 6)
    return (lat, long)

def random_device():
    devices = [
        (f"iPhone {random.randint(10, 13)}", f"iOS {random.uniform(14.0, 15.5):.1f}"),
        (f"Samsung Galaxy S{random.randint(20, 23)}", f"Android {random.uniform(11.0, 13.0):.1f}"),
        "Xbox Series X",
        "PS5",
    ]
    return random.choice(devices)

def random_ip():
    return fake.ipv4()

def random_communication():
    with open('communication_data.txt', 'r') as f:
        msgs = f.read().splitlines()
    return random.choice(msgs)

def random_gameplay():
    return random.randint(1, 100)

def random_usage():
    return f"{random.randint(1, 6)} hrs"

def random_cookie():
    return f"C{random.randint(10000, 99999)}"

def random_log_data():
    login_time = fake.date_time_this_month()
    logout_time = login_time + fake.time_delta(end_datetime=None)
    return f"Login: {login_time.strftime('%Y-%m-%d %H:%M')}; Logout: {logout_time.strftime('%Y-%m-%d %H:%M')}"

data = []

for _ in range(10000):
    data.append([
        
        fake.uuid4(),
        random_location(),
        random_device(),
        random_ip(),
        random_communication(),
        random_gameplay(),
        random_usage(),
        random_cookie(),
        random_log_data(),
    ])

header = ["User_ID", "Location", "Device_Info", "IP_Address", "Communication_Data", "Gameplay_Info", "Usage_Data", "Cookie_ID", "Log_Data"]

with open("synthetic_dataset.csv", "w", newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(header)
    csvwriter.writerows(data)


