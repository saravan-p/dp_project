import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load dataset
data = pd.read_csv("synthetic_dataset.csv")

# Prepare the features and target variables
X = data[["Location", "Device_Info", "IP_Address"]]
y = data["Gameplay_Info"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the preprocessing steps for the categorical and numerical columns
preprocessor = ColumnTransformer(
    transformers=[
        ("location", OneHotEncoder(), ["Location"]),
        ("device_info", OneHotEncoder(), ["Device_Info"]),
        ("ip_address", OneHotEncoder(), ["IP_Address"]),
    ],
    remainder="drop",
)

# Create a pipeline with the preprocessor and a RandomForest classifier
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42)),
    ]
)

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the model's performance
print(classification_report(y_test, y_pred))
