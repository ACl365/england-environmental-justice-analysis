import pandas as pd
import numpy as np

# Load the data
try:
    df = pd.read_csv(r"C:\\Users\\alex5\\Downloads\\churn-prediction-data-space-publicleaderboard-2025-02-26T17_58_48\\churn-prediction-data-space-publicleaderboard-2025-02-26T17_58_48.csv")
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: CSV file not found. Please ensure the path is correct.")
    exit()

# Display basic information about the data
print("\\nData shape:", df.shape)
print("\\nData types:\\n", df.dtypes)
print("\\nMissing values:\\n", df.isnull().sum())
print("\\nDescriptive statistics:\\n", df.describe())

# Print the first 5 rows
print("\\nFirst 5 rows:\\n", df.head())

# Handle missing values (example: fill with mean)
for col in df.columns:
    if df[col].isnull().any():
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

print("\\nMissing values after imputation:\\n", df.isnull().sum())

# Example: Encode categorical variables (example: one-hot encoding)
categorical_cols = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, columns=categorical_cols, dummy_na=False)

print("\\nData shape after preprocessing:", df.shape)
print("\\nFirst 5 rows after preprocessing:\\n", df.head())

# Example: Feature Scaling (MinMaxScaler)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
numerical_cols = df.select_dtypes(include=['number']).columns
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

print("\\nFirst 5 rows after scaling:\\n", df.head())

# Save the preprocessed data (optional)
df.to_csv("preprocessed_data.csv", index=False)

print("\\nPreprocessing complete. Preprocessed data saved to preprocessed_data.csv")

# Model Selection, Training, and Evaluation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Prepare the data
X = df.drop("TeamName", axis=1)  # Drop the target variable
y = df["Rank"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Select a model
model = LogisticRegression(solver='liblinear', multi_class='ovr')

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("\\nAccuracy:", accuracy)
print("\\nClassification Report:\\n", classification_report(y_test, y_pred, zero_division=0))