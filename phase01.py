# Import necessary libraries
import pandas as pd

# Step 1: Load the dataset
# Replace 'path_to_your_file.csv' with the actual path to your dataset
data = pd.read_csv("C:\\Users\\yuvra\\Downloads\\Data.csv")

# Step 2: Explore the dataset
print("First few rows of the dataset:")
print(data.head())

print("\nDataset information:")
print(data.info())

print("\nDescriptive statistics:")
print(data.describe())

# Step 3: Handle missing values
# Using forward fill for missing values. Adjust based on your dataset's needs.
data = data.fillna(method='ffill')  # Forward fill missing values

# Step 4: Convert the "Date" column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Step 5: Extract new features from the "Date" column
data['Day of Week'] = data['Date'].dt.dayofweek  # Day of the week (0 = Monday, 6 = Sunday)

# Step 6: Filter data for "Farmhouse Pizza" (if applicable)
# Uncomment the next line if you need to filter for a specific product
# data = data[data['Product Name'] == 'Farmhouse Pizza']

# Step 7: Create a lag feature (sales from the previous day)
data['Lag 1 Sale'] = data['Sale'].shift(1)

# Step 8: Drop rows with missing values caused by lagging
data.dropna(inplace=True)

# Step 9: Review the processed data
print("\nData after preprocessing:")
print(data.head())

# Step 10: Save the preprocessed data to a new CSV file (optional)
# This can be useful for debugging or future use
data.to_csv('preprocessed_data.csv', index=False)
print("\nPreprocessed data saved to 'preprocessed_data.csv'")
