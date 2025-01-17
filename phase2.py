# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Step 1: Load the preprocessed dataset
# Replace 'preprocessed_data.csv' with the path to your preprocessed dataset
data = pd.read_csv('preprocessed_data.csv')

# Step 2: Define the features and target variable
# Features: Day of Week, Lag 1 Sale
# Target: Sale
X = data[['Day of Week', 'Lag 1 Sale']]  # Feature matrix
y = data['Sale']  # Target variable

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Initialize and train the model
# Using Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Step 7: Save the trained model (optional)
import joblib
joblib.dump(model, 'sales_forecasting_model.pkl')
print("Trained model saved to 'sales_forecasting_model.pkl'")
