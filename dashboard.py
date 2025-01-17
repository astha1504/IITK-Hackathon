import pandas as pd
import streamlit as st
import joblib
from datetime import datetime, timedelta

# Load the trained model
model = joblib.load("C:\\Users\\yuvra\\sales_forecasting_model.pkl")

# Load the dataset
# Replace 'preprocessed_data.csv' with your actual preprocessed data file
data = pd.read_csv('preprocessed_data.csv')

# Streamlit app configuration
st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")

# Title and Description
st.title("Sales Forecasting Dashboard")
st.markdown("""
This dashboard helps you visualize sales data and predict sales for the next day. 
Use the filters to explore sales data by outlet and product.
""")

# Sidebar Filters
st.sidebar.header("Filters")
outlet_filter = st.sidebar.selectbox("Select Outlet", options=data["Outlet Name"].unique())
product_filter = st.sidebar.selectbox("Select Product", options=data["Product Name"].unique())

# Filter data based on selections
filtered_data = data[(data["Outlet Name"] == outlet_filter) & (data["Product Name"] == product_filter)]

# Display filtered data
st.subheader(f"Sales Data for {product_filter} at {outlet_filter}")
st.write(filtered_data)

# Sales Trend Visualization
st.subheader("Sales Trend Over Time")
filtered_data["Date"] = pd.to_datetime(filtered_data["Date"])
filtered_data = filtered_data.sort_values(by="Date")
st.line_chart(filtered_data.set_index("Date")["Sale"])

# Predict Next Day's Sales
st.subheader("Predict Sales for the Next Day")
# Get the most recent data point
latest_date = filtered_data["Date"].max()
latest_day_of_week = latest_date.weekday()  # Day of the week (0 = Monday, 6 = Sunday)
latest_sale = filtered_data.loc[filtered_data["Date"] == latest_date, "Sale"].values[0]

# Prepare input for the model
next_day = latest_date + timedelta(days=1)
next_day_of_week = next_day.weekday()  # Day of the week for the next day
input_features = pd.DataFrame({
    "Day of Week": [next_day_of_week],
    "Lag 1 Sale": [latest_sale]
})

# Predict and display the result
predicted_sales = model.predict(input_features)[0]
st.write(f"Predicted Sales for {product_filter} on {next_day.strftime('%Y-%m-%d')}: **{predicted_sales:.2f} units**")

# Footer
st.markdown("---")
st.markdown("Developed for the Hackathon 🚀")
