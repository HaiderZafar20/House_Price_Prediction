import streamlit as st
import joblib
import numpy as np
import json

# Load model
model = joblib.load("banglore_home_prices_model.pickle")

# Load feature columns
with open("columns.json", "r") as f:
    data = json.load(f)
columns = data["data_columns"]

st.title("🏠 Bangalore House Price Prediction App")
st.write("Enter property details below:")

# Numeric inputs
area = st.number_input("Enter area (sqft):", value=1000.0)
bhk = st.number_input("Enter number of BHK:", min_value=1, step=1)
bath = st.number_input("Enter number of bathrooms:", min_value=1, step=1)

# Extract location names (everything except numeric features)
locations = [col for col in columns if col not in ["total_sqft", "bath", "bhk"]]
location = st.selectbox("Choose Location:", locations)

# Build input vector
x = np.zeros(len(columns))

for i, col in enumerate(columns):
    if col == "total_sqft":
        x[i] = area
    elif col == "bhk":
        x[i] = bhk
    elif col == "bath":
        x[i] = bath
    elif col == location:   # match the chosen location column
        x[i] = 1


# Prediction
if st.button("Predict"):
    prediction = model.predict([x])
    st.success(f"💰 Predicted Price: ₹ {prediction[0]:,.2f} lakh")
