import streamlit as st
import pandas as pd
import joblib

# Load model & preprocessor
model = joblib.load("best_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

st.set_page_config(page_title="House Price Prediction", page_icon="🏠")
st.title("🏠 House Price Prediction App")

# Inputs
area = st.number_input("Area (sq ft)", 300, 10000)
bedrooms = st.number_input("Bedrooms", 1, 10)
bathrooms = st.number_input("Bathrooms", 1, 10)
floors = st.number_input("Floors", 1, 5)
year_built = st.number_input("Year Built", 1900, 2025)

location = st.selectbox("Location", ["Urban", "Suburban", "Rural"])
condition = st.selectbox("Condition", ["Poor", "Average", "Good"])
garage = st.selectbox("Garage", ["Yes", "No"])

if st.button("Predict Price"):
    input_df = pd.DataFrame([{
        "Area": area,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Floors": floors,
        "YearBuilt": year_built,
        "Location": location,
        "Condition": condition,
        "Garage": garage
    }])

    transformed_input = preprocessor.transform(input_df)
    prediction = model.predict(transformed_input)

    st.success(f"💰 Estimated Price: ₹ {prediction[0]:,.2f}")
    