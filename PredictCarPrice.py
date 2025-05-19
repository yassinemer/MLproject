# app.py

import streamlit as st
import pandas as pd
import pickle
import os

# Load models and encoders
model_dir = "models"

def load_pickle(file_name):
    with open(os.path.join(model_dir, file_name), "rb") as f:
        return pickle.load(f)

models = {
    "Linear Regression": load_pickle("Linear Regression.pkl"),
    "Random Forest": load_pickle("Random Forest.pkl"),
    "XGBoost": load_pickle("XGBoost.pkl"),
    "LightGBM": load_pickle("LightGBM.pkl"),
}
label_encoders = load_pickle("label_encoders.pkl")
feature_columns = load_pickle("feature_columns.pkl")

# Streamlit UI
st.title("Car Price Prediction")

# Model selection
selected_model_name = st.sidebar.selectbox("Choose Model", list(models.keys()))
model = models[selected_model_name]

# Inputs
brand = st.sidebar.selectbox("Brand", label_encoders["Brand"].classes_)
model_name = st.sidebar.selectbox("Model", label_encoders["Model"].classes_)
year = st.sidebar.number_input("Year", 2000, 2023, 2010)
engine_size = st.sidebar.number_input("Engine Size", 1.0, 5.0, 2.0)
fuel_type = st.sidebar.selectbox("Fuel Type", label_encoders["Fuel_Type"].classes_)
transmission = st.sidebar.selectbox("Transmission", label_encoders["Transmission"].classes_)
mileage = st.sidebar.number_input("Mileage", 0, 300000, 50000)
doors = st.sidebar.number_input("Doors", 2, 5, 4)
owner_count = st.sidebar.number_input("Owner Count", 1, 5, 2)

# Transform input
input_dict = {
    "Brand": label_encoders["Brand"].transform([brand])[0],
    "Model": label_encoders["Model"].transform([model_name])[0],
    "Year": year,
    "Engine_Size": engine_size,
    "Fuel_Type": label_encoders["Fuel_Type"].transform([fuel_type])[0],
    "Transmission": label_encoders["Transmission"].transform([transmission])[0],
    "Mileage": mileage,
    "Doors": doors,
    "Owner_Count": owner_count,
}
input_df = pd.DataFrame([input_dict])

# Reorder columns to match training
try:
    input_df = input_df[feature_columns]
except KeyError as e:
    st.error(f"Column mismatch! {e}")

# Predict button
if st.sidebar.button("Predict Price"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Price: ${prediction:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
