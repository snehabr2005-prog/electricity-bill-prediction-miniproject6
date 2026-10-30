import streamlit as st
import numpy as np
import pickle
from tensorflow import keras

st.set_page_config(page_title="Electricity Bill Predictor", page_icon="💡")
st.title("💡 Electricity Bill Prediction - Mini Project 6")
st.write("Using Neural Networks")

# Load files
model = keras.models.load_model('electricity_model.keras')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

units = st.slider("Units Consumed (kWh)", 50, 500, 200)
appliances = st.number_input("Number of Appliances", 1, 10, 3)
month = st.selectbox("Month", list(range(1,13)))

if st.button("Predict Bill"):
    data = np.array([[units, appliances, month]])
    data_scaled = scaler.transform(data)
    bill = model.predict(data_scaled)[0][0]
    st.success(f"Estimated Bill: Rs. {bill:.2f}")