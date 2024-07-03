import streamlit as st
import pandas as pd
import joblib

st.title("Predicción con Modelo de Machine Learning")

model = joblib.load("path_to_your_model.pkl")

feature1 = st.number_input("Feature 1")
feature2 = st.number_input("Feature 2")
feature3 = st.number_input("Feature 3")

if st.button("Predecir"):
    input_data = pd.DataFrame([[feature1, feature2, feature3]])
    prediction = model.predict(input_data)
    st.write(f"La predicción es: {prediction[0]}")
