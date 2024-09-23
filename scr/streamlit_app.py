import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the saved model
loaded_xgb_model = joblib.load('./models/xgb_model.pkl')
scaler = joblib.load('./models/scaler.pkl')

st.title("Glioma Classification Model with XGBoost")
st.write("This app predicts cancer types based on gene expression data.")

st.sidebar.header("Input Features")

num_features = len(X_selected.columns) 
input_features = []

for i in range(num_features):
    feature_value = st.sidebar.number_input(f"Feature {i+1}", min_value=0.0, max_value=100.0, value=50.0)
    input_features.append(feature_value)

input_data = np.array(input_features).reshape(1, -1)
input_data_scaled = scaler.transform(input_data)

if st.button("Predict"):
    prediction = loaded_xgb_model.predict(input_data_scaled)
    prediction_proba = loaded_xgb_model.predict_proba(input_data_scaled)
    cancer_type_dict = {0: "Astrocytoma", 1: "Glioblastoma", 2: "Oligodendroglioma"} 
    st.write(f"Predicted Cancer Type: {cancer_type_dict[int(prediction[0])]}")
    st.write(f"Prediction Probabilities: {prediction_proba}")

