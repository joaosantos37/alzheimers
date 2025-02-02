import streamlit as st
import joblib
import pandas as pd

# Load the model
model = joblib.load('best_rf_model_top.pkl')

# Define the input features
st.title("Alzheimer's Prediction App")

# Add description
st.write("This model uses the 5 most important features for prediction:")
st.write("- Age (59.31% importance)")
st.write("- Genetic Risk Factor (10.11% importance)")
st.write("- Family History (5.63% importance)")
st.write("- BMI (3.64% importance)")
st.write("- Cognitive Test Score (3.14% importance)")

# Create input fields with exact feature names, ordered by importance
features = {
    'Age': st.number_input("Age (Most Important Feature)", min_value=0, max_value=120, value=30),
    'Genetic Risk Factor (APOE-ε4 allele)': st.selectbox("Genetic Risk Factor (APOE-ε4 allele)", [0, 1]),
    'Family History of Alzheimer\'s': st.selectbox("Family History of Alzheimer's", [0, 1]),
    'BMI': st.number_input("BMI", min_value=0.0, max_value=50.0, value=22.5),
    'Cognitive Test Score': st.number_input("Cognitive Test Score", min_value=0, max_value=30, value=28)
}

# Create DataFrame with exact column names
input_data = pd.DataFrame([features])

# Make prediction
if st.button("Predict"):
    try:
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            st.error("Prediction: Higher Risk of Alzheimer's")
        else:
            st.success("Prediction: Lower Risk of Alzheimer's")
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")