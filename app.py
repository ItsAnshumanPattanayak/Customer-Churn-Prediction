import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved model and encoders
model = joblib.load('rf_model.pkl')
encoders = joblib.load('encoders.pkl')

st.title("📊 Customer Churn Prediction")
st.write("Enter the customer's data below to predict whether they will cancel their service.")

# Create UI columns
col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges ($)", 18.0, 120.0, 50.0)
    total_charges = st.number_input("Total Charges ($)", 18.0, 8000.0, 500.0)

with col2:
    contract = st.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
    internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])

if st.button("Predict Churn"):
    # Encode categorical inputs using the saved label encoders
    contract_encoded = encoders['Contract'].transform([contract])[0]
    internet_encoded = encoders['InternetService'].transform([internet_service])[0]
    
    # Bundle into an array matching the training feature order
    features = np.array([[tenure, monthly_charges, total_charges, contract_encoded, internet_encoded]])
    
    # Predict
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    
    st.divider()
    if prediction == 1:
        st.error(f"⚠️ High Risk of Churn! (Probability: {probability:.2%})")
        st.write("Recommendation: Offer a discount or upgrade to a longer-term contract.")
    else:
        st.success(f"✅ Customer is likely to stay. (Churn Probability: {probability:.2%})")