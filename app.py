
# 13. Streamlit Application

import streamlit as st
import pandas as pd
import pickle

# Configure the application
st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="centered"
)

# Load the trained model
with open("random_forest_model.pkl", "rb") as file:
    model = pickle.load(file)

# Project Introduction
st.title("🏦 Bank Customer Churn Prediction")
st.caption("Machine Learning Project | Developed by Zoleikha Mousavi Pak")

st.write(
    "A machine learning application that predicts "
    "whether a bank customer is likely to leave."
)

st.markdown("""
**Project Overview**
- Dataset: 10,000 bank customers
- Input Features: 6 customer attributes
- Algorithm: Random Forest Classifier
- Model Accuracy: 83.70%
- Evaluation: Accuracy, Precision, Recall and F1 Score
- Feature Importance Analysis
""")

st.divider()

# Customer Input Form
with st.form("customer_form"):

    st.subheader("Customer Information")

    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    num_products = st.number_input("NumOfProducts", min_value=1, max_value=4, value=1)
    salary = st.number_input("EstimatedSalary", min_value=0.0, value=50000.0)
    balance = st.number_input("Balance", min_value=0.0, value=50000.0)
    credit_score = st.number_input("CreditScore", min_value=300, max_value=850, value=650)
    tenure = st.number_input("Tenure", min_value=0, max_value=10, value=5)

    submitted = st.form_submit_button("Predict")

# Make Prediction
if submitted:

    # Prepare input data
    input_data = pd.DataFrame([{
        "Age": age,
        "NumOfProducts": num_products,
        "EstimatedSalary": salary,
        "Balance": balance,
        "CreditScore": credit_score,
        "Tenure": tenure
    }])

    # Predict customer churn
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    # Calculate probabilities
    churn_probability = probabilities[list(model.classes_).index(1)]
    confidence = max(probabilities)

    # Display prediction
    st.subheader("Prediction")

    if prediction == 1:
        st.error("Churn")
    else:
        st.success("No Churn")

    st.metric("Churn Probability", f"{churn_probability:.2%}")
    st.metric("Model Confidence", f"{confidence:.2%}")
