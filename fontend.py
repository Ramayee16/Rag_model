import streamlit as st
import pandas as pd
import joblib

# ----------------------------
# Load your trained model & scaler
# ----------------------------
with open("hr_model.joblib", "rb") as f:
    model = joblib.load(f)

with open("hr_scaler.joblib", "rb") as f:
    scaler = joblib.load(f)

# ----------------------------
# Load HR data
# ----------------------------
df = pd.read_csv("HR_comma_sep.csv")

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🤖 HR Prediction / Q&A System")
st.write("This app predicts or answers questions based on your HR dataset.")

# Example input: let's assume you want to predict employee attrition
st.subheader("Employee Details Input")
age = st.number_input("Age", min_value=18, max_value=70, value=30)
salary = st.number_input("Salary", min_value=10000, max_value=500000, value=50000)
experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=5)

if st.button("Predict"):
    # Example: scaling input and predicting
    X = scaler.transform([[age, salary, experience]])  # adjust columns to your model
    prediction = model.predict(X)
    
    st.write(f"Prediction Result: {prediction[0]}")

# Optional: show raw CSV data
if st.checkbox("Show HR Data"):
    st.write(df)
