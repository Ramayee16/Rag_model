import streamlit as st
import pandas as pd
import joblib

# ----------------------------
# Load model and scaler
# ----------------------------
try:
    model = joblib.load("hr_model.joblib")
    scaler = joblib.load("hr_scaler.joblib")
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

# ----------------------------
# Load HR CSV
# ----------------------------
try:
    df = pd.read_csv("HR_comma_sep.csv")
except Exception as e:
    st.warning(f"Could not load CSV: {e}")
    df = None

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🤖 HR Prediction System")
st.write("This app predicts outcomes based on HR dataset.")

st.subheader("Enter Employee Details")

# Example input fields (adjust according to your model features)
age = st.number_input("Age", min_value=18, max_value=70, value=30)
salary = st.number_input("Salary", min_value=10000, max_value=500000, value=50000)
experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=5)

if st.button("Predict"):
    try:
        # Scale input
        X = scaler.transform([[age, salary, experience]])  # Adjust columns if needed
        # Make prediction
        prediction = model.predict(X)
        st.success(f"Prediction Result: {prediction[0]}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Optional: show CSV data
if st.checkbox("Show HR Dataset"):
    if df is not None:
        st.dataframe(df)
