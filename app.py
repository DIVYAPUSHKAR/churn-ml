import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/churn_model_short.pkl")

st.title("📊 Customer Churn Prediction App")
st.write("Enter customer details to predict churn:")

# Collect inputs
gender = st.selectbox("Gender", ["Male", "Female"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
total_charges = monthly_charges * tenure  # Auto-calculate
payment_method = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])

# Convert to DataFrame
input_data = pd.DataFrame({
    "gender": [gender],
    "tenure": [tenure],
    "Contract": [contract],
    "InternetService": [internet_service],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges],
    "PaymentMethod": [payment_method],
    "Partner": [partner],
    "Dependents": [dependents]
})

# Predict
if st.button("Predict Churn"):
    try:
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.error(f"❌ Customer is likely to churn (Probability: {proba:.2f})")
        else:
            st.success(f"✅ Customer is likely to stay (Probability: {proba:.2f})")
    except Exception as e:
        st.error(f"⚠️ Error: {e}")

