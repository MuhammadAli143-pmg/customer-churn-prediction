from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "churn_model.joblib"

st.title("📊 Telecom Customer Churn Predictor")
st.write(
    "Estimate a customer's likelihood of leaving and use the result to guide "
    "a proactive retention conversation."
)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(
            "The trained model file 'churn_model.joblib' is missing. "
            "Run the Colab notebook through the model-saving section and place "
            "the downloaded model file beside app.py."
        )
        st.stop()
    return joblib.load(MODEL_PATH)

model = load_model()

st.subheader("Customer profile")
with st.form("customer_details"):
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        partner = st.selectbox("Has partner?", ["Yes", "No"])
        dependents = st.selectbox("Has dependents?", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)

    with col2:
        phone_service = st.selectbox("Phone service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device protection", ["Yes", "No", "No internet service"])

    with col3:
        tech_support = st.selectbox("Tech support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming movies", ["Yes", "No", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )

    st.subheader("Billing information")
    bill1, bill2 = st.columns(2)
    with bill1:
        monthly_charges = st.number_input("Monthly charges", min_value=0.0, max_value=2000.0, value=70.0, step=1.0)
    with bill2:
        total_charges = st.number_input(
            "Total charges", min_value=0.0, max_value=100000.0,
            value=float(monthly_charges * tenure), step=10.0
        )

    submitted = st.form_submit_button("Predict churn risk", type="primary")

if submitted:
    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    probability = float(model.predict_proba(customer)[0, 1])
    if probability >= 0.60:
        risk = "High"
        recommendation = (
            "Contact this customer promptly, understand the reason for dissatisfaction, "
            "and consider a relevant retention offer."
        )
    elif probability >= 0.35:
        risk = "Medium"
        recommendation = (
            "Schedule a proactive check-in, review service usage and billing concerns, "
            "and explain suitable plan options."
        )
    else:
        risk = "Low"
        recommendation = (
            "Continue normal engagement, monitor satisfaction, and maintain reliable service."
        )

    st.divider()
    st.subheader("Prediction")
    result1, result2 = st.columns(2)
    with result1:
        st.metric("Estimated churn probability", f"{probability:.1%}")
    with result2:
        st.metric("Churn risk category", risk)

    st.progress(min(max(probability, 0.0), 1.0))
    st.info(recommendation)
    st.caption(
        "This prediction is a decision-support estimate, not a guarantee. "
        "Risk thresholds are illustrative and should be validated against retention capacity and business costs."
    )

with st.expander("About this model"):
    st.write(
        "The model is trained in the accompanying notebook using a preprocessing pipeline "
        "with scaled numerical features and one-hot encoded categorical features. "
        "The notebook compares Logistic Regression, Decision Tree and Random Forest, "
        "then tunes the selected model using recall-focused cross-validation."
    )
