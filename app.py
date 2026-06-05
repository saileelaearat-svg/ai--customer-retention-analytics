import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="AI Customer Retention Analytics", layout="wide", page_icon="📊")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df.drop('customerID', axis=1, inplace=True)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    return df

df = load_data()

# Load model
model = joblib.load('churn_model.pkl')

# Get feature columns
df_encoded = pd.get_dummies(df, drop_first=True)
feature_cols = df_encoded.drop('Churn', axis=1).columns

# Sidebar
st.sidebar.title("📊 Dashboard")
st.sidebar.write("AI Customer Retention Analytics Platform")
st.sidebar.write("🔵 Model: Random Forest")
st.sidebar.write("📈 Accuracy: 84%")
st.sidebar.write("📁 F1 Score: 0.84")
st.sidebar.write("📁 AUC: 0.84")
st.sidebar.write("🗂 Dataset Size: 7043 Records")

# Main title
st.title("📊 AI Customer Retention Analytics Platform")
st.write("Predict customer churn using Machine Learning.")

# Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Customers", "7043")
col2.metric("Features", "31")
col3.metric("Accuracy", "84%")

st.divider()

# Churn distribution
st.subheader("📉 Churn Distribution")
churn_counts = df['Churn'].value_counts()
fig, ax = plt.subplots(figsize=(5, 3))
ax.bar(['No Churn', 'Churn'], churn_counts.values, color=['green', 'red'])
ax.set_ylabel("Count")
col1, col2 = st.columns([1, 1])
with col1:
    st.pyplot(fig)

st.divider()

# Feature importance
st.subheader("🔍 Top Features Causing Churn")
importances = model.feature_importances_
feat_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': importances
}).sort_values('Importance', ascending=False).head(10)

fig2, ax2 = plt.subplots(figsize=(6, 4))
ax2.barh(feat_df['Feature'], feat_df['Importance'], color='steelblue')
ax2.invert_yaxis()
col1, col2 = st.columns([1, 1])
with col1:
    st.pyplot(fig2)

st.divider()

# Prediction section
st.subheader("🎯 Customer Churn Prediction")

col1, col2, col3 = st.columns(3)
with col1:
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    monthly = st.slider("Monthly Charges ($)", 0.0, 200.0, 65.0)
with col2:
    total = st.slider("Total Charges ($)", 0.0, 10000.0, 1000.0)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
with col3:
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])

if st.button("🔍 Predict Churn", use_container_width=True):
    # Build input with all features as 0
    input_data = pd.DataFrame([{col: 0 for col in feature_cols}])
    input_data['tenure'] = tenure
    input_data['MonthlyCharges'] = monthly
    input_data['TotalCharges'] = total

    # Encode contract
    if contract == "One year":
        if 'Contract_One year' in input_data.columns:
            input_data['Contract_One year'] = 1
    elif contract == "Two year":
        if 'Contract_Two year' in input_data.columns:
            input_data['Contract_Two year'] = 1

    # Encode internet
    if internet == "Fiber optic":
        if 'InternetService_Fiber optic' in input_data.columns:
            input_data['InternetService_Fiber optic'] = 1
    elif internet == "No":
        if 'InternetService_No' in input_data.columns:
            input_data['InternetService_No'] = 1

    # Encode payment
    if payment == "Mailed check":
        if 'PaymentMethod_Mailed check' in input_data.columns:
            input_data['PaymentMethod_Mailed check'] = 1
    elif payment == "Bank transfer (automatic)":
        if 'PaymentMethod_Bank transfer (automatic)' in input_data.columns:
            input_data['PaymentMethod_Bank transfer (automatic)'] = 1
    elif payment == "Credit card (automatic)":
        if 'PaymentMethod_Credit card (automatic)' in input_data.columns:
            input_data['PaymentMethod_Credit card (automatic)'] = 1

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Churn Risk! Probability: {probability:.0%}")
    else:
        st.success(f"✅ Low Churn Risk! Probability: {probability:.0%}")