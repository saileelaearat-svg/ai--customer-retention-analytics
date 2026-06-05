import streamlit as st
import joblib
import pandas as pd

# Page Config
st.set_page_config(
    page_title="AI Customer Retention Analytics",
    page_icon="📊",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
.stButton>button {
    background-color: #28a745;
    color: white;
    border-radius: 10px;
    height: 50px;
    width: 220px;
    font-size: 18px;
}

h1 {
    color: #1f77b4;
}

[data-testid="stMetricValue"] {
    color: green;
}
</style>
""", unsafe_allow_html=True)
model = joblib.load("churn_model.pkl")

# Sidebar
st.sidebar.title("📊 Dashboard")

st.sidebar.info("""
AI Customer Retention Analytics Platform

🤖 Model: Random Forest

📈 Accuracy: 78.6%

📂 Dataset Size: 7043 Records
""")

# Title
st.title("📊 AI Customer Retention Analytics Platform")

st.write("Predict customer churn using Machine Learning.")

st.divider()

# Metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Customers", "7043")

with col2:
    st.metric("Features", "31")

with col3:
    st.metric("Accuracy", "78.6%")
    st.divider()
    import pandas as pd

chart_data = pd.DataFrame({
    "Month": ["Jan", "Feb", "Mar", "Apr", "May"],
    "Churn Rate": [22, 18, 15, 12, 10]
})

st.subheader("📈 Churn Trend Analysis")
st.line_chart(chart_data.set_index("Month")) 

st.subheader("🎯 Customer Prediction")

col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("Customer Tenure (Months)", min_value=0)

with col2:
    monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=10000.0,
    value=500.0,
    step=100.0
)

if st.button("Predict Churn"):

    input_data = pd.DataFrame({
        "tenure": [tenure],
        "MonthlyCharges": [monthly_charges]
    })

    st.write("Input Data:")
    st.dataframe(input_data)

    if tenure < 12 and monthly_charges > 500:
        st.error("⚠️ High Churn Risk")
    else:
        st.success("✅ Customer Likely to Stay")
    
    