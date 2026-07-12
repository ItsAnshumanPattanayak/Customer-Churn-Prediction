import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Page Configuration (Makes the app wider to fit graphs)
st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

# 2. Load the saved model, encoders, and raw data
@st.cache_resource
def load_models():
    return joblib.load('rf_model.pkl'), joblib.load('encoders.pkl')

@st.cache_data
def load_data():
    # Load data for visualizations
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    return df

model, encoders = load_models()
df = load_data()

st.title("📊 Customer Churn Prediction & Analytics")

# 3. Create Tabs for a clean UI
tab1, tab2 = st.tabs(["🔮 Churn Predictor", "📈 Data Visualizations"])

# --- TAB 1: PREDICTION TOOL ---
with tab1:
    st.write("Enter the customer's data below to predict whether they will cancel their service.")

    col1, col2 = st.columns(2)

    with col1:
        tenure = st.slider("Tenure (Months)", 0, 72, 12)
        monthly_charges = st.number_input("Monthly Charges ($)", 18.0, 120.0, 50.0)
        total_charges = st.number_input("Total Charges ($)", 18.0, 8000.0, 500.0)

    with col2:
        contract = st.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
        internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])

    if st.button("Predict Churn"):
        # Encode categorical inputs
        contract_encoded = encoders['Contract'].transform([contract])[0]
        internet_encoded = encoders['InternetService'].transform([internet_service])[0]
        
        # Bundle into an array
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

# --- TAB 2: DATA VISUALIZATIONS ---
with tab2:
    st.header("Exploratory Data Analysis")
    st.write("Understand the historical trends of why customers leave.")
    
    # First row of graphs
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("1. Churn Distribution (Pie Chart)")
        # Shows the overall imbalance of the dataset
        pie_fig = px.pie(df, names='Churn', color='Churn', 
                         color_discrete_map={'Yes': '#ff4b4b', 'No': '#00cc96'}, hole=0.4)
        st.plotly_chart(pie_fig, use_container_width=True)
        
    with col_b:
        st.subheader("2. Churn vs. Monthly Charges (Violin Plot)")
        # Shows where the density of churning customers sits based on price
        violin_fig = px.violin(df, x="Churn", y="MonthlyCharges", color="Churn", 
                               box=True, points="all", color_discrete_map={'Yes': '#ff4b4b', 'No': '#00cc96'})
        st.plotly_chart(violin_fig, use_container_width=True)

    st.divider()

    # Second row of graphs
    col_c, col_d = st.columns(2)
    
    with col_c:
        st.subheader("3. Churn by Contract Type (Bar Chart)")
        # Highly useful to prove that month-to-month contracts are risky
        bar_fig = px.histogram(df, x="Contract", color="Churn", barmode="group",
                               color_discrete_map={'Yes': '#ff4b4b', 'No': '#00cc96'})
        st.plotly_chart(bar_fig, use_container_width=True)
        
    with col_d:
        st.subheader("4. Feature Correlation (Heatmap)")
        # Shows the mathematical relationship between our numerical variables
        corr_df = df[['tenure', 'MonthlyCharges', 'TotalCharges']].copy()
        corr_df['Churn_Num'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
        corr = corr_df.corr()
        
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.heatmap(corr, annot=True, cmap="Blues", fmt=".2f", ax=ax, linewidths=0.5)
        # We use Matplotlib inside Streamlit for the heatmap as it looks cleaner
        st.pyplot(fig)

    st.divider()
    
    # Full width bottom graph
    st.subheader("5. Average Monthly Charges Over Time (Line Graph)")
    # Group the data to see if people who stay longer end up paying more or less on average
    line_data = df.groupby('tenure')['MonthlyCharges'].mean().reset_index()
    line_fig = px.line(line_data, x='tenure', y='MonthlyCharges', 
                       labels={'tenure': 'Tenure (Months)', 'MonthlyCharges': 'Average Monthly Charge ($)'})
    line_fig.update_traces(line_color='#3366cc', line_width=3)
    st.plotly_chart(line_fig, use_container_width=True)