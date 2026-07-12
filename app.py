import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Page Configuration
st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

# --- CURRENCY SETUP ---
CURRENCIES = {
    "United States Dollar (USD)": {"rate": 1.0, "symbol": "$"},
    "Euro (EUR)": {"rate": 0.92, "symbol": "€"},
    "British Pound Sterling (GBP)": {"rate": 0.79, "symbol": "£"},
    "Japanese Yen (JPY)": {"rate": 150.0, "symbol": "¥"},
    "Chinese Yuan (CNY)": {"rate": 7.2, "symbol": "¥"},
    "Canadian Dollar (CAD)": {"rate": 1.36, "symbol": "C$"},
    "Australian Dollar (AUD)": {"rate": 1.52, "symbol": "A$"},
    "Swiss Franc (CHF)": {"rate": 0.88, "symbol": "CHF"},
    "Hong Kong Dollar (HKD)": {"rate": 7.8, "symbol": "HK$"},
    "New Zealand Dollar (NZD)": {"rate": 1.65, "symbol": "NZ$"},
    "Indian Rupee (INR)": {"rate": 83.0, "symbol": "₹"}
}

st.sidebar.header("⚙️ Global Settings")
selected_currency = st.sidebar.selectbox("Select Currency", list(CURRENCIES.keys()))
rate = CURRENCIES[selected_currency]["rate"]
sym = CURRENCIES[selected_currency]["symbol"]

# 2. Load the saved model, encoders, and raw data
@st.cache_resource
def load_models():
    return joblib.load('rf_model.pkl'), joblib.load('encoders.pkl')

@st.cache_data
def load_data():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    return df

model, encoders = load_models()
df_raw = load_data()

# Convert raw data to selected currency for the graphs
df = df_raw.copy()
df['MonthlyCharges'] = df['MonthlyCharges'] * rate
df['TotalCharges'] = df['TotalCharges'] * rate

st.title("📊 Customer Churn Prediction & Analytics")

# 3. Create Tabs
tab1, tab2 = st.tabs(["🔮 Churn Predictor", "📈 Data Visualizations"])

# --- TAB 1: PREDICTION TOOL ---
with tab1:
    st.write("Enter the customer's data below to predict whether they will cancel their service.")

    col1, col2 = st.columns(2)

    with col1:
        tenure = st.slider("Tenure (Months)", 0, 72, 12)
        monthly_charges = st.number_input(f"Monthly Charges ({sym})", float(18.0 * rate), float(120.0 * rate), float(50.0 * rate))
        total_charges = st.number_input(f"Total Charges ({sym})", float(18.0 * rate), float(8000.0 * rate), float(500.0 * rate))

    with col2:
        contract = st.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
        internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])

    if st.button("Predict Churn"):
        # Encode categorical inputs
        contract_encoded = encoders['Contract'].transform([contract])[0]
        internet_encoded = encoders['InternetService'].transform([internet_service])[0]
        
        # Convert charges back to USD for the model prediction
        monthly_charges_usd = monthly_charges / rate
        total_charges_usd = total_charges / rate
        
        # Bundle into an array
        features = np.array([[tenure, monthly_charges_usd, total_charges_usd, contract_encoded, internet_encoded]])
        
        # Predict
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
        
        st.divider()
        
        # Display Prediction Result
        if prediction == 1:
            st.error(f"⚠️ High Risk of Churn! (Probability: {probability:.2%})")
            st.write("Recommendation: Offer a discount or upgrade to a longer-term contract.")
        else:
            st.success(f"✅ Customer is likely to stay. (Churn Probability: {probability:.2%})")
            
        st.divider()
        
        # --- NEW FEATURE: Financial Impact Analysis (CLV Risk) ---
        st.subheader("💼 Financial Impact Analysis (12-Month Projection)")
        st.write("This calculates the expected revenue loss over the next year based on this customer's churn probability.")
        
        # Calculate Yearly Revenue and Risk in the user's selected currency
        yearly_revenue = monthly_charges * 12
        value_at_risk = yearly_revenue * probability
        
        # Display metrics in 3 columns
        col_m1, col_m2, col_m3 = st.columns(3)
        
        with col_m1:
            st.metric(label="Projected 12-Month Revenue", value=f"{sym}{yearly_revenue:,.2f}")
            
        with col_m2:
            st.metric(label="Churn Probability", value=f"{probability:.1%}")
            
        with col_m3:
            # The delta shows the risk in red to emphasize the potential loss
            st.metric(label="Value at Risk", 
                      value=f"{sym}{value_at_risk:,.2f}", 
                      delta=f"-{sym}{value_at_risk:,.2f}", 
                      delta_color="inverse")

# --- TAB 2: DATA VISUALIZATIONS ---
with tab2:
    st.header("Exploratory Data Analysis")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("1. Churn Distribution")
        pie_fig = px.pie(df, names='Churn', color='Churn', 
                         color_discrete_map={'Yes': '#ff4b4b', 'No': '#00cc96'}, hole=0.4)
        st.plotly_chart(pie_fig, use_container_width=True)
        
    with col_b:
        st.subheader(f"2. Churn vs. Monthly Charges ({sym})")
        violin_fig = px.violin(df, x="Churn", y="MonthlyCharges", color="Churn", 
                               box=True, points="all", color_discrete_map={'Yes': '#ff4b4b', 'No': '#00cc96'})
        st.plotly_chart(violin_fig, use_container_width=True)

    st.divider()

    col_c, col_d = st.columns(2)
    
    with col_c:
        st.subheader("3. Churn by Contract Type")
        bar_fig = px.histogram(df, x="Contract", color="Churn", barmode="group",
                               color_discrete_map={'Yes': '#ff4b4b', 'No': '#00cc96'})
        st.plotly_chart(bar_fig, use_container_width=True)
        
    with col_d:
        st.subheader("4. Feature Correlation")
        corr_df = df[['tenure', 'MonthlyCharges', 'TotalCharges']].copy()
        corr_df['Churn_Num'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
        corr = corr_df.corr()
        
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.heatmap(corr, annot=True, cmap="Blues", fmt=".2f", ax=ax, linewidths=0.5)
        st.pyplot(fig)

    st.divider()
    
    st.subheader(f"5. Average Monthly Charges Over Time ({sym})")
    line_data = df.groupby('tenure')['MonthlyCharges'].mean().reset_index()
    line_fig = px.line(line_data, x='tenure', y='MonthlyCharges', 
                       labels={'tenure': 'Tenure (Months)', 'MonthlyCharges': f'Average Monthly Charge ({sym})'})
    line_fig.update_traces(line_color='#3366cc', line_width=3)
    st.plotly_chart(line_fig, use_container_width=True)