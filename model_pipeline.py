import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score

# 1. Load Data (ensure the CSV is in the same folder)
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# 2. Data Cleaning
# TotalCharges contains blank spaces for new customers. Coerce to NaN and drop.
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# Convert target variable to binary (1 = Churn, 0 = Stay)
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# 3. Exploratory Data Analysis (EDA)
# Save a correlation heatmap for numerical variables to see how tenure affects churn
plt.figure(figsize=(8,6))
sns.heatmap(df[['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.savefig("correlation_heatmap.png")
print("EDA Plot saved as correlation_heatmap.png")

# 4. Feature Selection & Encoding
# Selecting top 5 features to keep the Streamlit app user-friendly
features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'InternetService']
X = df[features].copy()
y = df['Churn']

# Label Encode categorical features so the model can read them
encoders = {}
for col in ['Contract', 'InternetService']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# 5. Train-Test Split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Model Training
# Logistic Regression (Requires scaled data for continuous variables)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)

# Random Forest (Tree-based, no scaling required)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 7. Evaluation
print("\n--- Logistic Regression Metrics ---")
print(f"F1-Score: {f1_score(y_test, lr.predict(X_test_scaled)):.4f}")
print(f"ROC-AUC:  {roc_auc_score(y_test, lr.predict_proba(X_test_scaled)[:, 1]):.4f}")

print("\n--- Random Forest Metrics ---")
print(f"F1-Score: {f1_score(y_test, rf.predict(X_test)):.4f}")
print(f"ROC-AUC:  {roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]):.4f}")

print("\n--- Feature Importance (Random Forest) ---")
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_})
print(importance_df.sort_values(by='Importance', ascending=False))

# 8. Export the Random Forest model and encoders for the app
joblib.dump(rf, 'rf_model.pkl')
joblib.dump(encoders, 'encoders.pkl')
print("\nModel and encoders saved successfully.")