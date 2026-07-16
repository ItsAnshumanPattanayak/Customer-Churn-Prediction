<div align="center">

# 📉 Customer Churn Prediction

### Predict telecom customer churn before valuable customers walk away

<p>
  <a href="https://github.com/ItsAnshumanPattanayak/Customer-Churn-Prediction">
    <img src="https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github" alt="GitHub Repository">
  </a>
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Random_Forest-Classifier-16A34A?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="Random Forest">
  <img src="https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge" alt="MIT License">
</p>

<p>
  An end-to-end machine learning application that estimates telecom customer
  churn probability, translates risk into projected revenue impact, and
  visualizes the business patterns behind customer attrition.
</p>

[Features](#-key-features) •
[How It Works](#-how-it-works) •
[Installation](#-getting-started) •
[Model Pipeline](#-machine-learning-pipeline) •
[Roadmap](#-roadmap)

</div>

---

## 🌟 Overview

Customer churn directly affects recurring revenue, acquisition costs, and long-term growth. **ChurnGuard AI** turns historical telecom customer data into an interactive decision-support application that helps identify customers who may cancel their service.

A user enters five customer attributes:

- Tenure
- Monthly charges
- Total charges
- Contract type
- Internet service

The application then returns:

- A churn classification
- Churn probability
- A practical retention recommendation
- Projected 12-month customer revenue
- Estimated revenue value at risk

The project also includes interactive exploratory data analysis for understanding churn distribution, pricing behavior, contract effects, feature relationships, and charge trends over customer tenure.

> [!IMPORTANT]
> This project is designed as an educational machine learning application. Predictions should support—not replace—business judgment and responsible customer-retention policies.

---

## ✨ Key Features

<table>
<tr>
<td width="50%">

### 🎯 Real-Time Churn Prediction

- Accepts customer details through a clean Streamlit interface
- Predicts whether a customer is likely to leave
- Displays the model's churn probability
- Separates high-risk and lower-risk outcomes visually
- Suggests a simple retention action for high-risk customers

</td>
<td width="50%">

### 💰 Financial Impact Analysis

- Calculates projected 12-month revenue
- Estimates expected revenue value at risk
- Combines churn probability with monthly charges
- Helps prioritize high-value retention opportunities
- Presents business metrics in an executive-friendly format

</td>
</tr>

<tr>
<td width="50%">

### 🌍 Multi-Currency Display

- Supports USD, EUR, GBP, JPY, CNY, CAD, AUD, CHF, HKD, NZD, and INR
- Converts billing values for dashboard display
- Converts inputs back to the model's original USD scale before prediction
- Keeps model behavior independent of the selected display currency

</td>
<td width="50%">

### 📊 Interactive Analytics

- Churn distribution donut chart
- Churn versus monthly-charge violin plot
- Contract-type churn comparison
- Numerical feature correlation heatmap
- Average monthly charges across tenure

</td>
</tr>

<tr>
<td width="50%">

### 🧠 Reproducible ML Workflow

- Cleans and prepares the Telco churn dataset
- Encodes categorical variables
- Compares Logistic Regression and Random Forest
- Evaluates models using F1-score and ROC-AUC
- Saves the production model and encoders with Joblib

</td>
<td width="50%">

### ⚡ Ready-to-Use Artifacts

- Pretrained `rf_model.pkl`
- Saved categorical encoders
- Included training pipeline
- Included source dataset
- Cached model and dataset loading in Streamlit

</td>
</tr>
</table>

---

## 🖼️ Application Preview

Create an `assets/` directory and add screenshots or a GIF:

```text
assets/
├── predictor-dashboard.png
├── high-risk-result.png
├── financial-impact.png
├── analytics-dashboard.png
└── churnguard-demo.gif
```

Then add a preview near the top of this README:

```html
<p align="center">
  <img src="assets/churnguard-demo.gif" width="900" alt="ChurnGuard AI application demo">
</p>
```

A strong preview should show:

1. Customer input controls
2. A high-risk prediction
3. Revenue-at-risk metrics
4. At least two analytics charts

---

## 🧩 How It Works

```text
Historical telecom data
          │
          ▼
Data cleaning and feature selection
          │
          ▼
Categorical label encoding
          │
          ▼
Logistic Regression + Random Forest training
          │
          ▼
F1-score and ROC-AUC comparison
          │
          ▼
Random Forest model serialization
          │
          ▼
Streamlit prediction dashboard
          │
          ├── Churn probability
          ├── Retention recommendation
          ├── Revenue-at-risk analysis
          └── Interactive visualizations
```

### Prediction Flow

```text
Customer details
      │
      ▼
Currency normalization
      │
      ▼
Saved categorical encoders
      │
      ▼
Random Forest classifier
      │
      ├── Predicted class
      └── Churn probability
              │
              ▼
12-month financial risk calculation
```

---

## 🧠 Machine Learning Pipeline

### 1. Data Cleaning

The training script loads the telecom customer dataset and converts `TotalCharges` into a numeric column.

Invalid or blank values are converted to missing values and removed before training.

The target is converted into binary form:

```text
Churn = Yes  →  1
Churn = No   →  0
```

### 2. Feature Selection

The application intentionally uses five practical features to keep the prediction form simple:

| Feature | Type | Description |
|---|---|---|
| `tenure` | Numerical | Number of months the customer has stayed |
| `MonthlyCharges` | Numerical | Current monthly bill |
| `TotalCharges` | Numerical | Total amount billed |
| `Contract` | Categorical | Month-to-month, one-year, or two-year contract |
| `InternetService` | Categorical | DSL, fiber optic, or no internet service |

### 3. Categorical Encoding

`Contract` and `InternetService` are transformed using `LabelEncoder`.

The fitted encoders are saved in:

```text
encoders.pkl
```

The application reloads these same encoders during inference to ensure consistent category mappings.

### 4. Train-Test Split

The processed data is divided into:

```text
80% training data
20% testing data
```

A fixed random state is used for reproducibility.

### 5. Model Comparison

The training pipeline evaluates two classifiers:

| Model | Preprocessing | Role |
|---|---|---|
| Logistic Regression | Standardized features | Linear baseline model |
| Random Forest | Unscaled encoded features | Nonlinear ensemble model |

Both models are evaluated using:

- **F1-score** — balances precision and recall
- **ROC-AUC** — measures the ability to rank churners above non-churners

The Random Forest model is exported for use in the Streamlit application.

> [!NOTE]
> The repository contains the evaluation code but does not store the latest printed metric values. Run `model_pipeline.py` locally to reproduce the current results.

### 6. Model Serialization

The trained production artifacts are saved using Joblib:

```text
rf_model.pkl
encoders.pkl
```

---

## 💰 Financial Risk Calculation

The application estimates expected revenue loss over the next 12 months.

```text
Projected Revenue = Monthly Charges × 12
```

```text
Value at Risk = Projected Revenue × Churn Probability
```

### Example

For a customer with:

```text
Monthly Charges = $80
Churn Probability = 65%
```

The application calculates:

```text
Projected 12-Month Revenue = $960
Estimated Value at Risk    = $624
```

This value is not a guaranteed loss. It is a probability-weighted risk estimate intended to help prioritize retention activity.

---

## 🌍 Currency Handling

The dashboard supports multiple display currencies through configured conversion rates.

The selected currency affects:

- Monthly-charge input display
- Total-charge input display
- Analytics chart values
- Projected revenue
- Value-at-risk output

Before prediction, charge values are converted back to USD because the model was trained on USD-denominated data.

> [!WARNING]
> The exchange rates in `app.py` are static configuration values, not live market rates. Update them manually or connect a trusted exchange-rate API before using the project in a production environment.

---

## 📊 Analytics Dashboard

### 1. Churn Distribution

A donut chart compares the number of customers who stayed with those who churned.

### 2. Churn vs. Monthly Charges

A violin plot shows how the distribution of monthly charges differs between churned and retained customers.

### 3. Churn by Contract Type

A grouped histogram highlights the relationship between contract duration and churn behavior.

### 4. Feature Correlation

A heatmap measures linear relationships among:

- Tenure
- Monthly charges
- Total charges
- Binary churn outcome

### 5. Average Charges Over Tenure

A line chart displays how average monthly charges vary across customer tenure.

---

## 🛠️ Technology Stack

| Category | Technology |
|---|---|
| Programming language | Python |
| Web application | Streamlit |
| Data manipulation | pandas, NumPy |
| Machine learning | scikit-learn |
| Model persistence | Joblib |
| Interactive visualization | Plotly |
| Statistical visualization | Seaborn, Matplotlib |
| Production model | Random Forest Classifier |
| Baseline model | Logistic Regression |
| Dataset | Telco Customer Churn |

---

## 📁 Project Structure

```text
Customer-Churn-Prediction/
│
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Historical telecom dataset
├── app.py                                 # Streamlit prediction and analytics app
├── model_pipeline.py                      # Cleaning, training, evaluation, export
├── rf_model.pkl                           # Serialized Random Forest classifier
├── encoders.pkl                           # Serialized categorical encoders
├── correlation_heatmap.png                # Generated EDA correlation plot
├── .gitattributes                         # Git attributes
├── LICENSE                                # MIT License
└── README.md                              # Project documentation
```

---

## ⚡ Getting Started

### Prerequisites

Install:

- Python 3.9 or newer
- Git
- `pip`

### 1. Clone the Repository

```bash
git clone https://github.com/ItsAnshumanPattanayak/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

### 2. Create a Virtual Environment

#### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

The repository does not currently include a `requirements.txt` file. Install the required packages directly:

```bash
pip install streamlit pandas numpy scikit-learn joblib plotly seaborn matplotlib
```

For a reproducible project, create `requirements.txt` with:

```text
streamlit
pandas
numpy
scikit-learn
joblib
plotly
seaborn
matplotlib
```

Then install with:

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

Open the local address shown in the terminal, normally:

```text
http://localhost:8501
```

---

## 🔁 Retraining the Model

The repository already contains trained artifacts, so retraining is optional.

To clean the dataset, train both models, evaluate them, regenerate the heatmap, and export the Random Forest model:

```bash
python model_pipeline.py
```

This command updates:

```text
correlation_heatmap.png
rf_model.pkl
encoders.pkl
```

> [!CAUTION]
> Retraining overwrites the existing serialized model and encoder files. Commit or back up the current artifacts before experimenting.

---

## 🧪 Input Example

Try the following high-risk-style profile:

| Input | Example |
|---|---|
| Tenure | 4 months |
| Monthly Charges | 95 |
| Total Charges | 380 |
| Contract | Month-to-month |
| Internet Service | Fiber optic |

The exact result depends on the saved model. The application will display the predicted class, churn probability, and financial impact.

---

## 📦 Dataset

The project includes the commonly used Telco Customer Churn dataset.

It contains customer-level information such as:

- Demographic attributes
- Account tenure
- Phone and internet services
- Contract and billing methods
- Monthly and total charges
- Churn status

Only five features are used by the current deployed model to keep the application focused and easy to operate.

> Review the original dataset source and licensing conditions before commercial redistribution.

---

## ✅ Suggested Testing Checklist

- [ ] The dataset loads without missing-file errors
- [ ] The model and encoder files deserialize correctly
- [ ] Every contract option is recognized by the saved encoder
- [ ] Every internet-service option is recognized
- [ ] Currency conversion is reversed correctly before inference
- [ ] Minimum and maximum input values are accepted
- [ ] Prediction probabilities remain between 0 and 1
- [ ] Revenue-at-risk calculations use the selected display currency
- [ ] All five visualizations render successfully
- [ ] The dashboard works on desktop and mobile widths
- [ ] Retraining reproduces valid model artifacts

---

## 🌐 Deployment

### Streamlit Community Cloud

1. Push the repository to GitHub.
2. Sign in to Streamlit Community Cloud.
3. Select this repository.
4. Set the main file path to:

```text
app.py
```

5. Ensure all required files are committed:
   - Dataset
   - Random Forest model
   - Encoders
   - Dependency file
6. Deploy the application.

After deployment, add a live-demo badge:

```html
<a href="YOUR_STREAMLIT_APP_URL">
  <img src="https://img.shields.io/badge/Live_Demo-Open_Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white">
</a>
```

---

## 🗺️ Roadmap

- [ ] Add a pinned `requirements.txt`
- [ ] Record model metrics in the README
- [ ] Use a stratified train-test split
- [ ] Add a confusion matrix and ROC curve
- [ ] Compare precision, recall, and calibration
- [ ] Add cross-validation and hyperparameter tuning
- [ ] Move preprocessing into a scikit-learn `Pipeline`
- [ ] Add probability calibration
- [ ] Replace static currency values with a live exchange-rate service
- [ ] Add SHAP explanations for individual predictions
- [ ] Add batch prediction from uploaded CSV files
- [ ] Export retention-priority reports
- [ ] Add unit and integration tests
- [ ] Add GitHub Actions for automated testing
- [ ] Containerize the application with Docker
- [ ] Deploy a public Streamlit demo
- [ ] Add application screenshots and a demo GIF

---

## ⚠️ Limitations

- The deployed model uses only five features from the full dataset.
- Label encoding introduces an artificial numeric order for categorical values.
- Exchange rates are hardcoded.
- The train-test split is not currently stratified.
- No probability calibration is performed.
- The saved model may become outdated when customer behavior changes.
- Revenue at risk is a simplified expected-value estimate.
- The application does not currently explain which features drove an individual prediction.

These limitations make excellent directions for future development.

---

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a feature branch:

```bash
git checkout -b feature/your-feature-name
```

3. Commit your changes:

```bash
git commit -m "Add: your feature description"
```

4. Push the branch:

```bash
git push origin feature/your-feature-name
```

5. Open a pull request.

---

## 🐛 Reporting Issues

When opening an issue, include:

- A clear problem description
- Steps to reproduce the behavior
- Expected and actual results
- Relevant terminal logs
- Python and operating-system versions
- Screenshots when applicable

Use the repository's [Issues](https://github.com/ItsAnshumanPattanayak/Customer-Churn-Prediction/issues) page.

---

## 👨‍💻 Author

<div align="center">

### Anshuman Pattanayak

[![GitHub](https://img.shields.io/badge/GitHub-ItsAnshumanPattanayak-181717?style=for-the-badge&logo=github)](https://github.com/ItsAnshumanPattanayak)

Turning customer behavior into actionable machine learning insights.

</div>

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

<div align="center">

### ⭐ Support the Project

If you found this project useful, consider giving the repository a star.

**Built with Python, Streamlit, machine learning, and business-focused analytics.**

</div>
