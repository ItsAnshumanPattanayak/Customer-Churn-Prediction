```markdown
# 📉 Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-100%25-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

I built an intelligent web application that helps telecom companies predict if a customer is about to cancel their service. I fed historical customer data into a machine learning algorithm so it could learn the warning signs of someone leaving—like having a short-term contract but very high monthly bills. Now, a user can type a customer's details into my app, and it instantly calculates a **'Churn Probability.'** If the risk is high, the company knows exactly who to target with a retention offer.

---

## ✨ Key Features

* **Instant Churn Prediction:** Get real-time probability scores on whether a customer will leave.
* **Customer Lifetime Value (CLV) Risk Metric:** Assesses the financial impact of customer churn to prioritize high-value retention efforts.
* **Data-Driven Insights:** Identifies key churn indicators from historical billing and contract data.
* **Pre-trained ML Pipeline:** Includes ready-to-use encoders (`encoders.pkl`) and a trained Random Forest model (`rf_model.pkl`).

---

## 📂 Repository Structure

```text
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Raw historical dataset
├── app.py                                # Main web application script
├── correlation_heatmap.png               # EDA visual of feature correlations
├── encoders.pkl                          # Saved data encoders for categorical variables
├── model_pipeline.py                     # Data processing and ML pipeline
├── rf_model.pkl                          # Serialized Machine Learning model
├── README.md                             # Project documentation
└── LICENSE                               # MIT License

```

---

## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

* Python 3.x installed on your system.

### Installation & Usage

1. **Clone the repository:**
```bash
git clone [https://github.com/ItsAnshumanPattanayak/Customer-Churn-Prediction.git](https://github.com/ItsAnshumanPattanayak/Customer-Churn-Prediction.git)
cd Customer-Churn-Prediction

```


2. **Install dependencies:**
Make sure you have the necessary libraries installed for the pipeline.
```bash
pip install pandas scikit-learn

```


*(Note: Add any other frameworks you used in `app.py`, such as Streamlit or Flask, to this install command).*
3. **Run the web application:**
```bash
python app.py

```


4. **Open your browser:** Navigate to the local URL provided in your terminal to interact with the app.

---

## 👨‍💻 Author

**Anshuman Pattanayak**

* [GitHub Profile](https://github.com/ItsAnshumanPattanayak)

---

## 📜 License

This project is licensed under the [MIT License](https://www.google.com/search?q=LICENSE).

```

```
