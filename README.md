# 🛡️ Insurance Policy Approval Predictor

A modern, professional **Machine Learning web application** built with Streamlit that predicts whether an insurance policy application will be **approved** or **rejected** based on applicant profile.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-FF4B4B?logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Charts-3F4F75?logo=plotly&logoColor=white)

---

## 📸 Features

| Feature | Description |
|:--------|:------------|
| **🎨 Premium UI** | Glassmorphism design with gradient backgrounds and smooth animations |
| **🤖 ML Prediction** | Real-time insurance approval prediction using a trained scikit-learn model |
| **📊 Interactive Charts** | Approval Gauge, Risk Meter, Feature Contributions & History charts via Plotly |
| **🕒 Prediction History** | Session-based history with CSV export functionality |
| **💡 Insights Panel** | Key factors that influence policy approval decisions |
| **📱 Responsive Layout** | Professional SaaS-style dashboard that works on all screen sizes |

---

## 🗂️ Project Structure

```
Insurance-Policy-Approval/
├── app.py                          # Main Streamlit application
├── Insurance.pkl                   # Pre-trained ML model (pickle format)
├── Insurance_Policy_approvar.ipynb # Jupyter notebook with model training
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules
└── README.md                       # Project documentation
```

---

## 📋 Dataset Features

The model uses the following **9 features** to predict policy approval:

| # | Feature | Type | Description |
|:-:|:--------|:-----|:------------|
| 1 | **Age** | Numeric | Applicant's age in years |
| 2 | **Gender** | Categorical | Male / Female (encoded as `Gender_Male`: 1/0) |
| 3 | **BMI** | Numeric | Body Mass Index |
| 4 | **Smoker** | Categorical | Yes / No (encoded as `Smoker_Yes`: 1/0) |
| 5 | **Diabetic** | Categorical | Yes / No (encoded as `Diabetic_Yes`: 1/0) |
| 6 | **AnnualIncome** | Numeric | Yearly income (₹) |
| 7 | **VehicleAge** | Numeric | Age of the insured vehicle (years) |
| 8 | **PreviousClaims** | Numeric | Number of prior insurance claims |
| 9 | **CreditScore** | Numeric | Financial credit score (300–900) |

**Target Variable:** `PolicyApproved` (1 = Approved, 0 = Rejected)

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/prasad2154/Insurance-Policy-Approval.git
   cd Insurance-Policy-Approval
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open in browser**
   The app will launch at [http://localhost:8501](http://localhost:8501)

---

## 📦 Dependencies

| Package | Purpose |
|:--------|:--------|
| `streamlit` | Web application framework |
| `pandas` | Data manipulation |
| `numpy` | Numerical computations |
| `scikit-learn` | Machine learning model |
| `plotly` | Interactive visualizations |

All dependencies are listed in `requirements.txt`.

---

## 🎯 How It Works

```
User Input ──▶ Encode Categoricals ──▶ Model Prediction ──▶ Display Results
               (One-Hot Encoding)       (Insurance.pkl)      (Charts + Cards)
```

1. The user fills in applicant details via the input form.
2. Categorical features (`Gender`, `Smoker`, `Diabetic`) are one-hot encoded to match training data.
3. The pre-trained model (`Insurance.pkl`) predicts approval and returns probability scores.
4. Results are displayed with interactive Plotly charts and metric cards.
5. Every prediction is stored in session state for history tracking.

---

## 📊 Visualizations

- **🎯 Approval Gauge** — Circular gauge showing approval probability percentage.
- **⚠️ Risk Meter** — Horizontal bar indicating risk level (Low → Very High).
- **📊 Feature Contributions** — Normalized bar chart of all input features.
- **📈 Prediction History** — Line chart tracking approval probability across predictions.

---

## 💡 Key Insights

- ✅ Higher **Credit Score** (700+) significantly increases approval probability.
- ✅ Higher **Annual Income** demonstrates financial stability.
- ⚠️ **Previous Claims** negatively impact approval chances.
- ⚠️ **Smoking** and **Diabetes** increase perceived health risk.
- ✅ **Younger Vehicle Age** indicates newer, safer vehicles.
- ✅ **Healthy BMI** (18.5–24.9) is associated with lower risk.

---

## 🛠️ Tech Stack

| Technology | Role |
|:-----------|:-----|
| **Streamlit** | Frontend & Backend |
| **Scikit-Learn** | ML Model (loaded via pickle) |
| **Plotly** | Interactive Charts |
| **Pandas / NumPy** | Data Processing |
| **Custom CSS** | Glassmorphism UI & Animations |

---

## 📄 License

This project is for educational purposes as part of the AI Course (G_38).

---

## 🙏 Acknowledgements

- Built with [Streamlit](https://streamlit.io/), [Scikit-Learn](https://scikit-learn.org/) & [Plotly](https://plotly.com/python/)
- UI inspired by modern SaaS dashboard design patterns
- Training mentor guidance for course G_38

---

<p align="center">
  <strong>🛡️ Insurance Policy Approval Predictor</strong><br>
  Built with ❤️ using Streamlit, Scikit-Learn & Plotly<br>
  © 2026 • All Rights Reserved
</p>

**Last Updated:** 2026
