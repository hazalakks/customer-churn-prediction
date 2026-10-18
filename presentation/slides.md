# Executive Presentation: Telecom Customer Churn Prediction

---

# Slide 1: Title & Executive Summary

## 📉 Customer Churn Prediction & Retention Optimization
### *An End-to-End Machine Learning Framework with SHAP Explainability & Interactive Decision Support*

- **Author**: Hazal
- **Date**: October 2026
- **Domain**: Telecommunications & Predictive Customer Analytics
- **Technologies**: Python | scikit-learn | XGBoost | SHAP | Streamlit

---

# Slide 2: Problem Statement & Dataset

## 🎯 The Retention Challenge & Data Foundation

### Business Problem
- **Cost Differential**: Retaining an existing customer is **5× to 25× cheaper** than acquiring a new subscriber.
- **Goal**: Proactively flag at-risk accounts, quantify attrition drivers, and optimize retention intervention ROI before cancellations occur.

### Dataset Overview (IBM Telco Churn)
- **Scale**: 7,043 Customer Records | 21 Features (Demographics, Services, Billing, Tenure)
- **Target Variable**: `Churn` (Yes: 26.5%, No: 73.5%) — inherent class imbalance requiring stratified evaluation.
- **Key Indicators**:
  - Contract Type (Month-to-month, 1-Year, 2-Year)
  - Tenure Range: 0 to 72 months (Median: 29 months)
  - Monthly Spend: $18.25 to $118.75 (Mean: $64.76)

---

# Slide 3: Engineering Methodology & Architecture

## ⚙️ Robust Data Pipeline & Multi-Model Benchmarking

### End-to-End Pipeline
- **Data Cleansing**: Type conversion, missing charge imputation (0.0 for zero-tenure accounts), binary target mapping.
- **Feature Engineering (8+ Domain Features)**:
  - `AvgMonthlyCharge`, `PriceDiscrepancy` (detecting sudden bill increases)
  - `NumServices` (bundling density) & `UnprotectedInternet` (no tech support / security)
  - `TenureCohort` (New, Mid-term, Long-term) & `ContractScore` (ordinal loyalty rating)
- **Leak-Free Transformation**: Scikit-Learn `ColumnTransformer` (StandardScaler + OneHotEncoder).

### Modeling Strategy
- Stratified 5-Fold Cross-Validation to safeguard against class imbalance.
- Hyperparameter tuning via GridSearchCV for 5 diverse architectures:
  - *Logistic Regression*, *Decision Tree*, *Random Forest*, *XGBoost*, and *SVM*.

---

# Slide 4: Empirical Results & Performance

## 📊 Model Comparison & Best-in-Class Classifier

### Model Benchmark Summary (Holdout Test Set $N = 1,409$)

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC | 5-Fold CV F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Tuned)** 🏆 | **76.4%** | **0.54** | **0.75** | **0.63** | **0.842** | **0.637** |
| **XGBoost (Tuned)** | 74.9% | 0.52 | 0.80 | 0.63 | 0.843 | 0.635 |
| **Support Vector Machine (RBF)** | 74.9% | 0.52 | 0.77 | 0.62 | 0.824 | 0.621 |
| **Decision Tree (Baseline)** | 75.5% | 0.53 | 0.74 | 0.62 | 0.833 | 0.613 |
| **Logistic Regression** | 73.7% | 0.50 | 0.79 | 0.61 | 0.843 | 0.627 |

### Key Results Takeaway
- **Champion Model**: Tuned Random Forest selected via 5-fold CV F1 (**0.637**), achieving **ROC-AUC: 0.842** and **Recall: 74.9%** on test holdout.
- **Regularization Impact**: Overfitting gap reduced from 21.2% down to **4.8%** (`max_depth=8`, `min_samples_leaf=1`).
- **Cost-Sensitive Cutoff**: Optimal probability threshold of **0.44** unlocks **$31,550** in simulated net retention savings.
- **Explainability**: Integrated SHAP TreeExplainer enables instant per-customer probability decomposition.

---

# Slide 5: Strategic Business Conclusions

## 💡 Key Drivers & Commercial Action Plan

### 1. Primary Attrition Drivers
- **Contract Type**: Month-to-month contracts exhibit a **42.7% churn rate** vs **2.8%** on 2-year contracts.
- **First-Year Vulnerability**: Over **50% of all churn** takes place within the first 12 months.
- **Support Deficits**: Fiber optic users without tech support churn at more than double the rate of protected users.
- **Payment Method Friction**: Manual Electronic Check billing churn is **45.3%** vs **~16%** on automatic payments.

### 2. High-Impact Strategic Actions
- 🎁 **Early-Tenure Retention Incentives**: Target month-to-month customers in months 3–9 with promotional 1-year contract lock-ins.
- 🛡️ **Free Support Bundling**: Bundle introductory `TechSupport` and `OnlineSecurity` with high-speed Fiber Optic plans.
- 💳 **Auto-Pay Incentives**: Offer a one-off billing credit ($10) for activating automatic credit card / ACH payments.

### 3. Deployed Deliverable
- Interactive **Streamlit Application** ready for commercial staff to simulate risk scores and customize intervention thresholds in real time.
