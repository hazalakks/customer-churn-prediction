# Technical Project Report: Telecommunication Customer Churn Prediction

**Project Title**: End-to-End Telecom Customer Churn Prediction and Explainable AI Architecture  
**Author**: Hazal  
**Date**: October 2026  
**Status**: Completed / Production Ready  
**Repository**: [customer-churn-prediction](file:///C:/Users/hazal/.gemini/antigravity/scratch/customer-churn-prediction)

---

## 1. Problem Statement

Customer churn is an existential challenge for subscription-based telecommunication providers. Retaining an existing customer is well established to cost between **5 to 25 times less** than acquiring a new one. In a mature telecom market with high subscriber acquisition costs (SAC), even marginal reductions in monthly churn yield substantial gains in enterprise valuation and annual recurring revenue (ARR).

The primary objective of this project is to develop a predictive machine learning pipeline that:
1. Accurately identifies subscribers at imminent risk of cancelling service prior to their churn event.
2. Delivers granular, transparent explanations of individual risk factors using game-theoretic explainable AI (SHAP) to inform targeted customer success interventions.
3. Provides an interactive decision-support interface (Streamlit) for non-technical account managers to simulate pricing, service tier modifications, and retention incentives.

---

## 2. Dataset Overview

The project utilizes the benchmark **IBM Telco Customer Churn** dataset representing customer activity across California:

- **Volume & Dimensionality**: 7,043 customer records across 21 original attributes (20 predictive features and 1 binary target).
- **Target Variable**: `Churn` — indicating whether the customer terminated services within the last month (`Yes`: 1,869 accounts / 26.54%, `No`: 5,174 accounts / 73.46%).
- **Data Attributes**:
  - *Demographics*: `gender`, `SeniorCitizen`, `Partner`, `Dependents`.
  - *Account & Billing*: `tenure` (0–72 months), `Contract` (Month-to-month, One year, Two year), `PaperlessBilling`, `PaymentMethod`.
  - *Telephony & Connectivity*: `PhoneService`, `MultipleLines`, `InternetService` (DSL, Fiber optic, No).
  - *Value-Added Add-ons*: `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`.
  - *Financial Metrics*: `MonthlyCharges` ($18.25 – $118.75), `TotalCharges` ($18.80 – $8,684.80).

---

## 3. Methodology & Engineering Architecture

The modeling architecture adheres to strict production engineering standards, preventing data leakage and guaranteeing reproducible splits:

```
Raw Data (7,043 rows)
  │
  ├── 1. Data Sanitation: Strip empty strings, impute tenure=0 TotalCharges to 0.0, map Churn to {0, 1}
  ├── 2. Feature Engineering: 8 domain-specific signals generated
  ├── 3. Stratified Partition: 80% Train (5,634 rows) / 20% Test (1,409 rows)
  ├── 4. ColumnTransformer Pipeline:
  │      ├── Numerical -> StandardScaler
  │      └── Categorical -> OneHotEncoder(handle_unknown='ignore')
  ├── 5. Model Benchmarking: 5 Algorithms evaluated with Stratified 5-Fold Cross-Validation
  ├── 6. Hyperparameter Optimization: GridSearchCV optimizing ROC-AUC and F1-Score
  └── 7. Model Interpretation: SHAP TreeExplainer & LinearExplainer
```

### Domain Feature Engineering
Eight specialized features were constructed to capture nuanced behavioral signals:
1. **`AvgMonthlyCharge`**: $\text{TotalCharges} / \max(\text{tenure}, 1)$ — benchmarks historical spend velocity.
2. **`PriceDiscrepancy`**: $\text{MonthlyCharges} - \text{AvgMonthlyCharge}$ — flags recent rate hikes and bill shocks.
3. **`NumServices`**: Integer sum ($0 \dots 8$) of active subscriptions — assesses product stickiness.
4. **`UnprotectedInternet`**: Binary flag for internet users lacking both `OnlineSecurity` and `TechSupport`.
5. **`TenureCohort`**: Lifecycle segmentation (`New`: $<12$ mo, `Mid-term`: $12-36$ mo, `Long-term`: $>36$ mo).
6. **`HasPartnerOrDependents`**: Binary indicator of family obligations and household stability.
7. **`ContractScore`**: Ordinal mapping of contractual commitment ($0$ for month-to-month, $1$ for 1-year, $2$ for 2-year).
8. **`HasAutoPayment`**: Binary indicator for bank transfer or credit card automatic debiting.

---

## 4. Key Findings from Exploratory Data Analysis (EDA)

1. **Contractual Vulnerability**: Customers on month-to-month agreements exhibit a **42.7% churn rate**, in stark contrast to **11.3%** on 1-year contracts and **2.8%** on 2-year commitments. Contract length is the single most dominant structural retention safeguard.
2. **First-Year "Cliff"**: Over 53% of all customer churn events occur within the first **12 months of tenure** (`TenureCohort = 'New'`). Retention risk declines exponentially after customer longevity surpasses 24 months.
3. **Fiber Optic Service Paradox**: While Fiber Optic generates higher monthly revenue ($\sim\$90-\$110$), its churn rate is **41.9%** (vs. **19.0%** for DSL). Subscribers frequently encounter unassisted setup friction when technical support is absent.
4. **Payment Method Influence**: Subscribers paying via **Electronic Check** experience a churn rate of **45.3%**, more than triple that of automated payment methods (**15.2%** for bank transfer, **16.7%** for credit card).
5. **Add-on Protection Density**: Accounts with active `TechSupport` and `OnlineSecurity` display a churn rate below **15%**, whereas accounts lacking these services have churn rates exceeding **41%**, underscoring the retention power of multi-service bundling.

---

## 5. Model Results & Benchmarking

Five classification architectures were evaluated on an identical 20% holdout test partition ($N = 1,409$). Because of class imbalance (26.5% positive instances), models were evaluated using **ROC-AUC**, **F1-Score**, and **Precision-Recall balance** alongside standard accuracy.

| Model Architecture | Test Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) | ROC-AUC | 5-Fold CV F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Tuned)** 🏆 | **76.4%** | **0.54** | **0.75** | **0.63** | **0.842** | **0.637** |
| **XGBoost (Tuned)** | 74.9% | 0.52 | 0.80 | 0.63 | 0.843 | 0.635 |
| **Support Vector Machine (RBF)** | 74.9% | 0.52 | 0.77 | 0.62 | 0.824 | 0.621 |
| **Decision Tree (Baseline)** | 75.5% | 0.53 | 0.74 | 0.62 | 0.833 | 0.613 |
| **Logistic Regression** | 73.7% | 0.50 | 0.79 | 0.61 | 0.843 | 0.627 |

### Best Model Selection
**Random Forest (Tuned)** was selected as the primary production engine:
- Ranked highest in **Stratified 5-Fold Cross-Validation on the training set (CV F1: 0.637)**, ensuring generalizability without holdout test set peeking.
- Restricts overfitting through tree regularization (`max_depth=8`, `min_samples_leaf=1`, `n_estimators=200`), shrinking the train-to-test accuracy gap from 21.2% (default forest) down to **4.8%**.
- Achieves high retention sensitivity (**Recall: 74.9%**) and strong discrimination capacity (**ROC-AUC: 0.842**).
- Empirically calibrated via cost-benefit threshold modeling to an optimal cutoff of **0.44**, yielding **$31,550 net savings** over an unmitigated churn baseline.
- Features seamless compatibility with SHAP TreeExplainer, facilitating instantaneous inference-time interpretability within the deployed web app.

---

## 6. Conclusions & Future Work

### Strategic Business Recommendations
- **Incentivize Annual Migration**: Offer targeted discounts (e.g., 10-15% discount for a 12-month commitment) to month-to-month subscribers within their first 90 days.
- **Bundle Essential Security & Support**: Automatically bundle introductory `OnlineSecurity` and `TechSupport` with Fiber Optic subscriptions to eliminate early technical frustration.
- **Promote Automated Billing**: Implement small one-time billing credits ($5–$10) for subscribers migrating from Electronic Check to automated credit card/ACH billing.

### Technical Next Steps
- **Cost-Benefit Threshold Calibration**: Fine-tune decision thresholds dynamically based on individual customer Lifetime Value (LTV) rather than a static 0.5 cutoff.
- **Temporal & Survival Modeling**: Transition from binary classification to Cox Proportional Hazards or Random Survival Forests to predict *when* a customer is expected to churn.
- **Automated Retraining**: Implement CI/CD model monitoring to detect data drift and trigger automated model updates via scheduled retraining pipelines.
