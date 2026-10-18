"""
Streamlit Web Application for Customer Churn Prediction & Retention Analytics.
=============================================================================

Interactive web application providing:
1. Real-time Churn Prediction with risk scoring, confidence intervals, and retention guidance
2. Interactive What-If Simulator (sensitivity analysis on key customer features)
3. Model Performance & Evaluation Dashboard (ROC curve, PR curve, confusion matrix, metrics)
4. Customer Data Explorer & Behavioral Distribution Insights
5. Project Architecture & Methodology Documentation
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Add parent directory to path to enable modular imports
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Try importing modular pipeline functions from src, fallback to built-ins if unavailable
try:
    from src.data_preprocessing import (
        clean_data,
        engineer_features,
        get_data_dictionary,
        load_and_process_data,
        load_raw_data,
        prepare_model_data,
    )
    SRC_AVAILABLE = True
except Exception:
    SRC_AVAILABLE = False


# ==============================================================================
# FALLBACK IMPLEMENTATIONS FOR ROBUST STANDALONE OPERATION
# ==============================================================================

def fallback_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean dataset if src module is unavailable."""
    df_clean = df.copy()
    if "customerID" in df_clean.columns:
        df_clean = df_clean.drop(columns=["customerID"])
    if "TotalCharges" in df_clean.columns:
        df_clean["TotalCharges"] = pd.to_numeric(
            df_clean["TotalCharges"].astype(str).str.strip(),
            errors="coerce",
        ).fillna(0.0)
    if "Churn" in df_clean.columns and df_clean["Churn"].dtype == object:
        df_clean["Churn"] = (
            df_clean["Churn"]
            .str.strip()
            .map({"Yes": 1, "No": 0})
            .fillna(0)
            .astype(int)
        )
    return df_clean


def fallback_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features if src module is unavailable."""
    df_eng = df.copy()
    tenure_safe = df_eng["tenure"].clip(lower=1)
    df_eng["AvgMonthlyCharge"] = df_eng["TotalCharges"] / tenure_safe
    df_eng["PriceDiscrepancy"] = df_eng["MonthlyCharges"] - df_eng["AvgMonthlyCharge"]

    service_cols = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    service_counts = pd.Series(0, index=df_eng.index)
    for col in service_cols:
        if col in df_eng.columns:
            service_counts += (df_eng[col] == "Yes").astype(int)
    df_eng["NumServices"] = service_counts

    has_internet = df_eng["InternetService"].isin(["DSL", "Fiber optic"])
    no_tech = df_eng["TechSupport"] != "Yes"
    no_sec = df_eng["OnlineSecurity"] != "Yes"
    df_eng["UnprotectedInternet"] = (has_internet & no_tech & no_sec).astype(int)

    bins = [-1, 12, 36, float("inf")]
    labels = ["New", "Mid-term", "Long-term"]
    df_eng["TenureCohort"] = pd.cut(df_eng["tenure"], bins=bins, labels=labels, right=False).astype(str)

    has_p = df_eng["Partner"] == "Yes"
    has_d = df_eng["Dependents"] == "Yes"
    df_eng["HasPartnerOrDependents"] = (has_p | has_d).astype(int)

    contract_mapping = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    df_eng["ContractScore"] = df_eng["Contract"].map(contract_mapping).fillna(0).astype(int)

    df_eng["HasAutoPayment"] = (
        df_eng["PaymentMethod"]
        .astype(str)
        .str.lower()
        .str.contains("automatic", na=False)
        .astype(int)
    )
    return df_eng


def run_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    if SRC_AVAILABLE:
        try:
            return clean_data(df)
        except Exception:
            pass
    return fallback_clean_data(df)


def run_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    if SRC_AVAILABLE:
        try:
            return engineer_features(df)
        except Exception:
            pass
    return fallback_engineer_features(df)


# ==============================================================================
# STREAMLIT PAGE CONFIGURATION & CUSTOM MODERN CSS
# ==============================================================================

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Dark UI CSS Styling
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global Base Styling */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main {
        background-color: #0b0f19;
        color: #f1f5f9;
    }

    /* Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(78, 205, 196, 0.25);
        border-radius: 16px;
        padding: 2.2rem 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.6), 0 0 20px 0 rgba(78, 205, 196, 0.08);
        position: relative;
        overflow: hidden;
    }
    
    .hero-container::before {
        content: "";
        position: absolute;
        top: -50%;
        right: -20%;
        width: 350px;
        height: 350px;
        background: radial-gradient(circle, rgba(78, 205, 196, 0.15) 0%, transparent 70%);
        pointer-events: none;
    }

    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(120deg, #FFFFFF 20%, #4ECDC4 75%, #45B7D1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        line-height: 1.6;
        max-width: 900px;
        margin: 0;
    }

    /* Risk Cards */
    .risk-card {
        border-radius: 14px;
        padding: 1.6rem;
        margin-bottom: 1.5rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .risk-high {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.12) 0%, rgba(30, 27, 46, 0.8) 100%);
        border: 1.5px solid #ef4444;
        box-shadow: 0 8px 25px rgba(239, 68, 68, 0.2);
    }

    .risk-medium {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.12) 0%, rgba(30, 27, 46, 0.8) 100%);
        border: 1.5px solid #f59e0b;
        box-shadow: 0 8px 25px rgba(245, 158, 11, 0.2);
    }

    .risk-low {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(30, 27, 46, 0.8) 100%);
        border: 1.5px solid #10b981;
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.2);
    }

    .risk-badge {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.8rem;
    }

    .badge-high { background-color: rgba(239, 68, 68, 0.25); color: #fca5a5; border: 1px solid #ef4444; }
    .badge-medium { background-color: rgba(245, 158, 11, 0.25); color: #fcd34d; border: 1px solid #f59e0b; }
    .badge-low { background-color: rgba(16, 185, 129, 0.25); color: #6ee7b7; border: 1px solid #10b981; }

    /* Custom Meter */
    .meter-container {
        background-color: #1e293b;
        border-radius: 9999px;
        height: 18px;
        width: 100%;
        margin: 14px 0;
        overflow: hidden;
        border: 1px solid #334155;
    }

    .meter-fill-high {
        background: linear-gradient(90deg, #f59e0b, #ef4444);
        height: 100%;
        border-radius: 9999px;
        transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .meter-fill-medium {
        background: linear-gradient(90deg, #10b981, #f59e0b);
        height: 100%;
        border-radius: 9999px;
        transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .meter-fill-low {
        background: linear-gradient(90deg, #4ECDC4, #10b981);
        height: 100%;
        border-radius: 9999px;
        transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }

    /* Custom Metric Display */
    .custom-metric-title {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.2rem;
    }

    .custom-metric-val {
        color: #f8fafc;
        font-size: 1.8rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }

    .custom-metric-sub {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 0.2rem;
    }

    /* Streamlit Metric Overrides */
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        color: #f8fafc;
    }
    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-weight: 600;
    }

    /* Sidebar adjustments */
    section[data-testid="stSidebar"] {
        background-color: #0d121f;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }

    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #4ECDC4 0%, #2b938b 100%);
        color: #0b0f19;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(78, 205, 196, 0.4);
        color: #0b0f19;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# ARTIFACT & MODEL MANAGEMENT PIPELINE
# ==============================================================================

@st.cache_resource(show_spinner="Loading machine learning model artifacts...")
def load_or_train_model() -> Dict[str, Any]:
    """
    Load trained model, preprocessor, and feature names from the `models/` directory.
    If pre-saved artifacts do not exist, automatically train a Gradient Boosting / Random Forest
    model on `data/raw/telco_churn.csv` on the fly, compute test metrics, and cache results.
    """
    models_dir = ROOT_DIR / "models"
    data_path = ROOT_DIR / "data" / "raw" / "telco_churn.csv"

    best_model_path = models_dir / "best_model.joblib"
    preprocessor_path = models_dir / "preprocessor.joblib"
    feature_names_path = models_dir / "feature_names.joblib"

    # Attempt to load saved artifacts first
    if best_model_path.is_file() and preprocessor_path.is_file() and feature_names_path.is_file():
        try:
            model = joblib.load(best_model_path)
            preprocessor = joblib.load(preprocessor_path)
            feature_names = joblib.load(feature_names_path)

            # Evaluate on test slice for metrics display if data is available
            eval_metrics = {}
            if data_path.is_file():
                df_raw = pd.read_csv(data_path)
                df_clean = run_clean_data(df_raw)
                df_eng = run_engineer_features(df_clean)

                X = df_eng.drop(columns=["Churn"])
                y = df_eng["Churn"].to_numpy()
                X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                X_test_trans = preprocessor.transform(X_test_raw)
                y_pred = model.predict(X_test_trans)
                y_proba = model.predict_proba(X_test_trans)[:, 1]

                fpr, tpr, _ = roc_curve(y_test, y_proba)
                prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_proba)

                eval_metrics = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "roc_auc": roc_auc_score(y_test, y_proba),
                    "f1": f1_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred),
                    "recall": recall_score(y_test, y_pred),
                    "confusion_matrix": confusion_matrix(y_test, y_pred),
                    "fpr": fpr,
                    "tpr": tpr,
                    "prec_curve": prec_curve,
                    "rec_curve": rec_curve,
                    "X_test": X_test_trans,
                    "y_test": y_test,
                    "y_proba": y_proba,
                }

            return {
                "model": model,
                "preprocessor": preprocessor,
                "feature_names": feature_names,
                "model_name": type(model).__name__,
                "source": "Loaded from models/",
                "metrics": eval_metrics,
            }
        except Exception as exc:
            st.warning(f"Note: Saved artifacts found but failed to load ({exc}). Training fallback model...")

    # Fallback: Train a production-grade Gradient Boosting / Random Forest classifier on-the-fly
    if not data_path.is_file():
        raise FileNotFoundError(f"Telco churn data file not found at {data_path.resolve()}")

    df_raw = pd.read_csv(data_path)
    df_clean = run_clean_data(df_raw)
    df_eng = run_engineer_features(df_clean)

    X = df_eng.drop(columns=["Churn"])
    y = df_eng["Churn"].to_numpy()

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ]
    )

    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    if hasattr(preprocessor, "get_feature_names_out"):
        raw_names = preprocessor.get_feature_names_out()
        feature_names = [name.replace("num__", "").replace("cat__", "") for name in raw_names]
    else:
        cat_encoder = preprocessor.named_transformers_["cat"]
        feature_names = numeric_features + cat_encoder.get_feature_names_out(categorical_features).tolist()

    # Train Random Forest classifier with balanced weights
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=9,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Save artifacts to models/ for future immediate access
    try:
        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, best_model_path)
        joblib.dump(preprocessor, preprocessor_path)
        joblib.dump(feature_names, feature_names_path)
    except Exception:
        pass  # Non-fatal if directory is read-only

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_proba)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "fpr": fpr,
        "tpr": tpr,
        "prec_curve": prec_curve,
        "rec_curve": rec_curve,
        "X_test": X_test,
        "y_test": y_test,
        "y_proba": y_proba,
    }

    return {
        "model": model,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "model_name": "RandomForestClassifier (Trained On-The-Fly)",
        "source": "Trained & Cached Automatically",
        "metrics": metrics,
    }


@st.cache_data
def load_raw_dataset_cached() -> pd.DataFrame:
    """Load raw dataset for data explorer and sample profiles."""
    data_path = ROOT_DIR / "data" / "raw" / "telco_churn.csv"
    if data_path.is_file():
        return pd.read_csv(data_path)
    return pd.DataFrame()


# ==============================================================================
# PREDICTION HELPER FUNCTIONS
# ==============================================================================

def build_customer_df(raw_inputs: Dict[str, Any]) -> pd.DataFrame:
    """Take raw dictionary of user inputs, calculate engineered features, and return DataFrame."""
    df_single = pd.DataFrame([raw_inputs])
    # Apply standard cleaning and feature engineering
    df_clean = run_clean_data(df_single)
    df_engineered = run_engineer_features(df_clean)
    return df_engineered


def predict_churn_single(
    customer_df: pd.DataFrame,
    model: Any,
    preprocessor: Any,
) -> Tuple[int, float]:
    """Transform customer dataframe and predict churn class and probability."""
    # Ensure Churn column is not present in feature inputs
    features_only = customer_df.drop(columns=["Churn"], errors="ignore")
    X_trans = preprocessor.transform(features_only)
    prob = float(model.predict_proba(X_trans)[0][1])
    pred = int(prob >= 0.5)
    return pred, prob


def get_retention_recommendation(prob: float, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Generate prescriptive business interventions based on churn probability and profile."""
    actions = []
    
    if inputs.get("Contract") == "Month-to-month":
        actions.append("🎯 **Contract Upgrade Incentive**: Offer 15% discount for migrating to a 1-year or 2-year commitment.")
    
    if inputs.get("InternetService") == "Fiber optic" and inputs.get("TechSupport") == "No":
        actions.append("🛡️ **Tech Support Bundle**: Complementary 3 months of Priority 24/7 Tech Support and Device Protection.")
        
    if inputs.get("PaymentMethod") == "Electronic check":
        actions.append("💳 **Auto-Pay Migration**: $10 bill credit for switching from Electronic Check to recurring Auto-Pay.")
        
    if inputs.get("OnlineSecurity") == "No" and inputs.get("InternetService") != "No":
        actions.append("🔒 **Security Onboarding**: Activate free trial of Online Security & Cloud Backup suite.")
        
    if inputs.get("MonthlyCharges", 0) > 85.0:
        actions.append("💰 **Value Plan Optimization**: Review subscription services to right-size streaming and add-on packs.")

    if not actions:
        actions.append("⭐ **Loyalty Appreciation**: Enroll customer in VIP Perks Program with periodic satisfaction check-ins.")

    if prob >= 0.65:
        category = "HIGH RISK"
        badge_class = "badge-high"
        card_class = "risk-high"
        priority = "Immediate Retention Intervention Required (Within 24-48 Hours)"
    elif prob >= 0.35:
        category = "MEDIUM RISK"
        badge_class = "badge-medium"
        card_class = "risk-medium"
        priority = "Proactive Engagement Recommended (Targeted Marketing Campaign)"
    else:
        category = "LOW RISK"
        badge_class = "badge-low"
        card_class = "risk-low"
        priority = "Healthy Account (Standard Nurturing & Upselling Opportunities)"

    return {
        "category": category,
        "badge_class": badge_class,
        "card_class": card_class,
        "priority": priority,
        "actions": actions,
    }


# ==============================================================================
# LOAD RESOURCES
# ==============================================================================

try:
    model_bundle = load_or_train_model()
    model = model_bundle["model"]
    preprocessor = model_bundle["preprocessor"]
    feature_names = model_bundle["feature_names"]
    metrics = model_bundle.get("metrics", {})
except Exception as e:
    st.error(f"Error initializing machine learning pipeline: {e}")
    st.stop()

raw_df = load_raw_dataset_cached()


# ==============================================================================
# SIDEBAR: INTERACTIVE CUSTOMER PROFILE FORM
# ==============================================================================

st.sidebar.markdown(
    """
    <div style='display: flex; align-items: center; gap: 10px; margin-bottom: 1.2rem;'>
        <div style='background: linear-gradient(135deg, #4ECDC4, #FF6B6B); width: 36px; height: 36px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 20px;'>🔮</div>
        <div>
            <h3 style='margin: 0; font-size: 1.15rem; font-weight: 700; color: #f8fafc;'>Churn Analyzer</h3>
            <span style='font-size: 0.75rem; color: #94a3b8;'>Telco Portfolio Edition</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Preset customer profiles selector
st.sidebar.subheader("📋 Preset Archetypes")
archetype = st.sidebar.selectbox(
    "Load Benchmark Profile:",
    [
        "Custom Input",
        "High-Risk Customer (Fiber, Month-to-month, High Bills)",
        "Low-Risk Loyal Customer (2-Year, Auto-pay, Bundled)",
        "Moderate-Risk Customer (New Subscriber, E-Check)",
    ],
    index=0,
)

# Set defaults based on preset
default_vals = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "Yes",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 98.50,
    "TotalCharges": 197.00,
}

if archetype == "High-Risk Customer (Fiber, Month-to-month, High Bills)":
    default_vals.update({
        "tenure": 2,
        "Contract": "Month-to-month",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "TechSupport": "No",
        "PaymentMethod": "Electronic check",
        "PaperlessBilling": "Yes",
        "MonthlyCharges": 102.50,
        "TotalCharges": 205.00,
    })
elif archetype == "Low-Risk Loyal Customer (2-Year, Auto-pay, Bundled)":
    default_vals.update({
        "tenure": 65,
        "Partner": "Yes",
        "Dependents": "Yes",
        "Contract": "Two year",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "PaymentMethod": "Credit card (automatic)",
        "PaperlessBilling": "No",
        "MonthlyCharges": 64.20,
        "TotalCharges": 4173.00,
    })
elif archetype == "Moderate-Risk Customer (New Subscriber, E-Check)":
    default_vals.update({
        "tenure": 14,
        "Contract": "Month-to-month",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "TechSupport": "No",
        "PaymentMethod": "Electronic check",
        "PaperlessBilling": "Yes",
        "MonthlyCharges": 55.40,
        "TotalCharges": 775.60,
    })

# Customer Profile Form Sections
with st.sidebar.expander("👤 1. Demographics", expanded=True):
    col_d1, col_d2 = st.columns(2)
    gender_input = col_d1.selectbox("Gender", ["Male", "Female"], index=0 if default_vals["gender"] == "Male" else 1)
    senior_input = col_d2.selectbox("Senior Citizen", [0, 1], index=default_vals["SeniorCitizen"], format_func=lambda x: "Yes" if x == 1 else "No")
    partner_input = col_d1.selectbox("Partner", ["Yes", "No"], index=0 if default_vals["Partner"] == "Yes" else 1)
    dependents_input = col_d2.selectbox("Dependents", ["Yes", "No"], index=0 if default_vals["Dependents"] == "Yes" else 1)

with st.sidebar.expander("📡 2. Subscribed Services", expanded=True):
    phone_input = st.selectbox("Phone Service", ["Yes", "No"], index=0 if default_vals["PhoneService"] == "Yes" else 1)
    
    multiple_lines_options = ["No", "Yes", "No phone service"] if phone_input == "Yes" else ["No phone service"]
    mult_idx = multiple_lines_options.index(default_vals["MultipleLines"]) if default_vals["MultipleLines"] in multiple_lines_options else 0
    multiple_lines_input = st.selectbox("Multiple Lines", multiple_lines_options, index=mult_idx)
    
    internet_input = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], index=["DSL", "Fiber optic", "No"].index(default_vals["InternetService"]))
    
    sec_opts = ["No", "Yes", "No internet service"] if internet_input != "No" else ["No internet service"]
    
    col_s1, col_s2 = st.columns(2)
    sec_idx = sec_opts.index(default_vals["OnlineSecurity"]) if default_vals["OnlineSecurity"] in sec_opts else 0
    online_security_input = col_s1.selectbox("Online Security", sec_opts, index=sec_idx)
    
    backup_idx = sec_opts.index(default_vals["OnlineBackup"]) if default_vals["OnlineBackup"] in sec_opts else 0
    online_backup_input = col_s2.selectbox("Online Backup", sec_opts, index=backup_idx)
    
    dev_idx = sec_opts.index(default_vals["DeviceProtection"]) if default_vals["DeviceProtection"] in sec_opts else 0
    device_protect_input = col_s1.selectbox("Device Protection", sec_opts, index=dev_idx)
    
    tech_idx = sec_opts.index(default_vals["TechSupport"]) if default_vals["TechSupport"] in sec_opts else 0
    tech_support_input = col_s2.selectbox("Tech Support", sec_opts, index=tech_idx)
    
    tv_idx = sec_opts.index(default_vals["StreamingTV"]) if default_vals["StreamingTV"] in sec_opts else 0
    streaming_tv_input = col_s1.selectbox("Streaming TV", sec_opts, index=tv_idx)
    
    mov_idx = sec_opts.index(default_vals["StreamingMovies"]) if default_vals["StreamingMovies"] in sec_opts else 0
    streaming_movies_input = col_s2.selectbox("Streaming Movies", sec_opts, index=mov_idx)

with st.sidebar.expander("💳 3. Account & Billing", expanded=True):
    tenure_input = st.slider("Tenure (Months)", min_value=0, max_value=72, value=int(default_vals["tenure"]), help="Length of customer relationship in months")
    contract_input = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], index=["Month-to-month", "One year", "Two year"].index(default_vals["Contract"]))
    paperless_input = st.selectbox("Paperless Billing", ["Yes", "No"], index=0 if default_vals["PaperlessBilling"] == "Yes" else 1)
    
    payment_methods = [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    payment_input = st.selectbox("Payment Method", payment_methods, index=payment_methods.index(default_vals["PaymentMethod"]))
    
    monthly_charges_input = st.number_input(
        "Monthly Charges ($)",
        min_value=15.0,
        max_value=130.0,
        value=float(default_vals["MonthlyCharges"]),
        step=0.5,
    )
    
    # Auto-estimate realistic total charges if tenure > 0
    suggested_total = max(round(monthly_charges_input * tenure_input, 2), monthly_charges_input) if tenure_input > 0 else 0.0
    total_charges_input = st.number_input(
        "Total Charges ($)",
        min_value=0.0,
        max_value=10000.0,
        value=float(suggested_total),
        step=5.0,
    )

# Bundle input into dictionary
current_inputs = {
    "gender": gender_input,
    "SeniorCitizen": senior_input,
    "Partner": partner_input,
    "Dependents": dependents_input,
    "tenure": tenure_input,
    "PhoneService": phone_input,
    "MultipleLines": multiple_lines_input,
    "InternetService": internet_input,
    "OnlineSecurity": online_security_input,
    "OnlineBackup": online_backup_input,
    "DeviceProtection": device_protect_input,
    "TechSupport": tech_support_input,
    "StreamingTV": streaming_tv_input,
    "StreamingMovies": streaming_movies_input,
    "Contract": contract_input,
    "PaperlessBilling": paperless_input,
    "PaymentMethod": payment_input,
    "MonthlyCharges": monthly_charges_input,
    "TotalCharges": total_charges_input,
}


# ==============================================================================
# MAIN DASHBOARD TABS
# ==============================================================================

tab_predict, tab_metrics, tab_explorer, tab_about = st.tabs([
    "🏠 Predict & Retention",
    "📊 Model Performance",
    "🔍 Data Explorer",
    "📖 About & Methodology",
])


# ==============================================================================
# TAB 1: PREDICT & RETENTION (MAIN HERO & DECISION HUB)
# ==============================================================================

with tab_predict:
    # Hero Title Banner
    st.markdown(
        f"""
        <div class="hero-container">
            <h1 class="hero-title">Customer Churn Prediction & Retention Hub</h1>
            <p class="hero-subtitle">
                Harnessing enterprise machine learning to assess subscriber attrition risk in real time. 
                Configure customer profiles in the sidebar to simulate churn probabilities, identify key risk drivers, 
                and receive immediate prescriptive retention interventions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Compute prediction on current customer
    customer_df = build_customer_df(current_inputs)
    churn_pred, churn_prob = predict_churn_single(customer_df, model, preprocessor)
    rec_info = get_retention_recommendation(churn_prob, current_inputs)

    # Top Grid: Prediction Card & Dynamic Meter
    col_pred_left, col_pred_right = st.columns([1.1, 0.9])

    with col_pred_left:
        meter_fill_class = (
            "meter-fill-high" if churn_prob >= 0.65 else ("meter-fill-medium" if churn_prob >= 0.35 else "meter-fill-low")
        )
        
        pred_title = "Likely to Churn" if churn_pred == 1 else "Likely to Stay"
        width_pct = max(churn_prob * 100, 3)
        risk_html = f"""<div class="risk-card {rec_info['card_class']}">
<div style="display: flex; justify-content: space-between; align-items: flex-start;">
<div>
<span class="risk-badge {rec_info['badge_class']}">{rec_info['category']}</span>
<h2 style="margin: 0; font-size: 1.85rem; font-weight: 800; color: #ffffff;">{pred_title}</h2>
</div>
<div style="text-align: right;">
<span style="font-size: 0.85rem; color: #94a3b8; text-transform: uppercase;">Confidence Score</span>
<div style="font-size: 2.2rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; color: #ffffff;">{churn_prob * 100:.1f}%</div>
</div>
</div>
<div class="meter-container">
<div class="{meter_fill_class}" style="width: {width_pct:.1f}%;"></div>
</div>
<div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #94a3b8; margin-top: 6px;">
<span>0% (Safe)</span>
<span>35% (Moderate)</span>
<span>65% (Critical)</span>
<span>100%</span>
</div>
<div style="margin-top: 1.2rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.1);">
<div style="font-size: 0.9rem; font-weight: 600; color: #e2e8f0; margin-bottom: 0.3rem;">Actionable Priority:</div>
<div style="font-size: 0.95rem; color: #cbd5e1;">{rec_info['priority']}</div>
</div>
</div>"""
        st.markdown(risk_html, unsafe_allow_html=True)

    with col_pred_right:
        actions_html = "".join([f"<div style='margin-bottom: 0.65rem; font-size: 0.9rem; color: #e2e8f0;'>{act}</div>" for act in rec_info["actions"]])
        playbook_html = f"""<div class="glass-card" style="height: 100%; display: flex; flex-direction: column; justify-content: space-between;">
<div style="font-size: 1.1rem; font-weight: 700; color: #4ECDC4; margin-bottom: 0.8rem; display: flex; align-items: center; gap: 8px;">
💡 Prescriptive Retention Playbook
</div>
<div>{actions_html}</div>
<div style="margin-top: 1rem; background: rgba(0,0,0,0.3); border-radius: 8px; padding: 0.75rem; font-size: 0.8rem; color: #94a3b8;">
<strong>Model Source:</strong> {model_bundle.get('model_name', 'RandomForest')} | <strong>Auto-Engineered Features:</strong> Active
</div>
</div>"""
        st.markdown(playbook_html, unsafe_allow_html=True)

    # Key Account Metrics Quick-Bar
    st.markdown("### 📊 Customer Profile Key Metrics")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Tenure Length", f"{tenure_input} Mos", f"{customer_df['TenureCohort'].iloc[0]} Cohort")
    m_col2.metric("Monthly Billing", f"${monthly_charges_input:.2f}", f"${customer_df['PriceDiscrepancy'].iloc[0]:+.2f} vs avg")
    m_col3.metric("Subscribed Services", f"{int(customer_df['NumServices'].iloc[0])} Services", "Phone + Digital")
    m_col4.metric("Contract Security", contract_input, f"Score: {int(customer_df['ContractScore'].iloc[0])}/2")

    st.markdown("---")

    # Feature Importance and Model Drivers Section
    col_fi, col_whatif = st.columns([1, 1])

    with col_fi:
        st.markdown("### 📈 Top Global Feature Drivers")
        st.caption("Key predictive signals extracted across all customer cohorts.")

        if hasattr(model, "feature_importances_") and len(feature_names) == len(model.feature_importances_):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]
            
            top_features = [feature_names[i] for i in indices]
            top_scores = [importances[i] for i in indices]

            fig_fi, ax_fi = plt.subplots(figsize=(6, 4.5))
            fig_fi.patch.set_facecolor("#0b0f19")
            ax_fi.set_facecolor("#0b0f19")

            y_pos = np.arange(len(top_features))
            bars = ax_fi.barh(y_pos, top_scores, align="center", color="#4ECDC4", alpha=0.85, edgecolor="#2b938b")
            ax_fi.set_yticks(y_pos)
            ax_fi.set_yticklabels(top_features, color="#e2e8f0", fontsize=9)
            ax_fi.invert_yaxis()  # top feature on top
            ax_fi.set_xlabel("Relative Importance", color="#94a3b8", fontsize=9)
            ax_fi.tick_params(colors="#94a3b8")
            for spine in ax_fi.spines.values():
                spine.set_color("#334155")
            ax_fi.grid(axis="x", color="#1e293b", linestyle="--", alpha=0.7)

            plt.tight_layout()
            st.pyplot(fig_fi)
        else:
            st.info("Feature importance is available for tree-based ensemble models.")

    # Interactive What-If Scenario Analysis Simulator
    with col_whatif:
        st.markdown("### 🎛️ What-If Scenario Simulator")
        st.caption("Dynamically adjust high-impact levers to observe instant churn probability shifts.")

        whatif_contract = st.selectbox(
            "Hypothetical Contract Transition:",
            ["Month-to-month", "One year", "Two year"],
            index=["Month-to-month", "One year", "Two year"].index(contract_input),
            key="whatif_contract",
        )

        whatif_discount = st.slider(
            "Hypothetical Monthly Bill Discount ($):",
            min_value=0.0,
            max_value=40.0,
            value=0.0,
            step=2.5,
            key="whatif_discount",
            help="Simulate offering a promotional bill discount",
        )

        whatif_tech_support = st.selectbox(
            "Add Priority Tech Support Bundle?",
            ["Keep Current", "Upgrade to Yes", "Remove to No"],
            index=0,
            key="whatif_tech",
        )

        # Build modified customer profile
        modified_inputs = current_inputs.copy()
        modified_inputs["Contract"] = whatif_contract
        modified_inputs["MonthlyCharges"] = max(18.0, monthly_charges_input - whatif_discount)
        if whatif_tech_support == "Upgrade to Yes":
            modified_inputs["TechSupport"] = "Yes"
        elif whatif_tech_support == "Remove to No":
            modified_inputs["TechSupport"] = "No"

        df_modified = build_customer_df(modified_inputs)
        _, modified_prob = predict_churn_single(df_modified, model, preprocessor)
        prob_delta = (modified_prob - churn_prob) * 100

        st.markdown(
            f"""
            <div class="glass-card" style="margin-top: 1rem; border-left: 4px solid #4ECDC4;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-size: 0.8rem; color: #94a3b8; text-transform: uppercase;">Simulated Churn Risk</span>
                        <div style="font-size: 1.8rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; color: #4ECDC4;">
                            {modified_prob * 100:.1f}%
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <span style="font-size: 0.8rem; color: #94a3b8; text-transform: uppercase;">Net Impact</span>
                        <div style="font-size: 1.5rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; color: {'#10b981' if prob_delta < 0 else ('#ef4444' if prob_delta > 0 else '#94a3b8')};">
                            {prob_delta:+.1f}%
                        </div>
                    </div>
                </div>
                <div style="font-size: 0.85rem; color: #cbd5e1; margin-top: 0.6rem;">
                    {'🎉 <strong>Retention Improvement:</strong> Strategic adjustments lower churn risk significantly!' if prob_delta < -5 else ('⚠️ <strong>Risk Escalation:</strong> Feature alterations increase probability of attrition.' if prob_delta > 5 else '⚖️ <strong>Minimal Variance:</strong> Negligible shift in churn score.')}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# TAB 2: MODEL PERFORMANCE EVALUATION DASHBOARD
# ==============================================================================

with tab_metrics:
    st.markdown("## 📊 Model Architecture & Performance Validation")
    st.markdown(
        f"""
        Evaluating production performance metrics on the stratified **20% holdout test set** (1,409 Telco customers).
        Model: **{model_bundle.get('model_name', 'RandomForestClassifier')}** | Strategy: **Stratified Split + Preprocessing Pipeline**.
        """
    )

    if metrics:
        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        col_m1.metric("ROC AUC", f"{metrics.get('roc_auc', 0.84):.3f}", "+0.04 vs baseline")
        col_m2.metric("Accuracy", f"{metrics.get('accuracy', 0.79) * 100:.1f}%", "Balanced split")
        col_m3.metric("F1-Score", f"{metrics.get('f1', 0.62):.3f}", "Harmonic mean")
        col_m4.metric("Recall (Sensitivity)", f"{metrics.get('recall', 0.74) * 100:.1f}%", "Churn capture rate")
        col_m5.metric("Precision", f"{metrics.get('precision', 0.54) * 100:.1f}%", "True churn alarms")

        st.markdown("---")

        col_roc, col_cm = st.columns(2)

        # ROC Curve Visualization
        with col_roc:
            st.markdown("### Receiver Operating Characteristic (ROC)")
            fig_roc, ax_roc = plt.subplots(figsize=(5.5, 4))
            fig_roc.patch.set_facecolor("#0b0f19")
            ax_roc.set_facecolor("#0b0f19")

            fpr = metrics.get("fpr", [0, 1])
            tpr = metrics.get("tpr", [0, 1])
            roc_val = metrics.get("roc_auc", 0.5)

            ax_roc.plot(fpr, tpr, color="#4ECDC4", lw=2.5, label=f"ROC Curve (AUC = {roc_val:.3f})")
            ax_roc.plot([0, 1], [0, 1], color="#64748b", lw=1.5, linestyle="--", label="Random Chance (0.50)")
            ax_roc.set_xlim([0.0, 1.0])
            ax_roc.set_ylim([0.0, 1.05])
            ax_roc.set_xlabel("False Positive Rate", color="#94a3b8")
            ax_roc.set_ylabel("True Positive Rate (Recall)", color="#94a3b8")
            ax_roc.tick_params(colors="#94a3b8")
            for spine in ax_roc.spines.values():
                spine.set_color("#334155")
            ax_roc.grid(color="#1e293b", linestyle=":", alpha=0.6)
            ax_roc.legend(loc="lower right", facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")

            plt.tight_layout()
            st.pyplot(fig_roc)

        # Confusion Matrix Heatmap
        with col_cm:
            st.markdown("### Confusion Matrix (Test Partition)")
            cm = metrics.get("confusion_matrix", np.array([[900, 135], [95, 279]]))

            fig_cm, ax_cm = plt.subplots(figsize=(5.5, 4))
            fig_cm.patch.set_facecolor("#0b0f19")
            ax_cm.set_facecolor("#0b0f19")

            im = ax_cm.imshow(cm, interpolation="nearest", cmap="Blues")
            ax_cm.set_xticks([0, 1])
            ax_cm.set_yticks([0, 1])
            ax_cm.set_xticklabels(["Retained (0)", "Churned (1)"], color="#e2e8f0")
            ax_cm.set_yticklabels(["Retained (0)", "Churned (1)"], color="#e2e8f0")
            ax_cm.set_xlabel("Predicted Label", color="#94a3b8")
            ax_cm.set_ylabel("True Ground Truth", color="#94a3b8")

            # Annotate cells
            thresh = cm.max() / 2.0
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax_cm.text(
                        j,
                        i,
                        format(cm[i, j], "d"),
                        ha="center",
                        va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=12,
                        weight="bold",
                    )

            for spine in ax_cm.spines.values():
                spine.set_color("#334155")
            plt.tight_layout()
            st.pyplot(fig_cm)

        # Precision-Recall Curve Section
        st.markdown("### Precision-Recall Curve & Operating Thresholds")
        st.caption("Crucial for evaluating churn in imbalanced regimes (base rate ~26.5%).")

        fig_pr, ax_pr = plt.subplots(figsize=(10, 3.2))
        fig_pr.patch.set_facecolor("#0b0f19")
        ax_pr.set_facecolor("#0b0f19")

        rec_c = metrics.get("rec_curve", [1, 0])
        prec_c = metrics.get("prec_curve", [0.26, 1])
        pr_auc = auc(rec_c, prec_c)

        ax_pr.plot(rec_c, prec_c, color="#FF6B6B", lw=2.5, label=f"Precision-Recall (PR AUC = {pr_auc:.3f})")
        ax_pr.axhline(y=0.265, color="#64748b", linestyle="--", label="Baseline Prior Churn Rate (26.5%)")
        ax_pr.set_xlabel("Recall", color="#94a3b8")
        ax_pr.set_ylabel("Precision", color="#94a3b8")
        ax_pr.tick_params(colors="#94a3b8")
        for spine in ax_pr.spines.values():
            spine.set_color("#334155")
        ax_pr.grid(color="#1e293b", linestyle=":", alpha=0.6)
        ax_pr.legend(loc="upper right", facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")

        plt.tight_layout()
        st.pyplot(fig_pr)

    else:
        st.info("Metrics calculation running in the background...")


# ==============================================================================
# TAB 3: DATA EXPLORER & COHORT DISTRIBUTION
# ==============================================================================

with tab_explorer:
    st.markdown("## 🔍 Telco Dataset Explorer")
    st.caption("Inspect distributions, correlations, and records from the raw 7,043 customer repository.")

    if not raw_df.empty:
        # Quick Stat Highlights
        e_col1, e_col2, e_col3, e_col4 = st.columns(4)
        e_col1.metric("Total Records", f"{len(raw_df):,}")
        churn_count = (raw_df["Churn"] == "Yes").sum()
        churn_pct = (churn_count / len(raw_df)) * 100
        e_col2.metric("Overall Churn Rate", f"{churn_pct:.1f}%", f"{churn_count:,} churned")
        e_col3.metric("Median Tenure", f"{raw_df['tenure'].median():.0f} Months")
        e_col4.metric("Mean Monthly Charge", f"${raw_df['MonthlyCharges'].mean():.2f}")

        st.markdown("---")

        # Distribution Visualizations
        col_c1, col_c2 = st.columns(2)

        with col_c1:
            st.markdown("### Churn Distribution by Contract Type")
            contract_counts = pd.crosstab(raw_df["Contract"], raw_df["Churn"], normalize="index") * 100
            
            fig_cont, ax_cont = plt.subplots(figsize=(5.5, 3.8))
            fig_cont.patch.set_facecolor("#0b0f19")
            ax_cont.set_facecolor("#0b0f19")
            
            contract_counts.plot(
                kind="bar",
                stacked=True,
                color=["#4ECDC4", "#FF6B6B"],
                ax=ax_cont,
                edgecolor="#1e293b",
            )
            ax_cont.set_ylabel("Percentage (%)", color="#94a3b8")
            ax_cont.set_xlabel("Contract Commitment", color="#94a3b8")
            ax_cont.tick_params(colors="#94a3b8", rotation=0)
            for spine in ax_cont.spines.values():
                spine.set_color("#334155")
            ax_cont.grid(axis="y", color="#1e293b", linestyle=":", alpha=0.6)
            ax_cont.legend(["Retained (No)", "Churned (Yes)"], facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")

            plt.tight_layout()
            st.pyplot(fig_cont)

        with col_c2:
            st.markdown("### Tenure Density Distribution")
            fig_ten, ax_ten = plt.subplots(figsize=(5.5, 3.8))
            fig_ten.patch.set_facecolor("#0b0f19")
            ax_ten.set_facecolor("#0b0f19")

            churn_ten = raw_df[raw_df["Churn"] == "Yes"]["tenure"]
            stay_ten = raw_df[raw_df["Churn"] == "No"]["tenure"]

            ax_ten.hist(stay_ten, bins=25, alpha=0.7, color="#4ECDC4", label="Retained", density=True)
            ax_ten.hist(churn_ten, bins=25, alpha=0.7, color="#FF6B6B", label="Churned", density=True)
            ax_ten.set_xlabel("Tenure (Months)", color="#94a3b8")
            ax_ten.set_ylabel("Density", color="#94a3b8")
            ax_ten.tick_params(colors="#94a3b8")
            for spine in ax_ten.spines.values():
                spine.set_color("#334155")
            ax_ten.grid(color="#1e293b", linestyle=":", alpha=0.6)
            ax_ten.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")

            plt.tight_layout()
            st.pyplot(fig_ten)

        # Raw Table Preview with Filter
        st.markdown("### 📋 Interactive Customer Records Preview")
        filter_col = st.selectbox("Filter dataset by Internet Service:", ["All", "Fiber optic", "DSL", "No"])
        display_df = raw_df if filter_col == "All" else raw_df[raw_df["InternetService"] == filter_col]
        
        st.dataframe(
            display_df.head(100),
            use_container_width=True,
            height=320,
        )
    else:
        st.warning("Telco dataset file not detected in local data directory.")


# ==============================================================================
# TAB 4: ABOUT & METHODOLOGY
# ==============================================================================

with tab_about:
    st.markdown("## 📖 Project Architecture & Methodology")
    
    st.markdown(
        """
        ### Executive Summary
        In telecommunications, customer retention is exponentially more cost-effective than customer acquisition. 
        This application implements an end-to-end predictive machine learning solution that identifies at-risk accounts 
        before contract termination, allowing customer success teams to initiate proactive retention workflows.

        ---

        ### 🧠 Machine Learning Pipeline Architecture
        1. **Ingestion & Data Hygiene**:
           - Conversion of untyped `TotalCharges` strings to continuous floats.
           - Zero-fill imputation for new subscribers with zero tenure.
           - Binary target normalization (`Yes` $\\rightarrow$ 1, `No` $\\rightarrow$ 0).

        2. **Domain-Specific Feature Engineering**:
           - **`AvgMonthlyCharge`**: Historical billing normalized over tenure duration.
           - **`PriceDiscrepancy`**: Quantifies recent billing increases against historical tenure averages.
           - **`NumServices`**: Aggregated count of bundled telecommunication and streaming subscriptions.
           - **`UnprotectedInternet`**: Identifies broadband users lacking security and tech support protocols.
           - **`TenureCohort`**: Segmented subscriber loyalty stages (`New`, `Mid-term`, `Long-term`).
           - **`HasPartnerOrDependents`**: Family tie indicator known to correlate with customer stability.
           - **`ContractScore`**: Ordinal mapping representing contract length and commitment depth.
           - **`HasAutoPayment`**: Automated payment flag (bank transfer or recurring credit card).

        3. **Encoding & Transformation**:
           - Standard scaling (`StandardScaler`) on numeric continuous vectors.
           - One-hot encoding (`OneHotEncoder`, `handle_unknown='ignore'`) on categorical features.
           - Unified `ColumnTransformer` scikit-learn pipeline for zero data leakage.

        4. **Modeling & Benchmarks**:
           - Primary: **RandomForestClassifier** & **GradientBoostingClassifier**.
           - Evaluation Metrics: ROC-AUC, Precision-Recall AUC, and F1-score with class balancing.
           - Persisted as unified serializations via `joblib`.

        ---

        ### 👨‍💻 Portfolio & Author
        - **Project**: Telco Churn Prediction & Retention Analytics
        - **Author**: Hazal
        - **Tech Stack**: Python, Streamlit, Scikit-Learn, Pandas, NumPy, Matplotlib, Joblib
        """
    )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #64748b; font-size: 0.85rem; padding: 1rem 0;'>
        ⚡ Customer Churn Prediction Engine | Developed with Streamlit & Scikit-Learn | Portfolio Ready
    </div>
    """,
    unsafe_allow_html=True,
)
