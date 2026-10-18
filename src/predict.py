"""
Customer Churn Prediction - Inference and Prediction Module
============================================================

This module provides utilities to:
- Save and load trained model artifacts (model, preprocessor, feature names)
- Generate real-time churn predictions for individual customer profiles
- Batch predict churn risk for tabular customer datasets
- Provide actionable business retention recommendations based on risk tiers
- Offer sample customer personas covering diverse subscriber segments
- Run an interactive Turkish console interface (`tahmin_et`) meeting curriculum requirements
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd

# Suppress warnings
warnings.filterwarnings("ignore")

# Default artifact storage path
DEFAULT_MODEL_DIR = Path("models")


def save_artifacts(
    model: Any,
    preprocessor: Any,
    feature_names: List[str],
    model_dir: Union[str, Path] = "models",
) -> Path:
    """Save model, preprocessor, and feature names to joblib files.

    Parameters
    ----------
    model : estimator
        Trained machine learning model.
    preprocessor : ColumnTransformer or pipeline
        Fitted preprocessing pipeline.
    feature_names : list of str
        List of encoded feature names matching model inputs.
    model_dir : str or Path, default='models'
        Directory where artifacts will be saved.

    Returns
    -------
    Path
        Path to the saved directory.
    """
    dest_path = Path(model_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, dest_path / "best_model.joblib")
    joblib.dump(preprocessor, dest_path / "preprocessor.joblib")
    joblib.dump(feature_names, dest_path / "feature_names.joblib")

    print(f"[Artifacts] Model and preprocessor saved successfully to '{dest_path}'.")
    return dest_path


def load_artifacts(model_dir: Union[str, Path] = "models") -> dict:
    """Load saved model, preprocessor, and feature names from joblib files.

    Parameters
    ----------
    model_dir : str or Path, default='models'
        Directory containing the saved joblib artifacts.

    Returns
    -------
    dict
        Dictionary containing 'model', 'preprocessor', and 'feature_names'.

    Raises
    ------
    FileNotFoundError
        If required artifact files are missing from model_dir.
    """
    src_path = Path(model_dir)
    model_file = src_path / "best_model.joblib"
    prep_file = src_path / "preprocessor.joblib"
    names_file = src_path / "feature_names.joblib"

    if not model_file.exists():
        raise FileNotFoundError(f"Model artifact not found at '{model_file}'.")
    if not prep_file.exists():
        raise FileNotFoundError(f"Preprocessor artifact not found at '{prep_file}'.")

    model = joblib.load(model_file)
    preprocessor = joblib.load(prep_file)
    feature_names = joblib.load(names_file) if names_file.exists() else []

    return {
        "model": model,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
    }


def _generate_recommendation(customer_data: dict, proba: float, risk_level: str) -> str:
    """Generate intelligent retention action recommendations based on customer attributes and risk level."""
    if risk_level == "Low":
        return (
            "[Low Churn Probability] Customer is satisfied. Maintain service standard and "
            "explore cross-sell opportunities (e.g. upgraded streaming packages or connected home devices)."
        )

    contract = str(customer_data.get("Contract", "")).lower()
    tech_support = str(customer_data.get("TechSupport", "")).lower()
    online_security = str(customer_data.get("OnlineSecurity", "")).lower()
    payment_method = str(customer_data.get("PaymentMethod", "")).lower()
    tenure = float(customer_data.get("tenure", 12))

    actions = []

    if "month-to-month" in contract:
        actions.append("Offer a 15% discount incentive to convert to an annual contract")
    if "no" in tech_support or "no" in online_security:
        actions.append("Bundle complimentary Premium Tech Support & Cyber Security for 3 months")
    if "electronic check" in payment_method:
        actions.append("Incentivize automatic credit card/bank billing with a one-time $20 statement credit")
    if tenure < 6:
        actions.append("Schedule an onboarding success call to ensure optimal service adoption")

    if not actions:
        actions.append("Initiate dedicated customer care check-in and survey for satisfaction feedback")

    if risk_level == "High":
        return f"[HIGH RISK PRIORITY] {'; '.join(actions)}. Flag for proactive retention team outreach within 24 hours."
    else:
        return f"[MEDIUM RISK] {'; '.join(actions)}. Send targeted automated retention email campaign."


def predict_single(
    model: Any,
    preprocessor: Any,
    customer_data: dict,
) -> dict:
    """Predict churn for a single customer.
    customer_data is a dict with raw feature values (before preprocessing).
    Return dict with: prediction (0/1), probability, risk_level ('Low'/'Medium'/'High'), recommendation.

    Parameters
    ----------
    model : estimator
        Trained machine learning model.
    preprocessor : ColumnTransformer
        Fitted preprocessor for feature transformation.
    customer_data : dict
        Raw feature dictionary representing customer profile.

    Returns
    -------
    dict
        Dictionary containing prediction (int), probability (float),
        risk_level (str), and recommendation (str).
    """
    df = pd.DataFrame([customer_data])

    # Clean TotalCharges if given as string
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(
            df["MonthlyCharges"].astype(float) * df["tenure"].astype(float)
        )

    # Exclude non-feature columns if present
    drop_cols = [c for c in ["customerID", "Churn", "name"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Apply feature engineering (preprocessor expects engineered features)
    try:
        from src.data_preprocessing import engineer_features
        df = engineer_features(df)
    except ImportError:
        # Inline feature engineering fallback
        tenure_safe = df["tenure"].clip(lower=1)
        df["AvgMonthlyCharge"] = df["TotalCharges"] / tenure_safe
        df["PriceDiscrepancy"] = df["MonthlyCharges"] - df["AvgMonthlyCharge"]
        service_cols = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
        df["NumServices"] = sum((df[c] == "Yes").astype(int) for c in service_cols if c in df.columns)
        has_internet = df["InternetService"].isin(["DSL", "Fiber optic"])
        df["UnprotectedInternet"] = (has_internet & (df["TechSupport"] != "Yes") & (df["OnlineSecurity"] != "Yes")).astype(int)
        bins = [-1, 12, 36, float("inf")]
        labels = ["New", "Mid-term", "Long-term"]
        df["TenureCohort"] = pd.cut(df["tenure"], bins=bins, labels=labels, right=False).astype(str)
        df["HasPartnerOrDependents"] = ((df["Partner"] == "Yes") | (df["Dependents"] == "Yes")).astype(int)
        contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
        df["ContractScore"] = df["Contract"].map(contract_map).fillna(0).astype(int)
        df["HasAutoPayment"] = df["PaymentMethod"].astype(str).str.lower().str.contains("automatic", na=False).astype(int)

    # Transform features
    if preprocessor is not None:
        X_trans = preprocessor.transform(df)
    else:
        X_trans = df.values

    # Compute probability
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(X_trans)[0, 1])
    elif hasattr(model, "decision_function"):
        score = float(model.decision_function(X_trans)[0])
        proba = float(1 / (1 + np.exp(-score)))
    else:
        pred_raw = int(model.predict(X_trans)[0])
        proba = 1.0 if pred_raw == 1 else 0.0

    prediction = int(proba >= 0.5)

    if proba < 0.35:
        risk_level = "Low"
    elif proba < 0.65:
        risk_level = "Medium"
    else:
        risk_level = "High"

    recommendation = _generate_recommendation(customer_data, proba, risk_level)

    return {
        "prediction": prediction,
        "probability": round(proba, 4),
        "risk_level": risk_level,
        "recommendation": recommendation,
    }


def predict_batch(
    model: Any,
    preprocessor: Any,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Predict churn for a batch of customers. Add prediction and probability columns.

    Parameters
    ----------
    model : estimator
        Trained classification model.
    preprocessor : ColumnTransformer
        Fitted data preprocessor.
    df : pd.DataFrame
        DataFrame with raw customer attributes.

    Returns
    -------
    pd.DataFrame
        Copy of input DataFrame augmented with Churn_Prediction, Churn_Probability,
        and Risk_Level columns.
    """
    res_df = df.copy()

    # Preprocess feature subset
    eval_df = res_df.copy()
    if "TotalCharges" in eval_df.columns:
        eval_df["TotalCharges"] = pd.to_numeric(eval_df["TotalCharges"], errors="coerce")
        eval_df["TotalCharges"] = eval_df["TotalCharges"].fillna(
            eval_df["MonthlyCharges"].astype(float) * eval_df["tenure"].astype(float)
        )

    drop_cols = [c for c in ["customerID", "Churn", "name"] if c in eval_df.columns]
    if drop_cols:
        eval_df = eval_df.drop(columns=drop_cols)

    # Apply feature engineering (preprocessor expects engineered features)
    try:
        from src.data_preprocessing import engineer_features
        eval_df = engineer_features(eval_df)
    except ImportError:
        pass  # Assume features already present if import fails

    if preprocessor is not None:
        X_trans = preprocessor.transform(eval_df)
    else:
        X_trans = eval_df.values

    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X_trans)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_trans)
        probas = 1 / (1 + np.exp(-scores))
    else:
        probas = model.predict(X_trans).astype(float)

    preds = (probas >= 0.5).astype(int)

    risk_levels = []
    for p in probas:
        if p < 0.35:
            risk_levels.append("Low")
        elif p < 0.65:
            risk_levels.append("Medium")
        else:
            risk_levels.append("High")

    res_df["Churn_Prediction"] = preds
    res_df["Churn_Probability"] = np.round(probas, 4)
    res_df["Risk_Level"] = risk_levels

    return res_df


def get_sample_customers() -> List[dict]:
    """Return 5+ sample customer profiles for testing predictions.
    Include diverse profiles: new customer, loyal customer, at-risk customer, etc.

    Returns
    -------
    list of dict
        Diverse customer profile representations.
    """
    return [
        {
            "name": "1. High-Risk Attrition Subscriber (New, Month-to-Month, Fiber Optic)",
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 2,
            "PhoneService": "Yes",
            "MultipleLines": "No",
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
            "MonthlyCharges": 89.85,
            "TotalCharges": 179.70,
        },
        {
            "name": "2. Loyal Long-Term Enterprise/Family Subscriber",
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "Yes",
            "tenure": 68,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Two year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Bank transfer (automatic)",
            "MonthlyCharges": 84.50,
            "TotalCharges": 5746.00,
        },
        {
            "name": "3. Budget Entry-Level Subscriber",
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 12,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "No",
            "OnlineSecurity": "No internet service",
            "OnlineBackup": "No internet service",
            "DeviceProtection": "No internet service",
            "TechSupport": "No internet service",
            "StreamingTV": "No internet service",
            "StreamingMovies": "No internet service",
            "Contract": "One year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Mailed check",
            "MonthlyCharges": 20.15,
            "TotalCharges": 241.80,
        },
        {
            "name": "4. Senior Citizen with Electronic Check Billing",
            "gender": "Male",
            "SeniorCitizen": 1,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 5,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "Yes",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 85.50,
            "TotalCharges": 427.50,
        },
        {
            "name": "5. Moderate Tenure Streaming Power User",
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 28,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "One year",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Credit card (automatic)",
            "MonthlyCharges": 104.20,
            "TotalCharges": 2917.60,
        },
        {
            "name": "6. Low-Usage Phone-Only Veteran",
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "Yes",
            "tenure": 54,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "No",
            "OnlineSecurity": "No internet service",
            "OnlineBackup": "No internet service",
            "DeviceProtection": "No internet service",
            "TechSupport": "No internet service",
            "StreamingTV": "No internet service",
            "StreamingMovies": "No internet service",
            "Contract": "Two year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Mailed check",
            "MonthlyCharges": 19.50,
            "TotalCharges": 1053.00,
        },
    ]


def format_prediction_result(result: dict) -> str:
    """Format prediction result as a readable string with emoji indicators.

    Parameters
    ----------
    result : dict
        Output from predict_single.

    Returns
    -------
    str
        Human-readable formatted analysis card.
    """
    prediction = result.get("prediction", 0)
    proba = result.get("probability", 0.0)
    risk_level = result.get("risk_level", "Low")
    recommendation = result.get("recommendation", "")

    status_tag = "[!] CHURN PREDICTED (Likely to Leave)" if prediction == 1 else "[v] RETAINED (Likely to Stay)"
    risk_tag = f"[{risk_level.upper()} RISK]"

    border = "=" * 65
    divider = "-" * 65

    lines = [
        border,
        f"  Result      : {status_tag}",
        f"  Probability : {proba * 100:.1f}% churn likelihood",
        f"  Risk Tier   : {risk_tag}",
        divider,
        "  Strategic Recommendation:",
        f"  {recommendation}",
        border,
    ]
    return "\n".join(lines)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _get_fallback_components():
    """Train a fallback model with the project's own preprocessing pipeline.

    Used only when no saved artifacts exist yet (e.g. a fresh clone before
    running ``python -m src.model``). Uses the same cleaning + feature
    engineering + ColumnTransformer as the main pipeline so that
    ``predict_single`` stays consistent.
    """
    import sys
    from sklearn.ensemble import RandomForestClassifier

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.data_preprocessing import load_and_process_data

    data = load_and_process_data(PROJECT_ROOT / "data" / "raw" / "telco_churn.csv")
    model = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    model.fit(data["X_train"], data["y_train"])
    return model, data["preprocessor"], data["feature_names"]


def tahmin_et() -> None:
    """Interactive console-based prediction function (curriculum requirement).
    Ask user for inputs via input(), preprocess, predict, display result with f-string.
    Use Turkish prompts for the curriculum requirement.
    """
    print("\n" + "=" * 65)
    print("   MÜŞTERİ TERK (CHURN) TAHMİN SİSTEMİ - KONSOL ARAYÜZÜ")
    print("=" * 65)
    print("Lütfen müşteri profil bilgilerini giriniz (Varsayılan değer için Enter'a basınız):")

    # Load artifacts or fit fallback components
    model_dir = PROJECT_ROOT / "models"
    if (model_dir / "best_model.joblib").exists() and (model_dir / "preprocessor.joblib").exists():
        artifacts = load_artifacts(model_dir)
        model = artifacts["model"]
        preprocessor = artifacts["preprocessor"]
    else:
        print("[Bilgi] Kayıtlı model dosyası aranıyor... (İlk çalıştırma için eğitiliyor)")
        model, preprocessor, _ = _get_fallback_components()

    # Turkish Prompts with smart defaults
    def ask(prompt: str, default: str) -> str:
        val = input(f"{prompt} [Varsayılan: {default}]: ").strip()
        return val if val else default

    gender = ask("1. Cinsiyet (Female / Male)", "Female")
    senior = ask("2. Emekli / Yaşlı Vatandaş mı? (0: Hayır, 1: Evet)", "0")
    partner = ask("3. Eşi/Partneri var mı? (Yes / No)", "No")
    dependents = ask("4. Bakmakla yükümlü olduğu kişi var mı? (Yes / No)", "No")
    tenure = ask("5. Abonelik Süresi (Ay cinsinden)", "3")
    phone_service = ask("6. Telefon Hizmeti var mı? (Yes / No)", "Yes")
    multiple_lines = ask("7. Çoklu Hat var mı? (Yes / No / No phone service)", "No")
    internet_service = ask("8. İnternet Servis Türü (DSL / Fiber optic / No)", "Fiber optic")
    online_security = ask("9. Çevrimiçi Güvenlik Paketi var mı? (Yes / No)", "No")
    online_backup = ask("10. Çevrimiçi Yedekleme Paketi var mı? (Yes / No)", "No")
    device_prot = ask("11. Cihaz Koruma Sigortası var mı? (Yes / No)", "No")
    tech_support = ask("12. Teknik Destek Paketi var mı? (Yes / No)", "No")
    streaming_tv = ask("13. İnternet TV Yayını var mı? (Yes / No)", "Yes")
    streaming_movies = ask("14. İnternet Film Yayını var mı? (Yes / No)", "Yes")
    contract = ask("15. Sözleşme Tipi (Month-to-month / One year / Two year)", "Month-to-month")
    paperless = ask("16. Kağıtsız Fatura Tercihi (Yes / No)", "Yes")
    payment_method = ask(
        "17. Ödeme Yöntemi (Electronic check / Mailed check / Bank transfer (automatic) / Credit card (automatic))",
        "Electronic check",
    )
    monthly_charges = ask("18. Aylık Fatura Tutarı ($)", "85.50")

    try:
        tenure_val = float(tenure)
    except (ValueError, TypeError):
        tenure_val = 3.0

    try:
        monthly_val = float(monthly_charges)
    except (ValueError, TypeError):
        monthly_val = 85.50

    total_val = tenure_val * monthly_val

    try:
        senior_val = int(senior)
    except (ValueError, TypeError):
        senior_val = 1 if str(senior).strip().lower() in ["1", "evet", "yes", "true"] else 0

    customer_input = {
        "gender": gender,
        "SeniorCitizen": senior_val,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure_val,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_prot,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_val,
        "TotalCharges": total_val,
    }

    # Run Prediction
    result = predict_single(model, preprocessor, customer_input)

    pred = result["prediction"]
    proba = result["probability"]
    risk = result["risk_level"]

    durum_str = "TERK EDECEK (CHURN)" if pred == 1 else "DEVAM EDECEK (RETAINED)"
    renk_icon = "[!]" if risk == "High" else ("[?]" if risk == "Medium" else "[v]")

    # Display results with Turkish f-strings
    print("\n" + "=" * 65)
    print("                [*] TAHMIN SONUC RAPORU")
    print("=" * 65)
    print(f"Musteri Durumu       : {durum_str}")
    print(f"Terk Etme Olasiligi  : %{proba * 100:.2f}")
    print(f"Risk Kategorisi      : {renk_icon} {risk} Risk Seviyesi")
    print("-" * 65)
    print("Onerilen Aksiyon:")
    print(f"-> {result['recommendation']}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    import sys

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    # `python -m src.predict --interaktif` -> interactive Turkish console app (Stage 6)
    if "--interaktif" in sys.argv or "-i" in sys.argv:
        tahmin_et()
        sys.exit(0)

    print("\n=== DEMO: Customer Churn Prediction Engine ===\n")

    # Load artifacts or use fallback training
    try:
        loaded = load_artifacts(PROJECT_ROOT / "models")
        demo_model = loaded["model"]
        demo_prep = loaded["preprocessor"]
        print("Successfully loaded pre-trained model from 'models/'.\n")
    except Exception:
        print("Artifacts not yet on disk. Training fallback model with the project pipeline...")
        demo_model, demo_prep, demo_features = _get_fallback_components()
        save_artifacts(demo_model, demo_prep, demo_features, PROJECT_ROOT / "models")
        print("Fallback model trained and saved to 'models/' for instant inference.\n")

    samples = get_sample_customers()
    print(f"Testing {len(samples)} diverse customer profiles:\n")

    for idx, sample in enumerate(samples, 1):
        name = sample.pop("name", f"Profile {idx}")
        res = predict_single(demo_model, demo_prep, sample)
        print(f"--- Persona: {name} ---")
        print(format_prediction_result(res))
        print()
