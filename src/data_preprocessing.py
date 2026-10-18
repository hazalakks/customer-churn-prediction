"""
Data Preprocessing and Feature Engineering Module for Telco Customer Churn Prediction.

This module provides data loading, data cleaning, feature engineering,
dataset summarization, and data preparation pipelines (scaling, encoding,
stratified train/test splitting) for production-grade churn modeling.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_raw_data(
    filepath: Union[str, Path] = "data/raw/telco_churn.csv"
) -> pd.DataFrame:
    """
    Load raw Telco Customer Churn data from CSV.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to the raw CSV dataset file.

    Returns
    -------
    pd.DataFrame
        Loaded raw dataset as a pandas DataFrame.

    Raises
    ------
    FileNotFoundError
        If the specified filepath does not exist.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Raw data file not found at: {path.resolve()}")

    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the Telco Customer Churn dataset.

    Performs the following cleaning operations:
    - Drops customer identifier column ('customerID') if present.
    - Handles whitespace and missing values in 'TotalCharges', converts to numeric,
      and imputes tenure=0 missing charges with 0.0.
    - Encodes target variable 'Churn' from Yes/No strings to binary 1/0 integers.

    Parameters
    ----------
    df : pd.DataFrame
        Raw or partially cleaned DataFrame.

    Returns
    -------
    pd.DataFrame
        Cleaned copy of the DataFrame.
    """
    df_clean = df.copy()

    # Drop customerID if it exists
    if "customerID" in df_clean.columns:
        df_clean = df_clean.drop(columns=["customerID"])

    # Fix TotalCharges whitespace / empty values and convert to float
    if "TotalCharges" in df_clean.columns:
        df_clean["TotalCharges"] = pd.to_numeric(
            df_clean["TotalCharges"].astype(str).str.strip(),
            errors="coerce",
        )
        # TotalCharges is null typically when tenure is 0, so fill with 0.0
        df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(0.0)

    # Encode Churn target variable to 0/1 binary
    if "Churn" in df_clean.columns:
        if df_clean["Churn"].dtype == object:
            df_clean["Churn"] = (
                df_clean["Churn"]
                .str.strip()
                .map({"Yes": 1, "No": 0})
                .fillna(df_clean["Churn"])
                .astype(int)
            )

    return df_clean


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate domain-specific derived features for customer churn prediction.

    Engineered features include:
    - AvgMonthlyCharge: TotalCharges / max(tenure, 1)
    - PriceDiscrepancy: MonthlyCharges - AvgMonthlyCharge (highlights price changes)
    - NumServices: Total count of subscribed telecom and streaming services
    - UnprotectedInternet: Internet user lacking both TechSupport and OnlineSecurity
    - TenureCohort: Categorical cohort ('New', 'Mid-term', 'Long-term')
    - HasPartnerOrDependents: Binary flag indicating family ties (Partner or Dependents)
    - ContractScore: Numeric mapping of contract commitment (Month-to-month=0, 1yr=1, 2yr=2)
    - HasAutoPayment: Binary flag for automatic payment methods (bank transfer or credit card)

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame containing required raw customer attributes.

    Returns
    -------
    pd.DataFrame
        DataFrame augmented with engineered feature columns.
    """
    df_eng = df.copy()

    # 1. AvgMonthlyCharge
    tenure_safe = df_eng["tenure"].clip(lower=1)
    df_eng["AvgMonthlyCharge"] = df_eng["TotalCharges"] / tenure_safe

    # 2. PriceDiscrepancy
    df_eng["PriceDiscrepancy"] = df_eng["MonthlyCharges"] - df_eng["AvgMonthlyCharge"]

    # 3. NumServices
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

    # 4. UnprotectedInternet: Has Internet (DSL/Fiber optic) but no TechSupport and no OnlineSecurity
    has_internet = df_eng["InternetService"].isin(["DSL", "Fiber optic"])
    no_tech_support = df_eng["TechSupport"] != "Yes"
    no_online_security = df_eng["OnlineSecurity"] != "Yes"
    df_eng["UnprotectedInternet"] = (
        has_internet & no_tech_support & no_online_security
    ).astype(int)

    # 5. TenureCohort: 'New' (<12), 'Mid-term' (12-36), 'Long-term' (36+)
    bins = [-1, 12, 36, float("inf")]
    labels = ["New", "Mid-term", "Long-term"]
    df_eng["TenureCohort"] = pd.cut(
        df_eng["tenure"], bins=bins, labels=labels, right=False
    ).astype(str)

    # 6. HasPartnerOrDependents
    has_partner = df_eng["Partner"] == "Yes"
    has_dependents = df_eng["Dependents"] == "Yes"
    df_eng["HasPartnerOrDependents"] = (has_partner | has_dependents).astype(int)

    # 7. ContractScore
    contract_mapping = {
        "Month-to-month": 0,
        "One year": 1,
        "Two year": 2,
    }
    df_eng["ContractScore"] = (
        df_eng["Contract"].map(contract_mapping).fillna(0).astype(int)
    )

    # 8. HasAutoPayment: Payment method contains 'automatic'
    df_eng["HasAutoPayment"] = (
        df_eng["PaymentMethod"]
        .astype(str)
        .str.lower()
        .str.contains("automatic", na=False)
        .astype(int)
    )

    return df_eng


def get_data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate an overview and metadata dictionary of columns in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to analyze.

    Returns
    -------
    pd.DataFrame
        DataFrame summarizing column names, data types, non-null counts,
        unique value counts, sample unique values, and functional descriptions.
    """
    col_descriptions = {
        "gender": "Customer gender (Male, Female)",
        "SeniorCitizen": "Whether customer is a senior citizen (1, 0)",
        "Partner": "Whether customer has a partner (Yes, No)",
        "Dependents": "Whether customer has dependents (Yes, No)",
        "tenure": "Number of months customer has stayed with the company",
        "PhoneService": "Whether customer has a phone service (Yes, No)",
        "MultipleLines": "Whether customer has multiple lines (Yes, No, No phone service)",
        "InternetService": "Customer internet service provider (DSL, Fiber optic, No)",
        "OnlineSecurity": "Whether customer has online security (Yes, No, No internet service)",
        "OnlineBackup": "Whether customer has online backup (Yes, No, No internet service)",
        "DeviceProtection": "Whether customer has device protection (Yes, No, No internet service)",
        "TechSupport": "Whether customer has tech support (Yes, No, No internet service)",
        "StreamingTV": "Whether customer has streaming TV (Yes, No, No internet service)",
        "StreamingMovies": "Whether customer has streaming movies (Yes, No, No internet service)",
        "Contract": "Contract term of customer (Month-to-month, One year, Two year)",
        "PaperlessBilling": "Whether customer uses paperless billing (Yes, No)",
        "PaymentMethod": "Customer payment method (Electronic check, Mailed check, Automatic bank/card)",
        "MonthlyCharges": "The amount charged to the customer monthly",
        "TotalCharges": "The total amount charged to the customer",
        "Churn": "Target: Whether customer churned (1: Yes, 0: No)",
        "AvgMonthlyCharge": "Engineered: TotalCharges divided by active tenure (min 1 month)",
        "PriceDiscrepancy": "Engineered: Difference between current monthly and historical average charge",
        "NumServices": "Engineered: Count of subscribed phone, internet, and add-on services",
        "UnprotectedInternet": "Engineered: Flag for internet users without tech support and online security",
        "TenureCohort": "Engineered: Tenure stage group (New, Mid-term, Long-term)",
        "HasPartnerOrDependents": "Engineered: Flag indicating customer has partner and/or dependents",
        "ContractScore": "Engineered: Ordinal loyalty score for contract duration (0, 1, 2)",
        "HasAutoPayment": "Engineered: Flag indicating recurring automated payment setup",
    }

    records = []
    for col in df.columns:
        unique_vals = df[col].dropna().unique()
        sample_str = (
            ", ".join(map(str, unique_vals[:4]))
            + ("..." if len(unique_vals) > 4 else "")
        )
        records.append(
            {
                "Column": col,
                "Dtype": str(df[col].dtype),
                "Non-Null Count": int(df[col].notnull().sum()),
                "Unique Values": int(df[col].nunique()),
                "Sample Values": sample_str,
                "Description": col_descriptions.get(col, "Feature attribute"),
            }
        )

    return pd.DataFrame(records)


def prepare_model_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Split and preprocess dataset into scaled and encoded train/test feature sets.

    Applies:
    - Stratified train/test splitting based on Churn class distribution.
    - StandardScaler to numerical continuous features.
    - OneHotEncoder to categorical features (with handle_unknown='ignore').
    - scikit-learn ColumnTransformer for unified pipeline integration.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame containing target column 'Churn' and features.
    test_size : float, default=0.2
        Fraction of dataset to allocate to the test partition.
    random_state : int, default=42
        Seed for pseudo-random number generator for reproducibility.

    Returns
    -------
    dict
        Dictionary containing:
        - 'X_train': np.ndarray of transformed training features
        - 'X_test': np.ndarray of transformed testing features
        - 'y_train': pd.Series or np.ndarray of training targets
        - 'y_test': pd.Series or np.ndarray of testing targets
        - 'preprocessor': Fitted ColumnTransformer instance
        - 'feature_names': List of feature names after transformation
    """
    if "Churn" not in df.columns:
        raise KeyError("Target column 'Churn' not found in DataFrame.")

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Stratified train/test split to preserve class imbalance ratios
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Distinguish numerical vs categorical feature columns
    numeric_features = (
        X.select_dtypes(include=["int64", "float64", "int32", "float32"])
        .columns.tolist()
    )
    categorical_features = (
        X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    )

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Fit transformer on training set and transform both partitions
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    # Extract transformed feature names
    feature_names: List[str] = []
    if hasattr(preprocessor, "get_feature_names_out"):
        raw_names = preprocessor.get_feature_names_out()
        # Clean prefix names ('num__', 'cat__') for clean readability
        feature_names = [
            name.replace("num__", "").replace("cat__", "") for name in raw_names
        ]
    else:
        cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        encoded_cat_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
        feature_names = numeric_features + encoded_cat_names

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train.to_numpy(),
        "y_test": y_test.to_numpy(),
        "preprocessor": preprocessor,
        "feature_names": feature_names,
    }


def load_and_process_data(
    filepath: Union[str, Path] = "data/raw/telco_churn.csv",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Execute full end-to-end data processing pipeline.

    Steps:
    1. Load raw data from CSV
    2. Clean dataset (missing values, types, target mapping)
    3. Generate 8+ engineered features
    4. Stratify and transform into modeling matrices

    Parameters
    ----------
    filepath : Union[str, Path], default='data/raw/telco_churn.csv'
        Path to raw dataset CSV file.
    test_size : float, default=0.2
        Fraction of data for test split.
    random_state : int, default=42
        Seed for reproducibility.

    Returns
    -------
    dict
        Dictionary containing X_train, X_test, y_train, y_test, preprocessor,
        feature_names, and 'df_clean' (the engineered DataFrame prior to matrix transformation).
    """
    df_raw = load_raw_data(filepath)
    df_cleaned = clean_data(df_raw)
    df_engineered = engineer_features(df_cleaned)

    processed_dict = prepare_model_data(
        df_engineered,
        test_size=test_size,
        random_state=random_state,
    )
    processed_dict["df_clean"] = df_engineered
    return processed_dict


if __name__ == "__main__":
    print("=" * 60)
    print("Executing Telco Churn Data Preprocessing Pipeline Demo")
    print("=" * 60)

    # Locate dataset path relative to repository root or module directory
    base_dir = Path(__file__).resolve().parent.parent
    dataset_path = base_dir / "data" / "raw" / "telco_churn.csv"

    print(f"Dataset path: {dataset_path}")

    try:
        data_bundle = load_and_process_data(filepath=dataset_path)

        df_final = data_bundle["df_clean"]
        print("\n[+] Data Pipeline executed successfully!")
        print(f"Total Rows: {len(df_final)}, Total Columns: {df_final.shape[1]}")
        print(f"X_train shape: {data_bundle['X_train'].shape}")
        print(f"X_test shape:  {data_bundle['X_test'].shape}")
        print(f"y_train shape: {data_bundle['y_train'].shape}")
        print(f"y_test shape:  {data_bundle['y_test'].shape}")
        print(f"Total Transformed Features: {len(data_bundle['feature_names'])}")

        print("\n--- Feature Names Sample (first 10) ---")
        for name in data_bundle["feature_names"][:10]:
            print(f"  - {name}")

        print("\n--- Data Dictionary Preview (first 8 features) ---")
        data_dict = get_data_dictionary(df_final)
        print(data_dict.head(8).to_string(index=False))

    except FileNotFoundError as err:
        print(f"\n[!] Notice: {err}")
        print("Creating mock dataset to demonstrate pipeline execution...")

        # Create a representative sample DataFrame matching Telco Churn schema
        mock_data = {
            "customerID": [f"CUST-{i:03d}" for i in range(10)],
            "gender": ["Female", "Male"] * 5,
            "SeniorCitizen": [0, 1] * 5,
            "Partner": ["Yes", "No"] * 5,
            "Dependents": ["No", "Yes"] * 5,
            "tenure": [1, 0, 12, 24, 48, 60, 5, 34, 70, 0],
            "PhoneService": ["Yes", "No", "Yes", "Yes", "Yes", "No", "Yes", "Yes", "Yes", "No"],
            "MultipleLines": ["No", "No phone service", "Yes", "No", "Yes", "No phone service", "No", "Yes", "Yes", "No phone service"],
            "InternetService": ["DSL", "Fiber optic", "DSL", "Fiber optic", "No", "DSL", "Fiber optic", "DSL", "No", "DSL"],
            "OnlineSecurity": ["No", "No internet service", "Yes", "No", "No internet service", "Yes", "No", "Yes", "No internet service", "No"],
            "OnlineBackup": ["Yes", "No internet service", "No", "Yes", "No internet service", "No", "Yes", "No", "No internet service", "Yes"],
            "DeviceProtection": ["No", "No internet service", "Yes", "Yes", "No internet service", "No", "No", "Yes", "No internet service", "No"],
            "TechSupport": ["No", "No internet service", "Yes", "No", "No internet service", "Yes", "No", "No", "No internet service", "No"],
            "StreamingTV": ["No", "No internet service", "Yes", "Yes", "No internet service", "No", "Yes", "Yes", "No internet service", "No"],
            "StreamingMovies": ["No", "No internet service", "Yes", "Yes", "No internet service", "No", "Yes", "Yes", "No internet service", "No"],
            "Contract": ["Month-to-month", "One year", "Month-to-month", "Two year", "Month-to-month", "One year", "Month-to-month", "Two year", "One year", "Month-to-month"],
            "PaperlessBilling": ["Yes", "No"] * 5,
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
                "Electronic check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
                "Mailed check",
                "Electronic check",
                "Credit card (automatic)",
            ],
            "MonthlyCharges": [29.85, 56.95, 53.85, 42.30, 70.70, 99.65, 89.10, 20.25, 45.30, 25.00],
            "TotalCharges": ["29.85", " ", "646.2", "1015.2", "3393.6", "5979.0", "445.5", "688.5", "3171.0", " "],
            "Churn": ["No", "No", "Yes", "No", "Yes", "No", "Yes", "No", "No", "Yes"],
        }
        df_mock = pd.DataFrame(mock_data)
        df_clean = clean_data(df_mock)
        df_eng = engineer_features(df_clean)
        bundle = prepare_model_data(df_eng, test_size=0.3, random_state=42)

        print("\n[+] Mock Pipeline successfully tested!")
        print(f"X_train shape: {bundle['X_train'].shape}")
        print(f"X_test shape:  {bundle['X_test'].shape}")
        print(f"Total features: {len(bundle['feature_names'])}")
        print(f"Feature Names: {bundle['feature_names']}")
