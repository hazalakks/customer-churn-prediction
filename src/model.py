"""
Customer Churn Prediction - Model Training, Evaluation, and Explainability Module
==================================================================================

This module handles:
- Model instantiation for multiple supervised learning algorithms
- Model training and tracking
- Comprehensive evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- Stratified 5-fold cross-validation
- Hyperparameter tuning via GridSearchCV
- Confusion matrix, ROC curve, and Precision-Recall curve visualizations
- Feature importance visualization
- SHAP explainability (Beeswarm and Waterfall plots)
- Cost-benefit classification threshold optimization
- Complete end-to-end training and evaluation pipeline
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Suppress warnings for clean report outputs
warnings.filterwarnings("ignore")

# Attempt XGBoost import with graceful fallback if not installed
try:
    from xgboost import XGBClassifier
    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False

# Attempt SHAP import with graceful fallback
try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

# Curated Professional Color Palette
COLORS = {
    "coral": "#FF6B6B",
    "teal": "#4ECDC4",
    "sky_blue": "#45B7D1",
    "sage": "#96CEB4",
    "plum": "#DDA0DD",
}
PALETTE_LIST = [
    COLORS["coral"],
    COLORS["teal"],
    COLORS["sky_blue"],
    COLORS["sage"],
    COLORS["plum"],
    "#F4A261",  # amber (tuned models)
    "#2A9D8F",  # deep teal (tuned models)
]


def _set_chart_style() -> None:
    """Set global consistent matplotlib and seaborn styling."""
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["figure.titlesize"] = 14
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 10


def get_models(scale_pos_weight: float = 1.0) -> dict:
    """Return dict of model_name: model_instance for:
    - Logistic Regression (with class_weight='balanced')
    - Decision Tree Classifier (max_depth=5)
    - Random Forest Classifier (n_estimators=100, class_weight='balanced')
    - XGBoost Classifier (use_label_encoder=False, eval_metric='logloss', scale_pos_weight=ratio)
    - Support Vector Machine (kernel='rbf', probability=True, class_weight='balanced')

    Parameters
    ----------
    scale_pos_weight : float, default=1.0
        Ratio of negative to positive class instances (neg / pos) for XGBoost.

    Returns
    -------
    dict
        Dictionary mapping model names to un-fitted classifier objects.
    """
    models: dict = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
            solver="lbfgs",
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            random_state=42,
            class_weight="balanced",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "Support Vector Machine": SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=42,
        ),
    }

    if _HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
        )

    return models


def train_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    models: Optional[dict] = None,
) -> dict:
    """Train all models. Return dict of model_name: fitted_model.

    Parameters
    ----------
    X_train : np.ndarray
        Preprocessed training feature matrix.
    y_train : np.ndarray
        Training target vector (binary 0/1).
    models : dict, optional
        Dictionary of model_name: model_instance. If None, generated via get_models().

    Returns
    -------
    dict
        Dictionary mapping model name to the trained model instance.
        Fitted models also have an attached `_training_time` attribute (in seconds).
    """
    if models is None:
        neg_count = np.sum(y_train == 0)
        pos_count = np.sum(y_train == 1)
        scale_pos_weight = float(neg_count / max(pos_count, 1))
        models = get_models(scale_pos_weight=scale_pos_weight)

    trained_models = {}
    for name, model in models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        elapsed_time = round(time.time() - start_time, 4)
        model._training_time = elapsed_time
        trained_models[name] = model

    return trained_models


def evaluate_models(
    trained_models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """Evaluate all models. Return DataFrame with columns:
    Model, Accuracy, Precision, Recall, F1-Score, ROC-AUC, Training Time.
    Sort by F1-Score descending.

    Parameters
    ----------
    trained_models : dict
        Dictionary of model_name: fitted_model.
    X_test : np.ndarray
        Preprocessed test feature matrix.
    y_test : np.ndarray
        Test target vector.

    Returns
    -------
    pd.DataFrame
        Ranked performance comparison table sorted by F1-Score descending.
    """
    records = []
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_proba = model.decision_function(X_test)
        else:
            y_proba = y_pred

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba)
        training_time = getattr(model, "_training_time", 0.0)

        records.append({
            "Model": name,
            "Accuracy": round(accuracy, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(roc_auc, 4),
            "Training Time": round(training_time, 4),
        })

    df = pd.DataFrame(records)
    df = df.sort_values(by="F1-Score", ascending=False).reset_index(drop=True)
    return df


def cross_validate_models(
    models: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = 5,
) -> pd.DataFrame:
    """5-fold stratified cross-validation for all models.
    Return DataFrame with Model, CV_Mean_Accuracy, CV_Std, CV_Mean_F1, CV_Std_F1.

    Parameters
    ----------
    models : dict
        Dictionary of model_name: model_instance.
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training target vector.
    cv : int, default=5
        Number of stratified cross-validation folds.

    Returns
    -------
    pd.DataFrame
        Cross-validation metrics per model.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    records = []

    scoring = ["accuracy", "f1"]

    for name, model in models.items():
        # Use clone or fresh model instance if fitted
        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=skf,
            scoring=scoring,
            n_jobs=-1,
        )

        acc_mean = scores["test_accuracy"].mean()
        acc_std = scores["test_accuracy"].std()
        f1_mean = scores["test_f1"].mean()
        f1_std = scores["test_f1"].std()

        records.append({
            "Model": name,
            "CV_Mean_Accuracy": round(acc_mean, 4),
            "CV_Std": round(acc_std, 4),
            "CV_Mean_F1": round(f1_mean, 4),
            "CV_Std_F1": round(f1_std, 4),
        })

    df = pd.DataFrame(records)
    df = df.sort_values(by="CV_Mean_F1", ascending=False).reset_index(drop=True)
    return df


def get_default_param_grids() -> dict:
    """Return default hyperparameter grids for tuning.

    The Random Forest grid deliberately includes regularisation parameters
    (``max_depth``, ``min_samples_leaf``) because the untuned forest memorises
    the training set (train accuracy ~99.8% vs. test ~78.6%).

    Returns
    -------
    dict
        Mapping of model names to hyperparameter search grids.
    """
    return {
        "Random Forest": {
            "n_estimators": [100, 200],
            "max_depth": [5, 8, 12, None],
            "min_samples_leaf": [1, 5, 10],
        },
        "XGBoost": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2],
        },
    }


def tune_best_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: dict,
    cv: int = 5,
) -> Tuple[Any, dict, float]:
    """GridSearchCV on the given model. Return best_model, best_params, best_score.
    Include param grids for Random Forest and XGBoost.

    Parameters
    ----------
    model : estimator
        Scikit-learn or XGBoost classifier to tune.
    X_train : np.ndarray
        Training feature set.
    y_train : np.ndarray
        Training target labels.
    param_grid : dict
        Hyperparameter grid dictionary.
    cv : int, default=5
        Number of cross-validation folds.

    Returns
    -------
    best_model : estimator
        The estimator with the highest mean cross-validation score.
    best_params : dict
        Optimal parameters discovered by grid search.
    best_score : float
        Highest F1 cross-validation score achieved.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="f1",
        cv=skf,
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_, round(grid_search.best_score_, 4)


def plot_confusion_matrices(
    trained_models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot confusion matrices for all models in a grid layout.
    Use seaborn heatmap with annotations. Beautiful styling.

    Parameters
    ----------
    trained_models : dict
        Dictionary of model_name: fitted_model.
    X_test : np.ndarray
        Test feature matrix.
    y_test : np.ndarray
        Ground truth test labels.
    save_path : str or Path, optional
        Filepath where the figure should be saved.

    Returns
    -------
    plt.Figure
        The created matplotlib figure.
    """
    _set_chart_style()
    n_models = len(trained_models)
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n_models == 1:
        axes = np.array([axes])
    axes = np.ravel(axes)

    # Custom teal/coral colormap styling
    cmap = sns.light_palette(COLORS["teal"], as_cmap=True)

    for idx, (name, model) in enumerate(trained_models.items()):
        ax = axes[idx]
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap=cmap,
            cbar=False,
            ax=ax,
            annot_kws={"size": 13, "weight": "bold"},
            linewidths=1.5,
            linecolor="white",
        )
        ax.set_title(f"{name}", fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)
        ax.set_xticklabels(["Retained (0)", "Churned (1)"])
        ax.set_yticklabels(["Retained (0)", "Churned (1)"])

    # Hide unused subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Confusion Matrix Comparison Across Models", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_roc_curves(
    trained_models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot ROC curves for all models on the same figure.
    Include AUC values in legend. Add diagonal reference line.

    Parameters
    ----------
    trained_models : dict
        Dictionary of model_name: fitted_model.
    X_test : np.ndarray
        Test feature matrix.
    y_test : np.ndarray
        Ground truth test labels.
    save_path : str or Path, optional
        Filepath to save figure.

    Returns
    -------
    plt.Figure
        The created matplotlib figure.
    """
    _set_chart_style()
    fig, ax = plt.subplots(figsize=(9, 7))

    for idx, (name, model) in enumerate(trained_models.items()):
        color = PALETTE_LIST[idx % len(PALETTE_LIST)]
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_proba = model.decision_function(X_test)
        else:
            y_proba = model.predict(X_test)

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(
            fpr,
            tpr,
            label=f"{name} (AUC = {roc_auc:.3f})",
            color=color,
            linewidth=2.4,
        )

    # Diagonal baseline reference line
    ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", linewidth=1.5, label="Random Guess (AUC = 0.500)")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=15, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_precision_recall_curves(
    trained_models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot Precision-Recall curves for all models.
    Include AP (Average Precision) values in legend.

    Parameters
    ----------
    trained_models : dict
        Dictionary of model_name: fitted_model.
    X_test : np.ndarray
        Test feature matrix.
    y_test : np.ndarray
        Ground truth test labels.
    save_path : str or Path, optional
        Filepath to save figure.

    Returns
    -------
    plt.Figure
        The created matplotlib figure.
    """
    _set_chart_style()
    fig, ax = plt.subplots(figsize=(9, 7))

    baseline_ap = np.sum(y_test) / len(y_test)

    for idx, (name, model) in enumerate(trained_models.items()):
        color = PALETTE_LIST[idx % len(PALETTE_LIST)]
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_proba = model.decision_function(X_test)
        else:
            y_proba = model.predict(X_test)

        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recall, precision)
        ax.plot(
            recall,
            precision,
            label=f"{name} (AP = {pr_auc:.3f})",
            color=color,
            linewidth=2.4,
        )

    ax.axhline(
        y=baseline_ap,
        color="#888888",
        linestyle="--",
        linewidth=1.5,
        label=f"No Skill Baseline (AP = {baseline_ap:.3f})",
    )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall (PR) Curves", fontsize=15, fontweight="bold", pad=12)
    ax.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    top_n: int = 15,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot top N feature importances for tree-based models.
    Horizontal bar chart, sorted.

    Parameters
    ----------
    model : estimator
        Trained model with feature_importances_ or coef_ attribute.
    feature_names : list of str
        List of feature column names.
    top_n : int, default=15
        Number of top features to display.
    save_path : str or Path, optional
        Filepath to save figure.

    Returns
    -------
    plt.Figure
        The created matplotlib figure.
    """
    _set_chart_style()

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        metric_label = "Feature Importance"
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
        metric_label = "Absolute Coefficient Magnitude"
    else:
        raise AttributeError("Model does not expose feature_importances_ or coef_ attributes.")

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
    }).sort_values(by="Importance", ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
    bars = ax.barh(
        importance_df["Feature"],
        importance_df["Importance"],
        color=COLORS["teal"],
        edgecolor=COLORS["coral"],
        linewidth=0.8,
    )

    # Annotate bar values
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.4f}",
            va="center",
            ha="left",
            fontsize=9,
            color="#333333",
        )

    ax.set_xlabel(metric_label, fontsize=12)
    ax.set_ylabel("Features", fontsize=12)
    ax.set_title(f"Top {top_n} Key Drivers of Customer Churn", fontsize=15, fontweight="bold", pad=12)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def shap_analysis(
    model: Any,
    X_test: np.ndarray,
    feature_names: List[str],
    save_path: Optional[Union[str, Path]] = None,
) -> dict:
    """SHAP analysis:
    1. Create TreeExplainer (for tree models) or KernelExplainer
    2. Calculate SHAP values
    3. Plot beeswarm summary plot
    4. Plot waterfall for first 3 test instances
    Save all plots.

    Parameters
    ----------
    model : estimator
        Trained machine learning model.
    X_test : np.ndarray
        Test feature matrix.
    feature_names : list of str
        Names corresponding to columns in X_test.
    save_path : str or Path, optional
        Base directory or file prefix to save the SHAP figures.

    Returns
    -------
    dict
        Dictionary containing explainer, shap_values, and generated figures.
    """
    if not _HAS_SHAP:
        print("[Warning] SHAP package not installed. Skipping SHAP analysis.")
        return {}

    save_dir = Path(save_path) if save_path else Path("reports/figures")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Initialize appropriate explainer
    if hasattr(model, "estimators_") or hasattr(model, "get_booster") or "Tree" in type(model).__name__:
        explainer = shap.TreeExplainer(model)
        shap_values_raw = explainer(X_test)
    else:
        # Background summary sample for kernel explainer
        background = shap.sample(X_test, min(50, len(X_test)))
        explainer = shap.Explainer(model.predict_proba, background)
        shap_values_raw = explainer(X_test)

    # Handle multi-class / binary structure for shap values
    if hasattr(shap_values_raw, "values") and len(shap_values_raw.values.shape) == 3:
        # Take churn positive class (index 1)
        shap_values = shap.Explanation(
            values=shap_values_raw.values[:, :, 1],
            base_values=shap_values_raw.base_values[:, 1],
            data=shap_values_raw.data,
            feature_names=feature_names,
        )
    else:
        shap_values = shap_values_raw
        if hasattr(shap_values, "feature_names") and (shap_values.feature_names is None or len(shap_values.feature_names) == 0):
            shap_values.feature_names = feature_names

    # 1. Beeswarm summary plot
    plt.figure(figsize=(10, 7))
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.title("SHAP Feature Importance & Impact (Beeswarm)", fontsize=14, fontweight="bold", pad=12)
    beeswarm_path = save_dir / "shap_beeswarm_summary.png"
    plt.tight_layout()
    plt.savefig(beeswarm_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Waterfall plots for the first 3 test instances
    waterfall_figs = []
    for i in range(min(3, len(X_test))):
        plt.figure(figsize=(9, 6))
        shap.plots.waterfall(shap_values[i], max_display=10, show=False)
        plt.title(f"SHAP Local Explanation - Customer Instance #{i + 1}", fontsize=13, fontweight="bold", pad=10)
        wf_path = save_dir / f"shap_waterfall_instance_{i + 1}.png"
        plt.tight_layout()
        plt.savefig(wf_path, dpi=300, bbox_inches="tight")
        plt.close()
        waterfall_figs.append(wf_path)

    return {
        "explainer": explainer,
        "shap_values": shap_values,
        "beeswarm_plot": beeswarm_path,
        "waterfall_plots": waterfall_figs,
    }


def optimize_threshold(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cost_fp: float = 50.0,
    cost_fn: float = 200.0,
    save_path: Optional[Union[str, Path]] = None,
) -> dict:
    """Find optimal classification threshold based on cost-benefit analysis.
    Test thresholds from 0.05 to 0.95.
    Return dict with: optimal_threshold, net_benefit, threshold_analysis_df.
    Plot threshold vs net benefit curve.

    Cost Model Rationale (simplified, illustrative assumptions):
    - Every customer flagged as at-risk (TP or FP) receives a retention offer
      costing ``cost_fp`` (e.g. $50 discount / support bundle).
    - A churner who is NOT flagged (FN) is lost, costing ``cost_fn``
      (e.g. $200 of future margin).
    - Flagged churners (TP) are assumed to be retained by the offer.
    - Net benefit = cost of doing nothing (all churners lost) - cost with the model.

    Parameters
    ----------
    model : estimator
        Trained classifier with predict_proba capability.
    X_test : np.ndarray
        Test feature matrix.
    y_test : np.ndarray
        Ground truth binary churn labels (0 = retained, 1 = churned).
    cost_fp : float, default=50.0
        Cost of one retention offer (paid for every flagged customer).
    cost_fn : float, default=200.0
        Cost incurred for a missed churner (lost customer value).
    save_path : str or Path, optional
        Filepath to save the optimization plot.

    Returns
    -------
    dict
        Dictionary containing optimal_threshold, net_benefit, and threshold_analysis_df.
    """
    _set_chart_style()

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        raw_scores = model.decision_function(X_test)
        y_proba = 1 / (1 + np.exp(-raw_scores))
    else:
        raise AttributeError("Model must provide probability estimates for threshold optimization.")

    thresholds = np.linspace(0.05, 0.95, 91)
    results = []

    # Cost of doing nothing: every churner is lost
    total_positives = np.sum(y_test == 1)
    baseline_cost = total_positives * cost_fn

    for th in thresholds:
        y_pred = (y_proba >= th).astype(int)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        # Business cost: offers sent to every flagged customer + lost churners
        total_cost = ((tp + fp) * cost_fp) + (fn * cost_fn)
        # Net savings compared to doing nothing
        savings = baseline_cost - total_cost

        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        results.append({
            "Threshold": round(th, 3),
            "True_Positives": tp,
            "False_Positives": fp,
            "False_Negatives": fn,
            "True_Negatives": tn,
            "Total_Cost": total_cost,
            "Net_Benefit": savings,
            "Precision": prec,
            "Recall": rec,
            "F1_Score": f1,
        })

    analysis_df = pd.DataFrame(results)
    best_row = analysis_df.loc[analysis_df["Net_Benefit"].idxmax()]
    optimal_threshold = float(best_row["Threshold"])
    max_net_benefit = float(best_row["Net_Benefit"])

    # Plot threshold vs Net Benefit
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(
        analysis_df["Threshold"],
        analysis_df["Net_Benefit"],
        color=COLORS["teal"],
        linewidth=2.8,
        label="Net Financial Benefit ($)",
    )
    ax1.axvline(
        x=optimal_threshold,
        color=COLORS["coral"],
        linestyle="--",
        linewidth=2,
        label=f"Optimal Threshold ({optimal_threshold:.2f})",
    )
    ax1.scatter(
        [optimal_threshold],
        [max_net_benefit],
        color=COLORS["coral"],
        s=100,
        zorder=5,
    )

    ax1.set_xlabel("Classification Decision Threshold", fontsize=12)
    ax1.set_ylabel("Net Financial Benefit ($)", color=COLORS["teal"], fontsize=12)
    ax1.tick_params(axis="y", labelcolor=COLORS["teal"])

    # Secondary axis for F1 and Recall
    ax2 = ax1.twinx()
    ax2.plot(
        analysis_df["Threshold"],
        analysis_df["F1_Score"],
        color=COLORS["sky_blue"],
        linestyle=":",
        linewidth=2,
        label="F1 Score",
    )
    ax2.plot(
        analysis_df["Threshold"],
        analysis_df["Recall"],
        color=COLORS["plum"],
        linestyle="-.",
        linewidth=2,
        label="Recall",
    )
    ax2.set_ylabel("Metric Score", color="#555555", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="#555555")
    ax2.set_ylim(0, 1.05)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True)

    plt.title(
        f"Cost-Sensitive Threshold Optimization (Max Savings: ${max_net_benefit:,.2f})",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close()

    return {
        "optimal_threshold": optimal_threshold,
        "net_benefit": max_net_benefit,
        "threshold_analysis_df": analysis_df,
    }


def get_model_comparison_summary(
    results_df: pd.DataFrame,
    cv_results_df: pd.DataFrame,
) -> str:
    """Return formatted string summarizing model comparison results.

    Parameters
    ----------
    results_df : pd.DataFrame
        Evaluation results dataframe with test metrics.
    cv_results_df : pd.DataFrame
        Cross-validation results dataframe.

    Returns
    -------
    str
        Formatted markdown comparison report.
    """
    merged = pd.merge(results_df, cv_results_df, on="Model", how="left")
    top_model = merged.iloc[0]["Model"]
    top_f1 = merged.iloc[0]["F1-Score"]
    top_auc = merged.iloc[0]["ROC-AUC"]

    summary_lines = [
        "=" * 78,
        "                 CUSTOMER CHURN PREDICTION - MODEL BENCHMARK SUMMARY",
        "=" * 78,
        f"Top test-F1 model : {top_model}  (final model is selected by CV F1, see metrics.json)",
        f"Its test F1       : {top_f1:.4f} | ROC-AUC: {top_auc:.4f}",
        "-" * 78,
        "Performance Leaderboard (Sorted by Test F1-Score):",
        results_df.to_string(index=False),
        "-" * 78,
        "Cross-Validation Stability (5-Fold Stratified):",
        cv_results_df.to_string(index=False),
        "=" * 78,
    ]
    return "\n".join(summary_lines)


def train_and_evaluate(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    output_dir: Union[str, Path] = "reports/figures",
) -> dict:
    """Full pipeline: train → evaluate → CV → tune → SHAP → threshold.
    Return dict with all results.

    Parameters
    ----------
    X_train : np.ndarray
        Preprocessed training features.
    X_test : np.ndarray
        Preprocessed test features.
    y_train : np.ndarray
        Training binary target.
    y_test : np.ndarray
        Test binary target.
    feature_names : list of str
        List of all feature column names.
    output_dir : str or Path, default='reports/figures'
        Directory to store visual outputs and artifacts.

    Returns
    -------
    dict
        Comprehensive pipeline results including trained models, evaluation tables,
        tuned best model, charts, and threshold recommendations.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight = float(neg_count / max(pos_count, 1))

    print("Step 1/7: Training baseline classification models...")
    trained_models = train_models(X_train, y_train, get_models(scale_pos_weight))

    print("Step 2/7: 5-fold Stratified Cross-Validation of baselines...")
    cv_df = cross_validate_models(get_models(scale_pos_weight), X_train, y_train, cv=5)

    print("Step 3/7: Hyperparameter tuning (GridSearchCV) for Random Forest & XGBoost...")
    param_grids = get_default_param_grids()
    tuning_results: Dict[str, dict] = {}
    for base_name, grid in param_grids.items():
        if base_name not in trained_models:
            continue
        start = time.time()
        tuned, params, cv_f1 = tune_best_model(
            model=get_models(scale_pos_weight)[base_name],
            X_train=X_train,
            y_train=y_train,
            param_grid=grid,
            cv=5,
        )
        tuned._training_time = round(time.time() - start, 4)
        tuned_name = f"{base_name} (Tuned)"
        trained_models[tuned_name] = tuned
        tuning_results[tuned_name] = {"best_params": params, "cv_f1": cv_f1}
        cv_df = pd.concat(
            [cv_df, pd.DataFrame([{"Model": tuned_name, "CV_Mean_F1": cv_f1}])],
            ignore_index=True,
        )
        print(f"   -> {tuned_name}: {params} (CV F1 = {cv_f1})")
    cv_df = cv_df.sort_values("CV_Mean_F1", ascending=False).reset_index(drop=True)

    print("Step 4/7: Evaluating all models on the untouched holdout test set...")
    eval_df = evaluate_models(trained_models, X_test, y_test)

    # Overfitting diagnostics: train vs. test accuracy for every model
    overfit_records = []
    for name, mdl in trained_models.items():
        tr_acc = accuracy_score(y_train, mdl.predict(X_train))
        te_acc = accuracy_score(y_test, mdl.predict(X_test))
        overfit_records.append({
            "Model": name,
            "Train Accuracy": round(tr_acc, 4),
            "Test Accuracy": round(te_acc, 4),
            "Gap": round(tr_acc - te_acc, 4),
        })
    overfit_df = pd.DataFrame(overfit_records).sort_values("Gap").reset_index(drop=True)

    # Final model selection is based on CROSS-VALIDATED F1 (not the test set)
    best_model_name = str(cv_df.iloc[0]["Model"])
    tuned_model = trained_models[best_model_name]
    best_params = tuning_results.get(best_model_name, {}).get("best_params", {})
    best_cv_score = float(cv_df.iloc[0]["CV_Mean_F1"])
    print(f"   -> Selected final model (highest CV F1): {best_model_name}")

    print("Step 5/7: Generating diagnostic comparison charts...")
    plot_confusion_matrices(trained_models, X_test, y_test, save_path=out_path / "confusion_matrices.png")
    plot_roc_curves(trained_models, X_test, y_test, save_path=out_path / "roc_curves.png")
    plot_precision_recall_curves(trained_models, X_test, y_test, save_path=out_path / "precision_recall_curves.png")

    # Feature importance / SHAP use the tuned Random Forest (tree model -> fast, exact SHAP)
    explain_name = "Random Forest (Tuned)" if "Random Forest (Tuned)" in trained_models else best_model_name
    explain_model = trained_models[explain_name]
    try:
        plot_feature_importance(
            explain_model,
            feature_names,
            top_n=15,
            save_path=out_path / "feature_importance.png",
        )
    except Exception as e:
        print(f"[Warning] Feature importance plot skipped: {e}")

    print(f"Step 6/7: SHAP explainability ({explain_name})...")
    shap_results = {}
    try:
        shap_results = shap_analysis(
            explain_model,
            X_test[:300],
            feature_names,
            save_path=out_path / "shap",
        )
    except Exception as e:
        print(f"[Warning] SHAP analysis failed or skipped: {e}")

    print("Step 7/7: Cost-benefit threshold optimization...")
    threshold_results = optimize_threshold(
        tuned_model,
        X_test,
        y_test,
        save_path=out_path / "threshold_optimization.png",
    )

    summary_str = get_model_comparison_summary(eval_df, cv_df)
    print("\n" + summary_str)
    print("\nOverfitting check (train vs. test accuracy):")
    print(overfit_df.to_string(index=False))

    return {
        "trained_models": trained_models,
        "evaluation_df": eval_df,
        "cv_df": cv_df,
        "overfit_df": overfit_df,
        "tuning_results": tuning_results,
        "best_model_name": best_model_name,
        "best_model": tuned_model,
        "best_params": best_params,
        "best_cv_score": best_cv_score,
        "explained_model_name": explain_name,
        "shap_results": shap_results,
        "threshold_results": threshold_results,
        "summary": summary_str,
    }


if __name__ == "__main__":
    import json
    import sys

    # Allow running as `python -m src.model` or `python src/model.py` from the project root
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))

    from src.data_preprocessing import load_and_process_data
    from src.predict import save_artifacts

    data = load_and_process_data(PROJECT_ROOT / "data" / "raw" / "telco_churn.csv")
    results = train_and_evaluate(
        data["X_train"],
        data["X_test"],
        data["y_train"],
        data["y_test"],
        data["feature_names"],
        output_dir=PROJECT_ROOT / "reports" / "figures",
    )

    # Persist the selected model + fitted preprocessor for the app and CLI
    save_artifacts(
        results["best_model"],
        data["preprocessor"],
        data["feature_names"],
        model_dir=PROJECT_ROOT / "models",
    )

    # Persist result tables so the documentation can quote real numbers
    reports_dir = PROJECT_ROOT / "reports"
    results["evaluation_df"].to_csv(reports_dir / "model_results.csv", index=False)
    results["cv_df"].to_csv(reports_dir / "cv_results.csv", index=False)
    results["overfit_df"].to_csv(reports_dir / "overfitting_check.csv", index=False)
    th = results["threshold_results"]
    th["threshold_analysis_df"].to_csv(reports_dir / "threshold_analysis.csv", index=False)

    final_row = results["evaluation_df"].set_index("Model").loc[results["best_model_name"]]
    metrics = {
        "final_model": results["best_model_name"],
        "selection_criterion": "highest 5-fold CV F1 on the training set",
        "cv_f1": results["best_cv_score"],
        "best_params": {k: (v if v is None or isinstance(v, (int, float, str)) else str(v))
                        for k, v in results["best_params"].items()},
        "test_metrics": {k: float(final_row[k]) for k in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]},
        "tuning_results": {
            name: {"cv_f1": r["cv_f1"],
                   "best_params": {k: (v if v is None or isinstance(v, (int, float, str)) else str(v))
                                   for k, v in r["best_params"].items()}}
            for name, r in results["tuning_results"].items()
        },
        "optimal_threshold": th["optimal_threshold"],
        "threshold_net_benefit": th["net_benefit"],
        "threshold_cost_assumptions": {"offer_cost_per_flagged_customer": 50, "lost_customer_cost": 200},
        "shap_explained_model": results["explained_model_name"],
    }
    with open(reports_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved: models/, reports/model_results.csv, cv_results.csv, "
          "overfitting_check.csv, threshold_analysis.csv, metrics.json")
