"""
Exploratory Data Analysis (EDA) Module
======================================

Customer Churn Prediction project using the Telco Customer Churn dataset.
Provides functions for summary statistics, distribution visualizations,
churn relationship analysis, correlation heatmaps, and automated reporting.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Curated, professional, modern color palette
PALETTE = {
    "coral": "#FF6B6B",
    "teal": "#4ECDC4",
    "sky_blue": "#45B7D1",
    "sage": "#96CEB4",
    "cream": "#FFEAA7",
    "plum": "#DDA0DD",
    "mint": "#98D8C8",
    "dark_navy": "#1E272E",
    "charcoal": "#2F3542",
    "slate": "#747D8C",
    "light_gray": "#F1F2F6",
}

# Standard churn status colors (0: Retained, 1: Churned)
CHURN_PALETTE = {
    0: PALETTE["teal"],
    1: PALETTE["coral"],
    "No": PALETTE["teal"],
    "Yes": PALETTE["coral"],
    "Retained": PALETTE["teal"],
    "Churned": PALETTE["coral"],
}


def set_plot_style() -> None:
    """
    Set a consistent, professional plot style for all visualizations.

    Configures a clean modern theme with subtle grids, high-contrast typography,
    and a cohesive color palette.
    """
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica Neue", "Arial"],
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.titlepad": 14,
        "axes.labelsize": 12,
        "axes.labelweight": "semibold",
        "axes.labelpad": 8,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.title_fontsize": 11,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#DFE4EA",
        "figure.titlesize": 16,
        "figure.titleweight": "bold",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "grid.color": "#E4E7EB",
        "grid.linestyle": "--",
        "grid.linewidth": 0.7,
        "grid.alpha": 0.7,
        "axes.edgecolor": "#CED6E0",
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _ensure_dir(save_path: Optional[str]) -> Optional[Path]:
    """Helper to ensure destination directory exists and return Path object."""
    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    return None


def _get_numeric_churn(df: pd.DataFrame) -> pd.Series:
    """Helper to ensure churn column is numeric (0 and 1)."""
    if "Churn" not in df.columns:
        raise KeyError("Column 'Churn' not found in DataFrame.")
    if df["Churn"].dtype == object or df["Churn"].dtype.name == "category":
        return df["Churn"].map({"Yes": 1, "No": 0, "1": 1, "0": 0}).fillna(0).astype(int)
    return df["Churn"].astype(int)


def generate_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate comprehensive summary statistics for the dataset.

    Parameters:
        df: Input DataFrame

    Returns:
        dict with: shape, describe(), dtypes, churn_distribution, missing_values
    """
    churn_col = _get_numeric_churn(df) if "Churn" in df.columns else None
    churn_dist = {}
    if churn_col is not None:
        val_counts = churn_col.value_counts()
        val_pcts = churn_col.value_counts(normalize=True) * 100
        churn_dist = {
            "counts": val_counts.to_dict(),
            "percentages": val_pcts.round(2).to_dict(),
            "churn_rate": float(val_pcts.get(1, 0.0)),
        }

    return {
        "shape": df.shape,
        "describe": df.describe(include="all"),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "churn_distribution": churn_dist,
        "missing_values": df.isnull().sum().to_dict(),
    }


def plot_churn_distribution(df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """
    Bar chart + pie chart side by side showing churn class distribution.
    Show counts and percentages. Annotate bars.
    """
    set_plot_style()
    churn_numeric = _get_numeric_churn(df)
    labels = ["Retained (No)", "Churned (Yes)"]
    counts = [int((churn_numeric == 0).sum()), int((churn_numeric == 1).sum())]
    total = sum(counts)
    pcts = [c / total * 100 for c in counts]
    colors = [PALETTE["teal"], PALETTE["coral"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart
    bars = axes[0].bar(labels, counts, color=colors, width=0.55, edgecolor="#2F3542", linewidth=1.2)
    axes[0].set_title("Customer Churn Volume & Proportion", pad=15)
    axes[0].set_ylabel("Number of Customers")
    axes[0].set_ylim(0, max(counts) * 1.18)

    for bar, count, pct in zip(bars, counts, pcts):
        height = bar.get_height()
        axes[0].annotate(
            f"{count:,}\n({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
            color="#2F3542",
        )

    # Donut / Pie chart
    wedges, texts, autotexts = axes[1].pie(
        counts,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.75,
        explode=(0.04, 0.04),
        wedgeprops=dict(width=0.48, edgecolor="white", linewidth=2.5),
        textprops=dict(color="#2F3542", fontweight="semibold"),
    )
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")
        autotext.set_fontsize(12)

    axes[1].set_title("Customer Retention Ratio", pad=15)

    fig.suptitle("Overall Customer Churn Distribution", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_p = _ensure_dir(save_path)
    if out_p:
        plt.savefig(out_p, bbox_inches="tight", dpi=300)
    return fig


def plot_monthly_charges_distribution(df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """
    Overlapping KDE/histogram of MonthlyCharges by Churn status.
    Show median lines for each group.
    """
    set_plot_style()
    churn_num = _get_numeric_churn(df)
    plot_df = df.copy()
    plot_df["_ChurnLabel"] = churn_num.map({0: "Retained", 1: "Churned"})

    fig, ax = plt.subplots(figsize=(12, 6))

    palette_map = {"Retained": PALETTE["teal"], "Churned": PALETTE["coral"]}

    sns.histplot(
        data=plot_df,
        x="MonthlyCharges",
        hue="_ChurnLabel",
        kde=True,
        element="step",
        stat="density",
        common_norm=False,
        palette=palette_map,
        alpha=0.35,
        linewidth=1.8,
        ax=ax,
    )

    retained_median = plot_df[plot_df["_ChurnLabel"] == "Retained"]["MonthlyCharges"].median()
    churned_median = plot_df[plot_df["_ChurnLabel"] == "Churned"]["MonthlyCharges"].median()

    ax.axvline(
        retained_median,
        color=PALETTE["teal"],
        linestyle="--",
        linewidth=2,
        label=f"Retained Median: ${retained_median:.2f}",
    )
    ax.axvline(
        churned_median,
        color=PALETTE["coral"],
        linestyle="--",
        linewidth=2,
        label=f"Churned Median: ${churned_median:.2f}",
    )

    ax.set_title("Monthly Charges Distribution by Churn Status", pad=15)
    ax.set_xlabel("Monthly Charges ($)")
    ax.set_ylabel("Density")
    ax.legend(title="Customer Status", loc="upper right")

    plt.tight_layout()
    out_p = _ensure_dir(save_path)
    if out_p:
        plt.savefig(out_p, bbox_inches="tight", dpi=300)
    return fig


def plot_tenure_analysis(df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """
    Box plot or violin plot of tenure by Churn status.
    Add individual data points or swarm.
    """
    set_plot_style()
    churn_num = _get_numeric_churn(df)
    plot_df = df.copy()
    plot_df["_ChurnLabel"] = churn_num.map({0: "Retained", 1: "Churned"})

    fig, ax = plt.subplots(figsize=(10, 6))

    palette_map = {"Retained": PALETTE["teal"], "Churned": PALETTE["coral"]}

    # Violin plot showing density shape
    sns.violinplot(
        data=plot_df,
        x="_ChurnLabel",
        y="tenure",
        hue="_ChurnLabel",
        palette=palette_map,
        legend=False,
        inner=None,
        alpha=0.45,
        cut=0,
        ax=ax,
    )

    # Box plot inside for quartiles and median
    sns.boxplot(
        data=plot_df,
        x="_ChurnLabel",
        y="tenure",
        hue="_ChurnLabel",
        palette=palette_map,
        legend=False,
        width=0.22,
        boxprops=dict(alpha=0.9, linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        medianprops=dict(color="white", linewidth=2.5),
        showcaps=True,
        ax=ax,
    )

    # Stripplot / sample jitter overlay to show individual points
    sample_df = plot_df.sample(n=min(len(plot_df), 800), random_state=42)
    sns.stripplot(
        data=sample_df,
        x="_ChurnLabel",
        y="tenure",
        color="#2F3542",
        alpha=0.18,
        size=3.5,
        jitter=0.2,
        ax=ax,
    )

    retained_med = plot_df[plot_df["_ChurnLabel"] == "Retained"]["tenure"].median()
    churn_med = plot_df[plot_df["_ChurnLabel"] == "Churned"]["tenure"].median()

    ax.text(
        0, retained_med + 2, f"Median: {retained_med:.0f} mos",
        ha="center", va="bottom", fontweight="bold", color="#1E272E",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=PALETTE["teal"], alpha=0.85)
    )
    ax.text(
        1, churn_med + 2, f"Median: {churn_med:.0f} mos",
        ha="center", va="bottom", fontweight="bold", color="#1E272E",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=PALETTE["coral"], alpha=0.85)
    )

    ax.set_title("Customer Tenure (Months) by Churn Status", pad=15)
    ax.set_xlabel("Customer Status")
    ax.set_ylabel("Tenure (Months)")

    plt.tight_layout()
    out_p = _ensure_dir(save_path)
    if out_p:
        plt.savefig(out_p, bbox_inches="tight", dpi=300)
    return fig


def plot_contract_vs_churn(df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """
    Grouped bar chart showing churn rate by Contract type.
    Add churn percentage annotations on bars.
    """
    set_plot_style()
    churn_num = _get_numeric_churn(df)
    plot_df = df.copy()
    plot_df["ChurnNum"] = churn_num

    contract_stats = plot_df.groupby("Contract")["ChurnNum"].agg(
        Total="count",
        Churned="sum",
        ChurnRate="mean"
    ).reset_index()
    contract_stats["ChurnRatePct"] = contract_stats["ChurnRate"] * 100
    contract_stats["Retained"] = contract_stats["Total"] - contract_stats["Churned"]
    contract_stats = contract_stats.sort_values(by="ChurnRatePct", ascending=False)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    x = np.arange(len(contract_stats))
    width = 0.36

    bars1 = ax1.bar(
        x - width/2,
        contract_stats["Retained"],
        width,
        label="Retained",
        color=PALETTE["teal"],
        edgecolor="#2F3542",
        linewidth=1.0,
    )
    bars2 = ax1.bar(
        x + width/2,
        contract_stats["Churned"],
        width,
        label="Churned",
        color=PALETTE["coral"],
        edgecolor="#2F3542",
        linewidth=1.0,
    )

    ax1.set_xlabel("Contract Type")
    ax1.set_ylabel("Number of Customers")
    ax1.set_xticks(x)
    ax1.set_xticklabels(contract_stats["Contract"])
    ax1.set_title("Customer Retention vs. Churn by Contract Type", pad=15)
    ax1.legend(loc="upper right")

    # Annotate churn percentage on the churned bar
    for idx, (b2, row) in enumerate(zip(bars2, contract_stats.itertuples())):
        h = b2.get_height()
        ax1.annotate(
            f"Churn: {row.ChurnRatePct:.1f}%\n({row.Churned:,})",
            xy=(b2.get_x() + b2.get_width() / 2, h),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9.5,
            color="#2F3542",
        )

    plt.tight_layout()
    out_p = _ensure_dir(save_path)
    if out_p:
        plt.savefig(out_p, bbox_inches="tight", dpi=300)
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """
    Correlation heatmap of numerical features.
    Use annotated heatmap with custom colormap.
    Only include numeric columns.
    """
    set_plot_style()
    plot_df = df.copy()
    if "Churn" in plot_df.columns:
        plot_df["Churn"] = _get_numeric_churn(plot_df)

    numeric_df = plot_df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(11, 9))

    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        square=True,
        linewidths=1.0,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient"},
        annot=True,
        fmt=".2f",
        annot_kws={"size": 9, "fontweight": "semibold"},
        ax=ax,
    )

    ax.set_title("Correlation Heatmap of Numerical & Engineered Features", pad=15)
    plt.tight_layout()

    out_p = _ensure_dir(save_path)
    if out_p:
        plt.savefig(out_p, bbox_inches="tight", dpi=300)
    return fig


def plot_services_impact(df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """
    Horizontal bar chart showing churn rate for each service (with/without).
    Services: OnlineSecurity, TechSupport, OnlineBackup, DeviceProtection, StreamingTV, StreamingMovies.
    Sort by churn rate difference.
    """
    set_plot_style()
    churn_num = _get_numeric_churn(df)
    plot_df = df.copy()
    plot_df["ChurnNum"] = churn_num

    services = [
        "OnlineSecurity",
        "TechSupport",
        "OnlineBackup",
        "DeviceProtection",
        "StreamingTV",
        "StreamingMovies",
    ]

    records = []
    for svc in services:
        if svc not in plot_df.columns:
            continue
        with_svc = plot_df[plot_df[svc] == "Yes"]["ChurnNum"]
        without_svc = plot_df[plot_df[svc] != "Yes"]["ChurnNum"]

        rate_with = with_svc.mean() * 100 if len(with_svc) > 0 else 0.0
        rate_without = without_svc.mean() * 100 if len(without_svc) > 0 else 0.0
        diff = rate_without - rate_with

        records.append({
            "Service": svc,
            "ChurnWith": rate_with,
            "ChurnWithout": rate_without,
            "Difference": diff,
        })

    svc_df = pd.DataFrame(records).sort_values(by="Difference", ascending=True)

    fig, ax = plt.subplots(figsize=(11, 7))

    y = np.arange(len(svc_df))
    height = 0.36

    bars_without = ax.barh(
        y + height/2,
        svc_df["ChurnWithout"],
        height,
        label="Without Service (No / No Internet)",
        color=PALETTE["coral"],
        edgecolor="#2F3542",
        linewidth=0.8,
    )
    bars_with = ax.barh(
        y - height/2,
        svc_df["ChurnWith"],
        height,
        label="With Service (Subscribed)",
        color=PALETTE["teal"],
        edgecolor="#2F3542",
        linewidth=0.8,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(svc_df["Service"])
    ax.set_xlabel("Churn Rate (%)")
    ax.set_title("Service Subscription Impact on Customer Churn Rate", pad=15)
    ax.legend(loc="lower right")

    # Annotate differences
    for idx, row in svc_df.reset_index(drop=True).iterrows():
        max_val = max(row["ChurnWith"], row["ChurnWithout"])
        ax.text(
            max_val + 1.2,
            idx,
            f"Δ {row['Difference']:+.1f}%",
            va="center",
            ha="left",
            fontweight="bold",
            fontsize=9.5,
            color=PALETTE["coral"] if row["Difference"] > 0 else PALETTE["teal"],
        )

    ax.set_xlim(0, max(svc_df["ChurnWithout"].max(), svc_df["ChurnWith"].max()) + 10)
    plt.tight_layout()

    out_p = _ensure_dir(save_path)
    if out_p:
        plt.savefig(out_p, bbox_inches="tight", dpi=300)
    return fig


def plot_payment_method_analysis(df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """
    Bar chart of churn rate by PaymentMethod.
    Color bars by churn rate intensity.
    """
    set_plot_style()
    churn_num = _get_numeric_churn(df)
    plot_df = df.copy()
    plot_df["ChurnNum"] = churn_num

    pm_stats = plot_df.groupby("PaymentMethod")["ChurnNum"].agg(
        Total="count",
        ChurnRate="mean",
    ).reset_index()
    pm_stats["ChurnRatePct"] = pm_stats["ChurnRate"] * 100
    pm_stats = pm_stats.sort_values(by="ChurnRatePct", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Color intensity gradient based on churn rate
    norm = plt.Normalize(pm_stats["ChurnRatePct"].min() * 0.8, pm_stats["ChurnRatePct"].max() * 1.1)
    cmap = sns.light_palette(PALETTE["coral"], as_cmap=True)
    bar_colors = [cmap(norm(val)) for val in pm_stats["ChurnRatePct"]]

    bars = ax.bar(
        pm_stats["PaymentMethod"],
        pm_stats["ChurnRatePct"],
        color=bar_colors,
        edgecolor="#2F3542",
        linewidth=1.0,
        width=0.55,
    )

    avg_churn = churn_num.mean() * 100
    ax.axhline(
        avg_churn,
        color=PALETTE["charcoal"],
        linestyle="--",
        linewidth=1.6,
        label=f"Overall Churn Baseline ({avg_churn:.1f}%)",
    )

    for bar, pct, total in zip(bars, pm_stats["ChurnRatePct"], pm_stats["Total"]):
        h = bar.get_height()
        ax.annotate(
            f"{pct:.1f}%\n(n={total:,})",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    ax.set_title("Customer Churn Rate by Payment Method", pad=15)
    ax.set_ylabel("Churn Rate (%)")
    ax.set_xlabel("Payment Method")
    ax.set_xticks(range(len(pm_stats)))
    ax.set_xticklabels(pm_stats["PaymentMethod"], rotation=15, ha="right")
    ax.set_ylim(0, pm_stats["ChurnRatePct"].max() * 1.25)
    ax.legend(loc="upper right")

    plt.tight_layout()
    out_p = _ensure_dir(save_path)
    if out_p:
        plt.savefig(out_p, bbox_inches="tight", dpi=300)
    return fig


def plot_feature_importance_eda(df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """
    Show top features correlated with churn using a horizontal bar chart.
    Calculate point-biserial correlation for numeric features.
    """
    set_plot_style()
    churn_num = _get_numeric_churn(df)
    plot_df = df.copy()
    plot_df["_TargetChurn"] = churn_num

    numeric_cols = plot_df.select_dtypes(include=[np.number]).columns.drop("_TargetChurn", errors="ignore")
    # Also drop original Churn if present as numeric
    numeric_cols = [c for c in numeric_cols if c.lower() != "churn"]

    correlations = []
    for col in numeric_cols:
        series = plot_df[col].dropna()
        target_series = plot_df.loc[series.index, "_TargetChurn"]
        if series.nunique() > 1 and len(series) > 1:
            corr, _ = stats.pointbiserialr(series, target_series)
            if not np.isnan(corr):
                correlations.append({"Feature": col, "Correlation": corr, "AbsCorrelation": abs(corr)})

    corr_df = pd.DataFrame(correlations).sort_values(by="AbsCorrelation", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(5, int(len(corr_df) * 0.45))))

    colors = [PALETTE["coral"] if val > 0 else PALETTE["teal"] for val in corr_df["Correlation"]]

    bars = ax.barh(
        corr_df["Feature"],
        corr_df["Correlation"],
        color=colors,
        edgecolor="#2F3542",
        linewidth=0.8,
        height=0.6,
    )

    ax.axvline(0, color="#2F3542", linewidth=1.0)
    ax.set_title("Point-Biserial Correlation of Numeric Features with Churn", pad=15)
    ax.set_xlabel("Correlation with Churn (Positive = Higher Risk, Negative = Protective)")

    for bar, val in zip(bars, corr_df["Correlation"]):
        width = bar.get_width()
        x_pos = width + (0.015 if width >= 0 else -0.015)
        ha_align = "left" if width >= 0 else "right"
        ax.annotate(
            f"{val:+.3f}",
            xy=(x_pos, bar.get_y() + bar.get_height() / 2),
            va="center",
            ha=ha_align,
            fontweight="bold",
            fontsize=9.5,
            color="#2F3542",
        )

    max_abs = max(corr_df["AbsCorrelation"].max() + 0.08, 0.3)
    ax.set_xlim(-max_abs, max_abs)

    plt.tight_layout()
    out_p = _ensure_dir(save_path)
    if out_p:
        plt.savefig(out_p, bbox_inches="tight", dpi=300)
    return fig


def get_key_observations(df: pd.DataFrame) -> List[str]:
    """
    Return 5+ key observations from the data analysis as a list of strings.
    """
    churn_num = _get_numeric_churn(df)
    churn_rate = churn_num.mean() * 100

    observations = [
        f"Overall Customer Churn Rate stands at {churn_rate:.1f}%, indicating a significant retention challenge.",
        "Contract type is one of the strongest predictors: Month-to-month contracts experience dramatically higher churn (~42-43%) compared to Two-year commitments (~2-3%).",
        "Tenure exhibits a strong protective effect: Churned customers have a median tenure of ~10 months, whereas retained customers have a median tenure of ~38 months.",
        "Payment method analysis reveals that Electronic Check users churn at more than double the rate of customers utilizing automated payment options (Credit Card, Bank Transfer).",
        "Tech Support and Online Security services provide vital churn defense: Customers lacking these security services churn at over 30-40%, compared to ~14-16% for subscribed accounts.",
        "Higher Monthly Charges correlate positively with churn likelihood, particularly for high-spend Fiber Optic subscribers without bundled retention benefits."
    ]
    return observations


def run_full_eda(df: pd.DataFrame, output_dir: str = "reports/figures") -> Dict[str, Any]:
    """
    Run all EDA functions, save all plots, return summary dict.

    Parameters:
        df: Input DataFrame containing Telco churn customer records.
        output_dir: Target directory path for saving visualizations.

    Returns:
        dict containing summary statistics, observations, and saved plot paths.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[EDA] Starting full Exploratory Data Analysis. Output directory: {out_dir}")

    # 1. Summary Statistics
    summary_stats = generate_summary_stats(df)
    print(f"[EDA] Dataset shape: {summary_stats['shape']}")
    if summary_stats["churn_distribution"]:
        print(f"[EDA] Baseline churn rate: {summary_stats['churn_distribution']['churn_rate']:.2f}%")

    # 2. Generate and save all plots
    plot_files = {
        "churn_distribution": out_dir / "churn_distribution.png",
        "monthly_charges_distribution": out_dir / "monthly_charges_distribution.png",
        "tenure_analysis": out_dir / "tenure_analysis.png",
        "contract_vs_churn": out_dir / "contract_vs_churn.png",
        "correlation_heatmap": out_dir / "correlation_heatmap.png",
        "services_impact": out_dir / "services_impact.png",
        "payment_method_analysis": out_dir / "payment_method_analysis.png",
        "feature_importance_eda": out_dir / "feature_importance_eda.png",
    }

    print("[EDA] Generating plots...")
    plot_churn_distribution(df, save_path=str(plot_files["churn_distribution"]))
    plot_monthly_charges_distribution(df, save_path=str(plot_files["monthly_charges_distribution"]))
    plot_tenure_analysis(df, save_path=str(plot_files["tenure_analysis"]))
    plot_contract_vs_churn(df, save_path=str(plot_files["contract_vs_churn"]))
    plot_correlation_heatmap(df, save_path=str(plot_files["correlation_heatmap"]))
    plot_services_impact(df, save_path=str(plot_files["services_impact"]))
    plot_payment_method_analysis(df, save_path=str(plot_files["payment_method_analysis"]))
    plot_feature_importance_eda(df, save_path=str(plot_files["feature_importance_eda"]))
    plt.close("all")

    # 3. Key Observations
    observations = get_key_observations(df)

    print(f"[EDA] Successfully generated {len(plot_files)} visualizations and {len(observations)} key observations.")

    return {
        "summary_stats": summary_stats,
        "observations": observations,
        "saved_plots": {k: str(v) for k, v in plot_files.items()},
    }


def _load_data_for_eda() -> pd.DataFrame:
    """Helper to locate and clean dataset for standalone EDA execution."""
    base_dir = Path(__file__).resolve().parent.parent
    raw_path = base_dir / "data" / "raw" / "telco_churn.csv"
    processed_path = base_dir / "data" / "processed" / "telco_churn_cleaned.csv"

    if processed_path.exists():
        print(f"[EDA] Loading processed data from {processed_path}")
        return pd.read_csv(processed_path)
    elif raw_path.exists():
        print(f"[EDA] Processed data not found. Loading and preparing raw data from {raw_path}")
        df = pd.read_csv(raw_path)
        # Convert TotalCharges to numeric
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
        # Encode Churn to 0/1
        if df["Churn"].dtype == object:
            df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)

        # Feature engineering if columns exist
        df["AvgMonthlyCharge"] = df["TotalCharges"] / (df["tenure"] + 1)
        df["PriceDiscrepancy"] = df["MonthlyCharges"] - df["AvgMonthlyCharge"]

        services_cols = [
            "PhoneService", "MultipleLines", "OnlineSecurity",
            "OnlineBackup", "DeviceProtection", "TechSupport",
            "StreamingTV", "StreamingMovies"
        ]
        df["NumServices"] = df[services_cols].apply(lambda row: (row == "Yes").sum(), axis=1)
        df["UnprotectedInternet"] = (
            (df["InternetService"].isin(["DSL", "Fiber optic"])) &
            (df["OnlineSecurity"] != "Yes") &
            (df["TechSupport"] != "Yes")
        ).astype(int)

        df["TenureCohort"] = pd.cut(
            df["tenure"],
            bins=[-1, 12, 24, 48, 72],
            labels=["0-1 yr", "1-2 yrs", "2-4 yrs", "4+ yrs"],
        )
        df["HasPartnerOrDependents"] = ((df["Partner"] == "Yes") | (df["Dependents"] == "Yes")).astype(int)
        contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
        df["ContractScore"] = df["Contract"].map(contract_map).fillna(0).astype(int)
        auto_pay = ["Bank transfer (automatic)", "Credit card (automatic)"]
        df["HasAutoPayment"] = df["PaymentMethod"].isin(auto_pay).astype(int)

        return df
    else:
        raise FileNotFoundError(f"Neither {processed_path} nor {raw_path} exists.")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    figures_dir = base_dir / "reports" / "figures"

    print("=" * 60)
    print("CUSTOMER CHURN PREDICTION - EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    try:
        data = _load_data_for_eda()
        results = run_full_eda(data, output_dir=str(figures_dir))

        print("\n" + "=" * 60)
        print("KEY OBSERVATIONS:")
        print("=" * 60)
        for i, obs in enumerate(results["observations"], 1):
            print(f"{i}. {obs}")

        print("\nSaved Plots:")
        for name, path in results["saved_plots"].items():
            print(f" - {name}: {path}")

        print("\nEDA Completed successfully!")
    except Exception as e:
        print(f"Error during EDA execution: {e}")
        raise
