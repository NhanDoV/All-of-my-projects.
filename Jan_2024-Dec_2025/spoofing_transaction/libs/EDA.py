import pandas as pd
import numpy as np
import scipy.stats as ss
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ================================= 1. Data Understanding & Profiling ========================================= #
# ============================================ SUMMARY ======================================================== #
# ..............................................................................................................#
def full_profile_report(df: pd.DataFrame) -> dict:
    report = {
        "duplicates": check_dupl_and_uniqueness(df),
        "missing_values": null_analytic(df),
        "categorical": categorical_profiling(df),
        "numerical": numerical_profiling(df),
        "timestamp_summary": timestamp_summary(df),
        "correlation": correlation_overview(df)
    }
    return report

# .......................................... 1.1. Data Validation ..............................................#
def check_dupl_and_uniqueness(df: pd.DataFrame, 
                              key_cols: list[str] = ['TransID', 'CardNo', 'EmployeeID', 'link']) -> pd.DataFrame:
    """
        Check duplicate counts and uniqueness for key fields, default:
            - TransID
            - CardNo
            - EmployeeID
            - link
        Returns a DataFrame summary.
    """
    summary = {}

    # Column-wise checks
    for col in key_cols:
        summary[col] = {
            "n_unique": df[col].nunique(dropna=True),
            "n_total": df[col].shape[0],
            "n_duplicates": df[col].duplicated().sum(),
            "is_unique": df[col].is_unique
        }

    # Multi-column duplicate check
    summary["ROW_DUPLICATES (all 4 fields)"] = {
        "n_duplicates": df[key_cols].duplicated().sum()
    }

    return pd.DataFrame(summary).T

def null_analytic(df : pd.DataFrame) -> pd.DataFrame:
    """
        Count & calculate on columns having missed values
    """
    # number of whole observations
    n = len(df)
    
    # Count all null values at all columns
    null_df = df.isnull().sum()
    
    # Extract exactly which columns existed null
    null_df = null_df[null_df > 0].sort_values(ascending = False).reset_index()

    # Rename columns (since your dataframe currently is indexed from a Series)
    null_df.columns = ['column', 'count_null']
    
    # Calculate the corresponding percentage
    null_df['percentage'] = null_df['count_null'] / n    

    return null_df

# .......................................... 1.2. Timestamps data ..............................................#
def timestamp_summary(
        df: pd.DataFrame,
        ts_cols: list[str] = ["TransDate", "DVRStart", "start_shift", "end_shift"]
    ) -> pd.DataFrame:
    """
        Summary max, min and time-range of all the timestamps columns
    """
    summary = {}

    for col in ts_cols:
        s = pd.to_datetime(df[col], errors="coerce")

        summary[col] = {
            "n_missing": s.isna().sum(),
            "min": s.min(),
            "max": s.max(),
            "n_unique": s.nunique(),
            "time_range_days": (s.max() - s.min()).days if s.notna().any() else None
        }

    return pd.DataFrame(summary).T

# ......................................... 1.3. Categorical data ..............................................#
def categorical_profiling(
        df: pd.DataFrame,
        target_col: str = "Label",
        cols: list[str] = ["Payment", "RegionName", "SiteName", "exception_type"]
    ) -> dict:
    """
        Profiling for categorical columns:
            - Frequency counts
            - Percentage
            - Fraud rate per category
            - Cardinality (returned separately)
        Returns a dict: {col: {"summary": df, "cardinality": int}}
    """

    results = {}

    # Ensure target not included
    cols = [c for c in cols if c != target_col and c in df.columns]

    for col in cols:

        freq = df[col].value_counts(dropna=False).rename("count")
        pct = (freq / len(df)).rename("percentage")

        fraud_rate = (
            df.groupby(col)[target_col]
            .mean()
            .rename("fraud_rate")
        )

        summary = pd.concat([freq, pct, fraud_rate], axis=1)

        results[col] = {
            "summary": summary.sort_values("count", ascending=False),
            "cardinality": df[col].nunique(dropna=True)
        }

    return results

def plot_full_category_overview(summary, col, bar_codes = ["#6495ED", "#FA5F55"]):
    """
        Full combined view of ALL categories:
            - Left y-axis: Count (bar)
            - Right y-axis: Fraud Rate (line)
    """
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x = summary[col],
            y = summary["count"],
            name = "Count",
            marker_color = bar_codes[0],
            opacity=0.7
        )
    )

    fig.add_trace(
        go.Scatter(
            x = summary[col],
            y = summary["fraud_rate"],
            mode = "lines+markers",
            name = "Fraud Rate",
            yaxis = "y2",
            marker=dict(color = bar_codes[1], size = 8)
        )
    )

    fig.update_layout(
        title=f"{col} — Full Overview (Count + Fraud Rate)",
        xaxis=dict(title=col, tickangle=45),
        yaxis=dict(title="Count"),
        yaxis2=dict(
            title="Fraud Rate",
            overlaying="y",
            side="right",
            showgrid=False
        ),
        width=1800,
        height=450,
        legend=dict(orientation="h")
    )

    fig.show()

def plot_topN_count_vs_topN_fraudrate(summary, col, top_N_displayed, 
                                      bar_codes = ["#6495ED", "#FA5F55"]):
    """
        Left panel: Top N by Count  (count main axis)
        Right panel: Top N by Fraud Rate (fraud rate main axis)
    """
    # Assign colors
    count_color, rate_color = bar_codes
    
    # Left – Top N by count
    top_count_df = summary.sort_values("count", ascending=False).head(top_N_displayed).reset_index(drop=True)

    # Right – Top N by fraud rate
    top_fraudrate_df = summary.sort_values("fraud_rate", ascending=False).head(top_N_displayed).reset_index(drop=True)

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"secondary_y": True}, {"secondary_y": True}]],
        subplot_titles=[
            f"Top {top_N_displayed} by Count ({col})",
            f"Top {top_N_displayed} by Fraud Rate ({col})"
        ]
    )

    # ============================================
    # LEFT SUBPLOT (count main axis)
    # ============================================
    fig.add_trace(
        go.Bar(
            x = top_count_df[col],
            y = top_count_df["count"],
            name = "Count",
            opacity = 0.7,
            marker_color = count_color,
            showlegend = True
        ),
        row = 1, col = 1, secondary_y = False
    )

    fig.add_trace(
        go.Scatter(
            x = top_count_df[col],
            y = top_count_df["fraud_rate"],
            mode = "lines+markers",
            name = "Fraud Rate",
            marker = dict(color = rate_color, size = 8)
        ),
        row = 1, col = 1, secondary_y=True
    )

    # ============================================
    # RIGHT SUBPLOT (fraud_rate main axis)
    # ============================================
    fig.add_trace(
        go.Bar(
            x = top_fraudrate_df[col],
            y = top_fraudrate_df["fraud_rate"],
            name = "Fraud Rate",
            opacity = 0.7,
            marker_color = count_color,
            showlegend = False
        ),
        row = 1, col = 2, secondary_y=False         # FRAUD RATE is primary axis
    )

    fig.add_trace(
        go.Scatter(
            x = top_fraudrate_df[col],
            y = top_fraudrate_df["count"],
            mode = "lines+markers",
            name = "Count",
            marker = dict(color=rate_color, size=8),
            showlegend = False
        ),
        row=1, col=2, secondary_y=True          # COUNT moves to secondary axis
    )

    # ============================================
    # Layout
    # ============================================
    fig.update_layout(
        width=1800,
        height=500,
        legend=dict(
            orientation="h",
            y=1.1, x=0.45,   # Top mid
            yanchor="bottom"
            )
    )

    # Left axis labels
    fig.update_yaxes(title_text="Count", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Fraud Rate", row=1, col=1, secondary_y=True)

    # Right axis labels
    fig.update_yaxes(title_text="Fraud Rate", row=1, col=2, secondary_y=False)
    fig.update_yaxes(title_text="Count", row=1, col=2, secondary_y=True)

    fig.update_xaxes(title_text=col, tickangle=45, row=1, col=1)
    fig.update_xaxes(title_text=col, tickangle=45, row=1, col=2)

    fig.show()

def plot_categorical_eda_plotly_per_col(info, col, top_N_displayed: int = 10):
    """
        cate_dict = categorical_profiling(df)

        Draw:
            1. Full overview of all categories (Count + Fraud rate)
            2. Comparison: Top N by Count vs Top N by Fraud Rate

        To plot all cate-cols at once:
        >>> top_N_displayed = 10
        >>> for col, info in cate_dict.items():
                plot_categorical_eda_plotly_per_cols(info, col, top_N_displayed)
    """
    summary = info["summary"].copy()
    summary = summary.reset_index().rename(columns={"index": col})
    summary = summary.sort_values("count", ascending=False)

    print(f"\n==================== {col} ====================")
    print(f"Cardinality: {info['cardinality']}")
    print(summary.head(10))

    # Plot full overview
    plot_full_category_overview(summary, col)

    # Plot top-N by count & top-N by fraud rate
    plot_topN_count_vs_topN_fraudrate(summary, col, top_N_displayed)

# .......................................... 1.4. Numerical data ..............................................#
def numerical_profiling(df: pd.DataFrame, target_col: str = "Label") -> dict:
    """
        Profiling for numerical columns (excluding the target label):
            - summary stats
            - missing count
            - outlier detection (IQR-based)
            - comparison: fraud vs non-fraud
        Returns a dictionary {col_name: DataFrame}.
    """

    # Select numeric columns EXCEPT target
    num_cols = [
        col for col in df.select_dtypes(include=["int", "float"]).columns
        if col != target_col
    ]

    results = {}

    for col in num_cols:
        series = df[col]

        # Summary stats
        summary = series.describe(percentiles=[0.01, 0.05, 0.95, 0.99]).to_frame(name="value")

        # Missing & zeros
        summary.loc["missing_count"] = series.isna().sum()
        summary.loc["zero_count"] = (series == 0).sum()

        # Outlier detection (IQR)
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        outlier_mask = (series < Q1 - 1.5 * IQR) | (series > Q3 + 1.5 * IQR)
        summary.loc["outliers"] = outlier_mask.sum()
        summary.loc["outliers_pct"] = outlier_mask.mean()

        # Fraud vs non-fraud mean comparison
        fraud_stats = df.groupby(target_col)[col].mean().rename("mean_by_label")
        summary = summary.join(fraud_stats)

        results[col] = summary

    return results

def numerical_histogram(series, col):
    """
        # ───────────────────────────────────────────────
        #    Histogram + KDE (overall distribution)
        # ───────────────────────────────────────────────
    """
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=series,
            name="Histogram",
            nbinsx=50,
            opacity=0.65
        )
    )

    fig.add_trace(
        go.Scatter(
            x=series.dropna().sort_values(),
            y=pd.Series(series.dropna()).sort_values().rolling(200, min_periods=1).mean(),
            mode="lines",
            name="Smoothed trend"
        )
    )

    fig.update_layout(
        title=f"{col} — Distribution (Histogram + Smoothed Trend)",
        xaxis_title=col,
        yaxis_title="Frequency",
        width=1200,
        height=400
    )
    fig.show()

def numerical_boxplot(series, col):
    """
        # ───────────────────────────────────────────────
        #               Boxplot (outliers)
        # ───────────────────────────────────────────────
    """
    fig = go.Figure()
    fig.add_trace(
        go.Box(
            y=series,
            name=col,
            boxpoints="outliers"
        )
    )
    fig.update_layout(
        title=f"{col} — Boxplot (Outlier Detection)",
        yaxis_title=col,
        width=900,
        height=350
    )
    fig.show()

def numerical_ingroup(col, target_col):
        """
            # ───────────────────────────────────────────────
            #       Fraud vs Non-fraud Distribution
            # ───────────────────────────────────────────────
        """
        fraud_df = df[df[target_col] == 1][col]
        legit_df = df[df[target_col] == 0][col]

        fig = go.Figure()

        fig.add_trace(
            go.Histogram(
                x=legit_df,
                nbinsx=50,
                opacity=0.6,
                name="Non-Fraud"
            )
        )
        fig.add_trace(
            go.Histogram(
                x=fraud_df,
                nbinsx=50,
                opacity=0.6,
                name="Fraud"
            )
        )

        fig.update_layout(
            title=f"{col} — Fraud vs Non-Fraud Distribution",
            barmode="overlay",
            xaxis_title=col,
            yaxis_title="Count",
            width=1200,
            height=400
        )
        fig.show()

def plot_numerical_eda_plotly_percol(summary, col, df, target_col="Label"):
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            f"{col} — Distribution (Histogram + Smoothed Trend)",
            f"{col} — Boxplot (Outlier Detection)",
            f"{col} — Fraud vs Non-Fraud Distribution",
        ],
        specs=[[{}, {}, {}]]
    )

    series = df[col]
    fraud_df = df[df[target_col] == 1][col]
    legit_df = df[df[target_col] == 0][col]

    # Histogram + KDE (Column 1)
    fig.add_trace(
        go.Histogram(
            x=series,
            nbinsx=50,
            opacity=0.65,
            name="Histogram"
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=series.dropna().sort_values(),
            y=pd.Series(series.dropna()).sort_values().rolling(200, min_periods=1).mean(),
            mode="lines",
            name="Smoothed Trend"
        ),
        row=1, col=1
    )

    # Boxplot (Column 2)
    fig.add_trace(
        go.Box(
            y=series,
            boxpoints="outliers",
            name=col
        ),
        row=1, col=2
    )

    # Fraud vs Non-Fraud Distribution (Column 3)
    fig.add_trace(
        go.Histogram(
            x=legit_df,
            nbinsx=50,
            opacity=0.6,
            name="Non-Fraud"
        ),
        row=1, col=3
    )
    fig.add_trace(
        go.Histogram(
            x=fraud_df,
            nbinsx=50,
            opacity=0.6,
            name="Fraud"
        ),
        row=1, col=3
    )

    fig.update_layout(
        width=1800,
        height=500,
        barmode="overlay",
        showlegend=True,
        title_text=f"Numerical EDA for {col}",
        legend=dict(
            orientation="h",
            y=1.1,
            yanchor="bottom",
            x=0.5,
            xanchor="center")
    )

    # Update axis titles individually if needed
    fig.update_xaxes(title_text=col, row=1, col=1)
    fig.update_xaxes(title_text=col, row=1, col=3)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text=col, row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=3)

    fig.show()

# .......................................... 1.5. Correlation ..............................................#
def cramers_v(x, y):
    """

    """
    confusion = pd.crosstab(x, y)
    chi2 = ss.chi2_contingency(confusion, correction=False)[0]
    n = confusion.sum().sum()
    phi2 = chi2 / n
    r, k = confusion.shape
    return np.sqrt(phi2 / min(k - 1, r - 1))

def correlation_overview(
        df: pd.DataFrame,
        target_col: str = "Label",
        categorical_cols: list[str] = ["Payment", "RegionName", "SiteName", "exception_type"]
    ) -> dict:
    """
    Correlation profiling:
        - Numeric correlation matrix
        - Numeric vs target correlation
        - Cramér's V for categorical-target correlation
    """

    results = {}

    # 1) Numeric correlation matrix
    num_df = df.select_dtypes(include=["int", "float"])
    results["numeric_corr"] = num_df.corr()

    # 2) Numeric correlation with target
    if target_col in num_df.columns:
        num_df_no_target = num_df.drop(columns=target_col)
        results["target_corr"] = num_df_no_target.corrwith(df[target_col]).sort_values(ascending=False)
    else:
        results["target_corr"] = None

    # 3) Categorical correlation w.r.t. target
    cat_corr = {}
    for col in categorical_cols:
        if col in df.columns:
            try:
                cat_corr[col] = cramers_v(df[col], df[target_col])
            except Exception:
                cat_corr[col] = np.nan

    results["categorical_corr"] = pd.DataFrame.from_dict(cat_corr, orient="index", columns=["CramersV"])

    return results

def EDA_corr_heatmap_per_col(corr_results, target_col="Label"):
    """
    Visualize correlation profiling results:
      - Numeric correlation matrix heatmap
      - Numeric vs target correlation bar chart
      - Categorical vs target correlation bar chart (Cramér's V)

    corr_results: output of correlation_overview function (dict of DataFrames)
    """

    numeric_corr = corr_results.get("numeric_corr")
    target_corr = corr_results.get("target_corr")
    cat_corr = corr_results.get("categorical_corr")

    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=[
                            "Numeric Correlation Matrix",
                            f"Numeric Features vs {target_col} Correlation",
                            f"Categorical Features vs {target_col} (Cramér's V)"
                        ],
                        specs=[[{"type": "heatmap"}, {"type": "bar"}, {"type": "bar"}]])

    # Numeric correlation matrix heatmap
    fig.add_trace(
        go.Heatmap(
            z=numeric_corr.values,
            x=numeric_corr.columns,
            y=numeric_corr.index,
            colorscale="Viridis",
            colorbar=dict(title="Corr"),
            text=numeric_corr.round(2).astype(str),  # Add this line
            texttemplate="%{text}",                   # Add this line
            textfont={"size": 10}                     # Add this line (optional)
        ),
        row=1, col=1
    )

    # Numeric vs target bar chart
    if target_corr is not None:
        fig.add_trace(
            go.Bar(
                x=target_corr.index,
                y=target_corr.values,
                marker_color="blue",
                text=target_corr.round(2).astype(str),    # Add this line
                textposition="auto",                      # Add this line
                textfont={"size": 12}
            ),
            row=1, col=2
        )
    else:
        fig.add_trace(
            go.Bar(
                x=[],
                y=[],
                name="No Data"
            ),
            row=1, col=2
        )

    # Categorical vs target bar chart (Cramér's V)
    if cat_corr is not None and not cat_corr.empty:
        fig.add_trace(
            go.Bar(
                x = cat_corr.index,
                y = cat_corr["CramersV"].values,
                marker_color = "orange",
                text = cat_corr["CramersV"].round(2).astype(str),  # Add this line
                textposition = "auto",                             # Add this line
                textfont = {"size": 12}                            # Add this line (optional)
            ),
            row=1, col=3
        )
    else:
        fig.add_trace(
            go.Bar(
                x = [],
                y = [],
                name="No Categorical Data"
            ),
            row=1, col=3
        )

    fig.update_layout(
        height = 500,
        width = 1800,
        title_text = f"Correlation Overview with Target: {target_col}",
        showlegend = False
    )

    fig.update_yaxes(title_text="Correlation", row=1, col=2)
    fig.update_yaxes(title_text="Cramér's V", row=1, col=3)

    fig.show()

# ====================================== 2. HIGHLIGHTED ================================================== #
