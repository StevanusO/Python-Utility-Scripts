"""EDA helpers.

Quick numeric/categorical summaries, skew, outliers, simple plots.
Importable helpers. Light deps: pandas, numpy.
"""
from __future__ import annotations

from typing import List, Optional, Dict, Any

import io
import numpy as np
import pandas as pd

# optional scipy for robust skew/kurtosis estimators; fall back to pandas if missing
try:
    from scipy import stats as _scipy_stats
    _HAVE_SCIPY = True
except Exception:
    _scipy_stats = None
    _HAVE_SCIPY = False


def describe_numerical(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Return numeric summary per column.

    Includes count, missing, unique, zeros, mean/median/std, skew/kurtosis, IQR/outliers.
    """
    if columns is None:
        columns = df.select_dtypes(include="number").columns.tolist()

    if not columns:
        return pd.DataFrame()

    total_rows = len(df)
    # coerce requested columns to numeric (non-convertible -> NaN) to avoid
    # accidental type errors when users pass non-numeric column names.
    df_num = df[columns].apply(pd.to_numeric, errors="coerce")
    # describe with quartiles available
    num_describe = df_num.describe(percentiles=[0.25, 0.5, 0.75])

    # compute skew/kurtosis using scipy when available (prefer unbiased), otherwise pandas
    if _HAVE_SCIPY:
        def _safe_skew(series: pd.Series) -> float:
            arr = series.dropna().values
            return float(_scipy_stats.skew(arr, bias=False)) if arr.size > 0 else np.nan

        def _safe_kurt(series: pd.Series) -> float:
            arr = series.dropna().values
            return float(_scipy_stats.kurtosis(arr, bias=False, fisher=True)) if arr.size > 0 else np.nan

        skews = df[columns].apply(_safe_skew)
        kurts = df[columns].apply(_safe_kurt)
    else:
        skews = df[columns].skew()
        kurts = df[columns].kurt()

    rows = []
    for col in columns:
        # safe access to describe values
        desc = num_describe[col]
        count_col = int(desc.get("count", 0))
        missing_count = total_rows - count_col
        missing_pct = (missing_count / total_rows * 100) if total_rows > 0 else 0.0
        # use the coerced numeric series for numeric metrics
        series = df_num[col]
        unique_val = int(series.nunique(dropna=True))
        zero_count = int((series == 0).sum())

        q1 = desc.get("25%", np.nan)
        q2 = desc.get("50%", np.nan)
        q3 = desc.get("75%", np.nan)
        mean_col = desc.get("mean", np.nan)
        std_col = desc.get("std", np.nan)

        # iqr / outlier bounds
        try:
            iqr = q3 - q1
        except Exception:
            iqr = np.nan
        lower_bound = (q1 - 1.5 * iqr) if pd.notna(iqr) else np.nan
        upper_bound = (q3 + 1.5 * iqr) if pd.notna(iqr) else np.nan

        outlier_count = 0
        if pd.notna(lower_bound) and pd.notna(upper_bound):
            outlier_mask = (series < lower_bound) | (series > upper_bound)
            outlier_count = int(outlier_mask.sum())
        outlier_pct = (outlier_count / total_rows * 100) if total_rows > 0 else 0.0

        skew_raw = skews.get(col, np.nan)
        skew_val = float(skew_raw) if pd.notna(skew_raw) else np.nan
        kurt_raw = kurts.get(col, np.nan)
        kurt_val = float(kurt_raw) if pd.notna(kurt_raw) else np.nan

        # skew interpretation
        direction = "Right (+)" if skew_val > 0.1 else "Left (-)" if skew_val < -0.1 else "Balanced (~)"
        if pd.isna(skew_val):
            severity = "N/A"
        else:
            severity = (
                "Fairly Symmetrical"
                if abs(skew_val) < 0.5
                else "Moderately Skewed"
                if abs(skew_val) <= 1
                else "Highly Skewed"
            )

        kurt_desc = (
            "Heavy Tails" if kurt_val > 1 else "Light Tails" if kurt_val < -1 else "Normal-ish"
        )

        rows.append(
            {
                "column": col,
                "count": count_col,
                "missing_count": missing_count,
                "missing_pct": round(float(missing_pct), 2),
                "unique": unique_val,
                "zeros": zero_count,
                "mean": round(float(mean_col), 4) if pd.notna(mean_col) else None,
                "median": round(float(q2), 4) if pd.notna(q2) else None,
                "std": round(float(std_col), 4) if pd.notna(std_col) else None,
                "skew": round(float(skew_val), 4) if not pd.isna(skew_val) else None,
                "skew_direction": direction,
                "skew_severity": severity,
                "kurtosis": round(float(kurt_val), 4) if not pd.isna(kurt_val) else None,
                "kurtosis_desc": kurt_desc,
                "iqr": round(float(iqr), 4) if pd.notna(iqr) else None,
                "lower_bound": float(lower_bound) if pd.notna(lower_bound) else None,
                "upper_bound": float(upper_bound) if pd.notna(upper_bound) else None,
                "outlier_count": outlier_count,
                "outlier_pct": round(float(outlier_pct), 2),
            }
        )

    summary = pd.DataFrame(rows).set_index("column")
    return summary


def print_numerical_report(df: pd.DataFrame, columns: Optional[List[str]] = None) -> None:
    """Print compact textual report for numeric columns."""
    summary = describe_numerical(df, columns=columns)
    if summary.empty:
        print("No numeric columns found.")
        return

    total_rows = len(df)
    for col in summary.index:
        row = summary.loc[col]
        print(f"\nCOLUMN: {col}")
        print(f"{'='*12} INTEGRITY & VARIETY {'='*12}")
        print(
            f"Missing: {int(row['missing_count'])} ({row['missing_pct']:.2f}%) | Unique Vals: {int(row['unique'])} | Zeros: {int(row['zeros'])}"
        )
        print(f"{'='*12} DISTRIBUTION {'='*16}")
        print(f"Mean: {row['mean']} | Median: {row['median']} | Std Dev: {row['std']}")
        skew = row['skew'] if row['skew'] is not None else float('nan')
        print(f"Skew: {skew} ({row['skew_severity']} | {row['skew_direction']})")
        print(f"Kurtosis: {row['kurtosis']} ({row['kurtosis_desc']})")
        print(f"{'='*12} OUTLIER CHECK {'='*15}")
        print(f"IQR: {row['iqr']} | Bounds: [{row['lower_bound']}, {row['upper_bound']}]")
        print(f"Detected Outliers: {int(row['outlier_count'])} ({row['outlier_pct']:.2f}%)")
        print("-" * 60)


def analyze_categorical(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Return frequency/top/missing for categorical columns."""
    if columns is None:
        columns = df.select_dtypes(include=["category", "object", "bool"]).columns.tolist()

    rows = []
    for col in columns:
        vc = df[col].value_counts(dropna=False)
        top = vc.index[0] if not vc.empty else None
        top_count = int(vc.iloc[0]) if not vc.empty else 0
        rows.append(
            {
                "column": col,
                "unique": int(df[col].nunique(dropna=True)),
                "top": top,
                "top_count": top_count,
                "missing_count": int(df[col].isna().sum()),
            }
        )

    return pd.DataFrame(rows).set_index("column")


def dataset_overview(df: pd.DataFrame, sample_n: int = 5) -> Dict[str, Any]:
    """Return dict: shape, cols, dtypes, sample, info, numeric cols, cat cols."""
    total = len(df)
    sample_n = max(0, min(sample_n, total))
    sample = df.sample(n=sample_n, random_state=0) if sample_n > 0 else df.head(0)

    buf = io.StringIO()
    df.info(buf=buf)
    info_str = buf.getvalue()

    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "sample": sample,
        "info": info_str,
        "num_columns": df.select_dtypes(include="number").columns.tolist(),
        "cat_columns": df.select_dtypes(include=["category", "object", "bool"]).columns.tolist(),
    }


def missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing_count and missing_pct per column."""
    total = len(df)
    missing = df.isnull().sum()
    missing_df = pd.DataFrame(
        {
            "missing_count": missing,
            "missing_pct": ((missing / total) * 100).round(2) if total > 0 else missing * 0,
        }
    )
    return missing_df.sort_values("missing_count", ascending=False)


def suggest_transform(series: pd.Series) -> Dict[str, Any]:
    """Suggest transform from skew. Returns dict: skew, recommendation, reason."""
    s = series.dropna()
    if s.empty:
        return {"skew": None, "recommendation": "N/A", "reason": "no non-missing values"}

    if _HAVE_SCIPY:
        try:
            skew_val = float(_scipy_stats.skew(s.values, bias=False))
        except Exception:
            skew_val = float(s.skew())
    else:
        skew_val = float(s.skew())

    if pd.isna(skew_val):
        return {"skew": None, "recommendation": "N/A", "reason": "insufficient data to compute skew"}

    abs_skew = abs(skew_val)
    if abs_skew < 0.5:
        return {"skew": round(skew_val, 4), "recommendation": "none", "reason": "approximately symmetric"}

    if skew_val > 0:
        if s.min() > 0:
            return {
                "skew": round(skew_val, 4),
                "recommendation": "box-cox / log1p",
                "reason": "positive skew and strictly positive values — Box‑Cox or log1p may help",
            }
        else:
            return {
                "skew": round(skew_val, 4),
                "recommendation": "yeo-johnson",
                "reason": "positive skew with non-positive values — Yeo‑Johnson handles zero/negative values",
            }
    # negative skew
    return {
        "skew": round(skew_val, 4),
        "recommendation": "yeo-johnson (or reflect+Box‑Cox)",
        "reason": "negative skew — consider reflection then power transform or Yeo‑Johnson",
    }


def safe_apply_transform(series: pd.Series, method: str = "log1p", standardize: bool = False) -> pd.Series:
    """Apply transform safely. Preserve NaNs.

    Methods: log1p, box-cox, yeo-johnson (uses sklearn PowerTransformer).
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)

    res = series.copy().astype(float)
    mask = series.notna()
    if not mask.any():
        return res

    values = series.loc[mask]

    if method == "log1p":
        # log1p handles zeros; values <= -1 will produce -inf/NaN, so the caller
        # should be aware of domain issues for very negative values.
        try:
            transformed = np.log1p(values.astype(float))
        except Exception:
            transformed = values.apply(lambda x: np.log1p(x) if pd.notna(x) and (1 + x) > 0 else np.nan)
        res.loc[mask] = transformed
        return res

    if method in {"box-cox", "yeo-johnson"}:
        try:
            from sklearn.preprocessing import PowerTransformer
        except Exception:
            raise RuntimeError("scikit-learn is required for Box-Cox / Yeo-Johnson transforms")

        if method == "box-cox" and (values <= 0).any():
            raise ValueError("Box-Cox requires strictly positive values (no zeros or negatives).")

        pt = PowerTransformer(method="box-cox" if method == "box-cox" else "yeo-johnson", standardize=standardize)
        arr = values.values.reshape(-1, 1)
        transformed = pt.fit_transform(arr).ravel()
        res.loc[mask] = transformed
        return res

    raise ValueError(f"Unknown transform method: {method}")


def numeric_target_analysis(df: pd.DataFrame, target: str, numeric_cols: Optional[List[str]] = None) -> Dict[str, Any]:
    """Compute Pearson and Spearman correlation vs numeric target."""
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target and c in df.columns]

    results: Dict[str, Any] = {}
    if not numeric_cols:
        return results

    # Pearson
    try:
        pearson = df[numeric_cols + [target]].corr(method="pearson").get(target).drop(labels=[target], errors="ignore")
        results["pearson"] = pearson.sort_values(key=lambda s: s.abs(), ascending=False)
    except Exception:
        results["pearson"] = None

    # Spearman
    try:
        spearman = df[numeric_cols + [target]].corr(method="spearman").get(target).drop(labels=[target], errors="ignore")
        results["spearman"] = spearman.sort_values(key=lambda s: s.abs(), ascending=False)
    except Exception:
        results["spearman"] = None

    return results


def categorical_target_analysis(df: pd.DataFrame, target: str, cat_cols: Optional[List[str]] = None, numeric_cols: Optional[List[str]] = None) -> Dict[str, Any]:
    """Target categorical analysis: distribution, numeric stats by target, crosstabs."""
    if cat_cols is None:
        cat_cols = df.select_dtypes(include=["category", "object", "bool"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c != target and c in df.columns]

    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target and c in df.columns]

    out: Dict[str, Any] = {}
    out["target_distribution"] = df[target].value_counts(dropna=False)

    if numeric_cols:
        try:
            out["numeric_stats_by_target"] = df.groupby(target)[numeric_cols].agg(["mean", "std", "count"])
        except Exception:
            out["numeric_stats_by_target"] = None

    if cat_cols:
        crosstabs = {}
        for col in cat_cols:
            try:
                crosstabs[col] = pd.crosstab(df[col], df[target], dropna=False)
            except Exception:
                crosstabs[col] = None
        out["crosstabs"] = crosstabs

    return out


def run_eda(df: pd.DataFrame, target: Optional[str] = None, sample_n: int = 5, show: bool = False) -> Dict[str, Any]:
    """Run high-level EDA and return results dict.

    Keys: overview, missing, numeric_summary, categorical_summary, target_analysis.
    """
    overview = dataset_overview(df, sample_n=sample_n)
    missing = missing_value_summary(df)

    numeric_cols = [c for c in overview["num_columns"] if c != target]
    cat_cols = [c for c in overview["cat_columns"] if c != target]

    numeric_summary = describe_numerical(df, columns=numeric_cols) if numeric_cols else pd.DataFrame()
    categorical_summary = analyze_categorical(df, columns=cat_cols) if cat_cols else pd.DataFrame()

    target_analysis: Dict[str, Any] = {}
    if target is not None:
        if target not in df.columns:
            target_analysis["error"] = f"Target column '{target}' not found in DataFrame"
        else:
            if pd.api.types.is_numeric_dtype(df[target]):
                # numeric target: use numeric_target_analysis
                target_analysis = numeric_target_analysis(df, target, numeric_cols=numeric_cols)
                # also include group statistics for categorical features when present
                if cat_cols:
                    group_stats = {}
                    for col in cat_cols:
                        try:
                            group_stats[col] = df.groupby(col)[numeric_cols].agg(["mean", "std", "count"])
                        except Exception:
                            group_stats[col] = None
                    target_analysis["group_stats_by_cat_feature"] = group_stats
            else:
                # categorical target: use categorical_target_analysis
                target_analysis = categorical_target_analysis(df, target, cat_cols=cat_cols, numeric_cols=numeric_cols)

    result = {
        "overview": overview,
        "missing": missing,
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
        "target_analysis": target_analysis,
    }

    if show:
        print("=== Dataset overview ===")
        print(f"Shape: {overview['shape']}")
        print(f"Numeric columns: {overview['num_columns']}")
        print(f"Categorical columns: {overview['cat_columns']}")
        print("\n=== Missing values (top) ===")
        print(missing.head(10))
        print("\n=== Numerical summary ===")
        print(numeric_summary)
        print("\n=== Categorical summary ===")
        print(categorical_summary)
        if target_analysis:
            print("\n=== Target analysis ===")
            for k, v in target_analysis.items():
                print(f"--- {k} ---")
                print(v)

    return result


def plot_hist_kde(df: pd.DataFrame, column: str, bins: int = 30, figsize: tuple = (8, 4), show: bool = True):
    """Plot histogram + KDE for numeric column."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception:
        raise RuntimeError("matplotlib and seaborn are required for plotting")

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise TypeError(f"Column '{column}' must be numeric for histogram/KDE")

    plt.figure(figsize=figsize)
    sns.histplot(df[column].dropna(), bins=bins, kde=True)
    plt.title(f"Histogram + KDE: {column}")
    if show:
        plt.show()


def plot_box(df: pd.DataFrame, column: str, by: Optional[str] = None, figsize: tuple = (8, 4), show: bool = True):
    """Boxplot for column or grouped by `by`."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception:
        raise RuntimeError("matplotlib and seaborn are required for plotting")

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    plt.figure(figsize=figsize)
    if by:
        if by not in df.columns:
            raise ValueError(f"Grouping column '{by}' not found in DataFrame")
        sns.boxplot(x=by, y=column, data=df)
        plt.title(f"Boxplot of {column} by {by}")
    else:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise TypeError(f"Column '{column}' must be numeric for a boxplot")
        sns.boxplot(x=df[column].dropna())
        plt.title(f"Boxplot: {column}")
    if show:
        plt.show()


def plot_scatter_vs_target(df: pd.DataFrame, feature: str, target: str, figsize: tuple = (6, 6), show: bool = True):
    """Scatter + regression: feature vs numeric target."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception:
        raise RuntimeError("matplotlib and seaborn are required for plotting")

    if feature not in df.columns or target not in df.columns:
        raise ValueError("feature and target must be column names in the DataFrame")
    if not pd.api.types.is_numeric_dtype(df[feature]) or not pd.api.types.is_numeric_dtype(df[target]):
        raise TypeError("Both feature and target must be numeric for scatter/regression plot")

    plt.figure(figsize=figsize)
    sns.regplot(x=feature, y=target, data=df, scatter_kws={"s": 20}, line_kws={"color": "red"})
    plt.title(f"{feature} vs {target}")
    if show:
        plt.show()


def plot_bar_counts(df: pd.DataFrame, column: str, top_n: int = 20, figsize: tuple = (8, 4), show: bool = True):
    """Bar chart of value counts (top_n)."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        raise RuntimeError("matplotlib is required for plotting")

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    vc = df[column].value_counts(dropna=False).head(top_n)
    plt.figure(figsize=figsize)
    vc.plot(kind="bar")
    plt.title(f"Value counts: {column}")
    plt.tight_layout()
    if show:
        plt.show()


def init_viz(style: str = "seaborn", figsize: tuple = (8, 6)) -> None:
    """Init plotting defaults for seaborn/matplotlib/plotly."""
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns
    except Exception:
        sns = None

    if style == "seaborn" and sns is not None:
        sns.set_theme()
    elif style == "matplotlib":
        try:
            plt.style.use("classic")
        except Exception:
            pass

    if style == "plotly":
        try:
            import plotly.io as pio

            pio.templates.default = "plotly_white"
        except Exception:
            pass

    plt.rcParams["figure.figsize"] = figsize


if __name__ == "__main__":
    # quick runnable example
    sample = pd.DataFrame(
        {
            "small_ints": [1, 2, 3, 4, 100],
            "floats": [1.0, np.nan, 2.5, 3.5, 4.0],
            "zeros": [0, 0, 0, 1, 2],
        }
    )
    print("Running demo with sample DataFrame")
    print(sample)
    print('\nSummary table:')
    print(describe_numerical(sample))
    print('\nPretty report:')
    print_numerical_report(sample)
