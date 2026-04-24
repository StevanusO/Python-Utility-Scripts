import pandas as pd
from scipy import stats
from typing import Optional, List

def numerical_basic(df: pd.DataFrame(), columns: Optional[List[str]] = None) -> pd.DataFrame:
    if columns is None:
        print("[!] Columns is empty, automatically select numerical based on df")
        columns = df.select_dtypes(include=["number"]).columns.tolist()
    if not columns:
        return pd.DataFrame()

    total_rows = len(df)
    df_num = df[columns].apply(pd.to_numeric, errors="coerce") # Enforces Numerical
    num_describe = df_num.describe(percentiles=[0.25, 0.5, 0.75]) # Get Q1, Q2, and Q3
    
    def _skew(series: pd.Series) -> float:
        arr = series.dropna().values
        return float(stats.skew(arr, bias=False)) if arr.size > 0 else np.nan

    def _kurt(series: pd.Series) -> float:
        arr = series.dropna().values
        return float(stats.kurtosis(arr, bias=False, fisher=True)) if arr.size > 0 else np.nan

    skews = df_num[columns].apply(_skew)
    kurts = df_num[columns].apply(_kurt)

    rows = []
    for col in columns:
        desc = num_describe[col]
        count_col = int(desc.get("count", 0))
        missing_count = total_rows - count_col
        missing_pct = (missing_count / total_rows * 100) if total_rows > 0 else 0.0
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

def print_numerical_basic(df: pd.DataFrame, columns: Optional[List[str]] = None) -> None:
    """Print compact textual report for numeric columns."""
    summary = numerical_basic(df, columns=columns)
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
    
