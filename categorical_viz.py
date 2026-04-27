import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List

def categorical_countplot(df: pd.DataFrame(), columns: Optional[List[str]] = None, target_col: Optional[str] = None, n_cols:int =3) -> None:
    
    if columns is None:
        print("[!] Columns is empty, automatically select categorical based on df")
        columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if not columns:
        print("[!] No categorical columns found in the DataFrame.")
        return
    
    if target_col and target_col not in df.columns:
        print(f"[!] Target column '{target_col}' not found in the DataFrame. Ignoring target_col.")
        target_col = None
        
    n_rows = int(np.ceil(len(columns) / n_cols))

    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

    # Flatten axes for easy iteration
    axes = axes.flatten()
    for i, col in enumerate(columns):
        ax = axes[i]
        
        sns.countplot(data=df, x=col, ax=ax, hue=target_col)
        ax.grid(color='r', linestyle='-', linewidth=2)
        ax.set_axisbelow(True) 
        ax.set_title(col)
        ax.tick_params(axis='x', rotation=25)

    # Remove unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()
