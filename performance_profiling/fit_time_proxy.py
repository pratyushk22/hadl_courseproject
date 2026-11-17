import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import nnls
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error, r2_score

INPUT_CSV = "profiling_results.csv"
PROXY_JSON = "time_proxy_model.json"
PLOT_PATH = "proxy_fit_analysis.png"

def build_design_matrix(B, L):
    """
    Constructs physics-informed Transformer FLOP features including the intercept column:
    X = [1, B, B * L, B * (L^2)]
    Using scipy.optimize.nnls guarantees ALL coefficients (including intercept beta_0) >= 0.
    """
    B = np.asarray(B, dtype=np.float64)
    L = np.asarray(L, dtype=np.float64)
    ones = np.ones_like(B)
    return np.column_stack([ones, B, B * L, B * (L ** 2)])

def filter_swap_outliers(df):
    """
    Removes OS unified-memory disk-swap outliers where latency per token
    exceeds 2.5x the median per-token latency.
    """
    ms_per_token = df["step_time_ms"] / df["total_tokens"]
    median_rate = ms_per_token.median()
    clean_df = df[ms_per_token <= 2.5 * median_rate].copy()
    removed = len(df) - len(clean_df)
    if removed > 0:
        print(f"Filtered out {removed} OS memory-swap outlier row(s) (where RAM spilled to SSD).")
    return clean_df

def main():
    raw_df = pd.read_csv(INPUT_CSV)
    df = filter_swap_outliers(raw_df)

    B = df["batch_size"].values
    L = df["seq_len"].values
    y_actual = df["step_time_ms"].values

    X = build_design_matrix(B, L)

    # 1. Fit strictly non-negative model (beta_0, beta_B, beta_BL, beta_BL2 >= 0)
    coeffs, _ = nnls(X, y_actual)
    y_pred_in_sample = X @ coeffs

    # 2. Leave-One-Out Cross-Validation (unbiased Proxy Method Error metric)
    loo = LeaveOneOut()
    y_pred_loo = np.zeros_like(y_actual)
    for train_idx, test_idx in loo.split(X):
        c_cv, _ = nnls(X[train_idx], y_actual[train_idx])
        y_pred_loo[test_idx] = X[test_idx] @ c_cv

    # 3. Compute Error Metrics for Report
    in_sample_mape = mean_absolute_percentage_error(y_actual, y_pred_in_sample) * 100
    loo_mape = mean_absolute_percentage_error(y_actual, y_pred_loo) * 100
    loo_rmse = root_mean_squared_error(y_actual, y_pred_loo)
    r2 = r2_score(y_actual, y_pred_in_sample)

    beta_0, beta_B, beta_BL, beta_BL2 = [float(c) for c in coeffs]

    print("\n=== Fitted Analytical Time Proxy Formula ===")
    print(f"T_step(B, L) [ms] = {beta_0:.4f} + {beta_B:.6f}*B + {beta_BL:.6f}*(B*L) + {beta_BL2:.9f}*(B*L^2)")
    print("\n=== Proxy Method Error Metrics ===")
    print(f"In-Sample MAPE      : {in_sample_mape:.2f}%")
    print(f"LOOCV MAPE (Unseen) : {loo_mape:.2f}%")
    print(f"LOOCV RMSE          : {loo_rmse:.2f} ms")
    print(f"R^2 Goodness of Fit : {r2:.5f}")

    # 4. Save fitted coefficients & metrics to time_proxy_model.json
    proxy_data = {
        "gpu_name": str(df["gpu_name"].iloc[0]),
        "formula": "T_step_ms = beta_0 + beta_B * B + beta_BL * (B * L) + beta_BL2 * (B * L^2)",
        "coefficients": {
            "beta_0": beta_0,
            "beta_B": beta_B,
            "beta_BL": beta_BL,
            "beta_BL2": beta_BL2
        },
        "metrics": {
            "in_sample_mape_pct": round(in_sample_mape, 3),
            "loocv_mape_pct": round(loo_mape, 3),
            "loocv_rmse_ms": round(loo_rmse, 3),
            "r2_score": round(r2, 5)
        }
    }
    with open(PROXY_JSON, "w") as f:
        json.dump(proxy_data, f, indent=4)
    print(f"\nSaved proxy model parameters to '{PROXY_JSON}'")

    # 5. Generate Diagnostic Plots
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for b_val in sorted(df["batch_size"].unique()):
        sub = df[df["batch_size"] == b_val].sort_values("seq_len")
        axes[0].scatter(sub["seq_len"], sub["step_time_ms"], label=f"Actual (B={b_val})")
        l_grid = np.linspace(sub["seq_len"].min(), sub["seq_len"].max(), 100)
        b_grid = np.full_like(l_grid, b_val)
        pred_curve = build_design_matrix(b_grid, l_grid) @ coeffs
        axes[0].plot(l_grid, pred_curve, linestyle="--")

    axes[0].set_title("Step Latency vs. Token Length (Actual vs. Proxy Curve)")
    axes[0].set_xlabel("Sequence Length (L)")
    axes[0].set_ylabel("Training Step Time (ms)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(y_actual, y_pred_loo, color="royalblue", edgecolors="k", alpha=0.8)
    max_val = max(y_actual.max(), y_pred_loo.max())
    axes[1].plot([0, max_val], [0, max_val], "r--", label="Ideal (y = x)")
    axes[1].set_title(f"Proxy Prediction Parity (LOOCV MAPE = {loo_mape:.2f}%, R² = {r2:.4f})")
    axes[1].set_xlabel("Actual Hardware Step Time (ms)")
    axes[1].set_ylabel("Proxy Predicted Step Time (ms)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=300)
    print(f"Saved report diagnostic plots to '{PLOT_PATH}'")

if __name__ == "__main__":
    main()