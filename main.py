"""
AirPure Hospital Grid - End-to-End Pipeline
=============================================
Runs the full loop for a live demo:

  1. Simulate sensor data (stand-in for MQTT ingestion from real IoT nodes)
  2. Train the autoencoder -> flag anomalies
  3. Train the forecaster -> predict next-interval CO2 / airflow
  4. Run the optimizer over the most recent reading per ward
  5. Save a results CSV + a summary report + plots for the presentation

Run with:  python3 main.py
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulate_data import simulate_hospital, WARDS, FEATURES
from autoencoder_anomaly import train_and_detect, evaluate
from airflow_forecast import train_forecaster, forecast_next, WINDOW
from optimizer import decide_action


def run_pipeline(n_steps=2000, anomaly_rate=0.03):
    print("=" * 60)
    print("STEP 1/4  Simulating hospital sensor network")
    print("=" * 60)
    df = simulate_hospital(n_steps=n_steps, anomaly_rate=anomaly_rate)
    print(f"{len(df)} readings across {len(WARDS)} wards\n")

    print("=" * 60)
    print("STEP 2/4  Training autoencoder anomaly detector")
    print("=" * 60)
    scored_df, ae, threshold, scaler = train_and_detect(df)
    metrics = evaluate(scored_df)
    print()

    print("=" * 60)
    print("STEP 3/4  Training airflow/CO2 forecaster")
    print("=" * 60)
    bundle, forecast_metrics = train_forecaster(df)
    print()

    print("=" * 60)
    print("STEP 4/4  Running optimizer for latest reading, per ward")
    print("=" * 60)
    actions = []
    for ward in WARDS:
        ward_df = scored_df[scored_df["ward"] == ward].sort_values("timestamp")
        current = ward_df.iloc[-1]
        window = ward_df.iloc[-WINDOW - 1:-1]
        forecast = forecast_next(bundle, window)
        action = decide_action(
            ward=ward,
            current_reading=current,
            forecast=forecast,
            is_anomaly=bool(current["predicted_anomaly"]),
            anomaly_type=current["anomaly_type"] if current["is_anomaly"] else "predicted",
        )
        actions.append(action)
        print(f"  [{action.priority:>8}] {ward:<18} -> {action.action:<28} | {action.reason}")

    actions_df = pd.DataFrame([a.__dict__ for a in actions])
    actions_df.to_csv("outputs/control_actions.csv", index=False)
    scored_df.to_csv("outputs/full_pipeline_results.csv", index=False)

    make_plots(scored_df, actions_df)

    write_summary(metrics, forecast_metrics, actions_df)

    print("\nDone. Outputs saved in ./outputs/")
    return scored_df, actions_df, metrics, forecast_metrics


def make_plots(scored_df, actions_df):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("AirPure Hospital Grid — Simulation Results", fontsize=14, fontweight="bold")

    # 1. CO2 timeline for ICU with anomalies marked
    icu = scored_df[scored_df["ward"] == "ICU"].sort_values("timestamp").reset_index()
    ax = axes[0, 0]
    ax.plot(icu.index, icu["co2_ppm"], color="steelblue", lw=0.8, label="CO2 (ppm)")
    anom = icu[icu["predicted_anomaly"]]
    ax.scatter(anom.index, anom["co2_ppm"], color="red", s=12, zorder=5, label="Detected anomaly")
    ax.set_title("ICU: CO2 levels with detected anomalies")
    ax.set_xlabel("Reading #")
    ax.set_ylabel("CO2 (ppm)")
    ax.legend(fontsize=8)

    # 2. Reconstruction error distribution, normal vs anomaly
    ax = axes[0, 1]
    normal_err = scored_df.loc[~scored_df["is_anomaly"], "reconstruction_error"]
    anom_err = scored_df.loc[scored_df["is_anomaly"], "reconstruction_error"]
    ax.hist(normal_err, bins=40, alpha=0.6, label="Normal", color="steelblue")
    ax.hist(anom_err, bins=40, alpha=0.6, label="True anomaly", color="red")
    ax.set_title("Autoencoder reconstruction error distribution")
    ax.set_xlabel("Reconstruction MSE")
    ax.legend(fontsize=8)

    # 3. Anomaly rate per ward
    ax = axes[1, 0]
    rate = scored_df.groupby("ward")["predicted_anomaly"].mean().sort_values(ascending=False)
    ax.bar(rate.index, rate.values, color="coral")
    ax.set_title("Detected anomaly rate by ward")
    ax.set_ylabel("Fraction of readings flagged")
    ax.tick_params(axis="x", rotation=30)

    # 4. Current control action priority per ward
    ax = axes[1, 1]
    priority_rank = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
    colors = {"NONE": "seagreen", "LOW": "gold", "MEDIUM": "orange", "HIGH": "orangered", "CRITICAL": "darkred"}
    ranks = actions_df["priority"].map(priority_rank)
    bar_colors = actions_df["priority"].map(colors)
    ax.bar(actions_df["ward"], ranks, color=bar_colors)
    ax.set_yticks(list(priority_rank.values()))
    ax.set_yticklabels(list(priority_rank.keys()))
    ax.set_title("Current optimizer recommendation priority")
    ax.tick_params(axis="x", rotation=30)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("outputs/dashboard_summary.png", dpi=150)
    print("Saved plot -> outputs/dashboard_summary.png")


def write_summary(metrics, forecast_metrics, actions_df):
    lines = []
    lines.append("AirPure Hospital Grid — Pipeline Run Summary")
    lines.append("=" * 50)
    lines.append("")
    lines.append("Anomaly Detection (Autoencoder)")
    lines.append(f"  Precision: {metrics['precision']:.3f}")
    lines.append(f"  Recall:    {metrics['recall']:.3f}")
    lines.append(f"  F1 score:  {metrics['f1']:.3f}")
    lines.append(f"  TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}  TN={metrics['tn']}")
    lines.append("")
    lines.append("Airflow / CO2 Forecasting")
    lines.append(f"  CO2 MAE:     {forecast_metrics['mae_co2']:.2f} ppm")
    lines.append(f"  Airflow MAE: {forecast_metrics['mae_flow']:.4f} m/s")
    lines.append("")
    lines.append("Current Optimizer Recommendations")
    for _, row in actions_df.iterrows():
        lines.append(f"  [{row['priority']:>8}] {row['ward']:<18} -> {row['action']} ({row['reason']})")
    lines.append("")
    with open("outputs/run_summary.txt", "w") as f:
        f.write("\n".join(lines))
    print("Saved summary -> outputs/run_summary.txt")


if __name__ == "__main__":
    run_pipeline()
