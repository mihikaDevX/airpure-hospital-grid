"""
AirPure Hospital Grid - Sensor Data Simulator
==============================================
Since real hospital IoT sensor data is not available, this module generates
realistic synthetic multi-ward time-series data:
    - CO2 (ppm)
    - PM2.5 (ug/m3)
    - PM10 (ug/m3)
    - Temperature (C)
    - Humidity (%)
    - Airflow velocity (m/s)

Each ward gets a baseline "normal" airflow regime plus randomly injected
anomaly events (stagnation, contamination spikes, HVAC faults) so the
downstream autoencoder has something realistic to learn from and detect.

In production, this module is replaced by the MQTT ingestion layer reading
from real ESP32/RaspberryPi sensor nodes (see README section "Swapping in
real hardware").
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

WARDS = ["ICU", "General_Ward_A", "General_Ward_B", "Isolation_Ward", "Nursing_Station"]

FEATURES = ["co2_ppm", "pm25", "pm10", "temp_c", "humidity_pct", "airflow_ms"]

# Baseline "healthy" operating ranges per feature (mean, std)
BASELINE = {
    "co2_ppm": (550, 40),
    "pm25": (12, 3),
    "pm10": (20, 4),
    "temp_c": (22, 0.8),
    "humidity_pct": (45, 3),
    "airflow_ms": (0.35, 0.05),
}

ANOMALY_TYPES = ["stagnation", "contamination_spike", "hvac_fault"]


def _apply_anomaly(row, kind):
    """Perturb a single reading to simulate a specific failure mode."""
    row = row.copy()
    if kind == "stagnation":
        row["airflow_ms"] *= RNG.uniform(0.1, 0.3)
        row["co2_ppm"] += RNG.uniform(300, 600)
    elif kind == "contamination_spike":
        row["pm25"] *= RNG.uniform(3, 6)
        row["pm10"] *= RNG.uniform(2.5, 5)
    elif kind == "hvac_fault":
        row["airflow_ms"] *= RNG.uniform(0.0, 0.15)
        row["temp_c"] += RNG.uniform(2, 5)
        row["humidity_pct"] += RNG.uniform(10, 20)
    return row


def simulate_ward(ward_name, n_steps=2000, anomaly_rate=0.03, interval_sec=30):
    """Simulate one ward's sensor stream with slow drift + occasional anomalies."""
    t0 = pd.Timestamp("2026-01-01")
    timestamps = [t0 + pd.Timedelta(seconds=i * interval_sec) for i in range(n_steps)]

    # slow sinusoidal drift (e.g. day/night occupancy cycles) + noise
    drift = np.sin(np.linspace(0, 6 * np.pi, n_steps))

    rows = []
    labels = []
    for i in range(n_steps):
        row = {}
        for feat, (mean, std) in BASELINE.items():
            drift_scale = 0.15 * std if feat in ("co2_ppm", "airflow_ms") else 0.05 * std
            row[feat] = mean + drift[i] * drift_scale + RNG.normal(0, std)

        is_anomaly = RNG.random() < anomaly_rate
        anomaly_type = "none"
        if is_anomaly:
            anomaly_type = RNG.choice(ANOMALY_TYPES)
            row = _apply_anomaly(row, anomaly_type)

        row["ward"] = ward_name
        row["timestamp"] = timestamps[i]
        rows.append(row)
        labels.append(anomaly_type)

    df = pd.DataFrame(rows)
    df["anomaly_type"] = labels
    df["is_anomaly"] = df["anomaly_type"] != "none"
    return df


def simulate_hospital(n_steps=2000, anomaly_rate=0.03, seed=42):
    global RNG
    RNG = np.random.default_rng(seed)
    frames = [simulate_ward(w, n_steps=n_steps, anomaly_rate=anomaly_rate) for w in WARDS]
    full = pd.concat(frames, ignore_index=True)
    return full[["timestamp", "ward"] + FEATURES + ["anomaly_type", "is_anomaly"]]


if __name__ == "__main__":
    df = simulate_hospital(n_steps=2000, anomaly_rate=0.03)
    out_path = "data/simulated_sensor_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Simulated {len(df)} readings across {df['ward'].nunique()} wards -> {out_path}")
    print(df["anomaly_type"].value_counts())
