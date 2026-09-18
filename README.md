# AirPure Hospital Grid
### AI-Based Intelligent Ventilation Optimisation for Infection Control
**Project ID:** HEALTH-05 | **SDGs:** 3 (Good Health & Well-being), 6 (Clean Water & Sanitation → Air Quality)

Poor ward ventilation increases airborne infection and cross-contamination risk.
This system continuously senses ward air quality, detects abnormal airflow
patterns with an autoencoder, forecasts near-future conditions, and
recommends HVAC adjustments — validated first in a digital twin before
touching real hospital HVAC hardware.

## Architecture

```
IoT Sensors (CO2, PM2.5/10, temp, humidity, airflow)
        │  MQTT
        ▼
Time-series pipeline ──► Autoencoder ──► anomaly flag (stagnation / contamination / HVAC fault)
        │                                        │
        └──────────────► Forecaster (next-step CO2 & airflow)
                                                  │
                                                  ▼
                                            Optimizer (rule engine)
                                                  │
                                                  ▼
                                   HVAC control recommendation + alert
                                                  │
                                                  ▼
                                Digital Twin (CFD-calibrated) + Dashboard
```

## What's in this prototype

| File | Role |
|---|---|
| `simulate_data.py` | Generates realistic multi-ward sensor time series with injected anomalies (stand-in for real IoT/MQTT data, which we don't have access to yet) |
| `autoencoder_anomaly.py` | Trains an autoencoder on normal ventilation patterns; flags high-reconstruction-error readings as anomalies |
| `airflow_forecast.py` | Predicts next-interval CO2 and airflow velocity per ward from a sliding window of recent readings |
| `optimizer.py` | Rule-based controller: turns anomaly flags + forecasts into HVAC actions (fan speed / damper adjustments, maintenance alerts) |
| `main.py` | Runs the full pipeline end-to-end and generates the dashboard + summary report |
| `outputs/` | Generated results: `dashboard_summary.png`, `run_summary.txt`, `full_pipeline_results.csv`, `control_actions.csv` |

## Why this implementation, not raw TensorFlow

TensorFlow isn't installed in the environment this was built in, and no real
sensor hardware or hospital dataset is available yet. So the core algorithms
are implemented with **the same architecture and interface** using NumPy
(autoencoder) and scikit-learn (forecaster) — fully runnable anywhere with no
install step, so you can demo it today. Each module includes a
`build_keras_*()` function showing the exact 1:1 Keras/TensorFlow
equivalent to swap in once you have TensorFlow available and/or real data —
no architecture changes needed, just a different training call.

## How to run

```bash
pip install -r requirements.txt   # numpy, pandas, scikit-learn, matplotlib
python3 main.py
```

This will:
1. Simulate 10,000 sensor readings across 5 wards (ICU, 2 general wards, isolation ward, nursing station)
2. Train the autoencoder and report precision/recall/F1 on detecting injected anomalies
3. Train the forecaster and report MAE for CO2 and airflow predictions
4. Run the optimizer on the latest reading per ward and print recommended actions
5. Save a 4-panel dashboard image and text summary to `outputs/`

## Current results (synthetic data, this run)

- **Anomaly detection:** Precision 0.57 / Recall 1.00 / F1 0.72 (catches every injected anomaly; some false positives — tunable via `threshold_percentile` in `autoencoder_anomaly.py`)
- **Forecasting:** CO2 MAE ≈ 39 ppm, Airflow MAE ≈ 0.05 m/s
- These are baseline numbers from a small NumPy/sklearn model on synthetic data — expect them to improve substantially with a real LSTM/Keras model and actual sensor data.

## Extending to a real deployment

**Real hardware:** Replace `simulate_data.py`'s output with an MQTT subscriber
writing into the same DataFrame schema (`timestamp, ward, co2_ppm, pm25, pm10,
temp_c, humidity_pct, airflow_ms`). ESP32 + MH-Z19 (CO2), PMS5003 (PM2.5/10),
DHT22 (temp/humidity), and a hot-wire anemometer are common low-cost choices.

**Real AI models:** Swap `NumpyAutoencoder` for `build_keras_autoencoder()`
and the sklearn `MLPRegressor` for `build_keras_lstm()` (both already
stubbed in the respective files) once TensorFlow is installed.

**CFD / digital twin:** Model each ward's geometry in OpenFOAM or SimScale;
use real sensor placements as boundary/validation points. The digital twin
is where new optimizer actions get tested in simulation before being sent to
real HVAC controllers — critical for a safety-relevant system like this.

**RL optimizer (future work):** `optimizer.py` includes an `RLPlaceholderNote`
describing how to replace the rule engine with a trained RL agent
(Stable-Baselines3 DQN/PPO) once enough (state, action, outcome) data has
accumulated from the rule-based controller running in production.

## Presenting this project

The `outputs/dashboard_summary.png` plot (ICU CO2 timeline with flagged
anomalies, reconstruction-error distribution, per-ward anomaly rates, and
current optimizer priorities) is designed to be the headline visual for a
review/demo — it shows the whole pipeline's output in one image.
