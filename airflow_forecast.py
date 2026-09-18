"""
AirPure Hospital Grid - Airflow & Pollutant Forecasting
=========================================================
Predicts each ward's next-interval CO2 and airflow velocity from a sliding
window of recent readings, so the optimizer can act *before* a threshold is
breached rather than reacting after the fact.

Implementation note
--------------------
Uses scikit-learn's MLPRegressor over windowed (lag) features, which is a
lightweight stand-in for a TensorFlow LSTM (no extra install required).
The windowing approach (`build_windows`) is the same one used to prepare
sequences for a Keras `LSTM` layer -- see `build_keras_lstm()` below for the
production swap-in once TensorFlow is available.
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

from simulate_data import simulate_hospital

TARGETS = ["co2_ppm", "airflow_ms"]
WINDOW = 6  # look back 6 readings (~3 min at 30s interval)


def build_windows(series_df, window=WINDOW, targets=TARGETS):
    """Turn a single ward's time-ordered readings into (X, y) lag windows."""
    feats = series_df[["co2_ppm", "pm25", "pm10", "temp_c", "humidity_pct", "airflow_ms"]].values
    X, y = [], []
    for i in range(window, len(feats)):
        X.append(feats[i - window:i].flatten())
        y.append(series_df[targets].values[i])
    return np.array(X), np.array(y)


def train_forecaster(df):
    """Train one shared forecaster across all wards (ward-agnostic model)."""
    X_parts, y_parts = [], []
    for ward, group in df.groupby("ward"):
        group = group.sort_values("timestamp")
        X, y = build_windows(group)
        X_parts.append(X)
        y_parts.append(y)
    X_all = np.vstack(X_parts)
    y_all = np.vstack(y_parts)

    split = int(0.8 * len(X_all))
    X_train, X_test = X_all[:split], X_all[split:]
    y_train, y_test = y_all[:split], y_all[split:]

    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train)
    X_train_s, X_test_s = x_scaler.transform(X_train), x_scaler.transform(X_test)
    y_train_s = y_scaler.transform(y_train)

    model = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=800, random_state=0,
                          early_stopping=True, n_iter_no_change=15)
    print("Training airflow/CO2 forecaster...")
    model.fit(X_train_s, y_train_s)

    preds = y_scaler.inverse_transform(model.predict(X_test_s))
    mae_co2 = mean_absolute_error(y_test[:, 0], preds[:, 0])
    mae_flow = mean_absolute_error(y_test[:, 1], preds[:, 1])
    print(f"  Test MAE -> CO2: {mae_co2:.2f} ppm | Airflow: {mae_flow:.4f} m/s")
    return dict(model=model, x_scaler=x_scaler, y_scaler=y_scaler), dict(mae_co2=mae_co2, mae_flow=mae_flow)


def forecast_next(bundle, recent_window_df):
    """recent_window_df: last WINDOW rows (6 x 6 features) for one ward, time-ordered."""
    feats = recent_window_df[["co2_ppm", "pm25", "pm10", "temp_c", "humidity_pct", "airflow_ms"]].values
    x = feats.flatten().reshape(1, -1)
    x_s = bundle["x_scaler"].transform(x)
    pred_s = bundle["model"].predict(x_s)
    pred = bundle["y_scaler"].inverse_transform(pred_s)[0]
    return dict(zip(TARGETS, pred))


def build_keras_lstm(window=WINDOW, n_features=6):
    """
    Production version (requires `pip install tensorflow`).
    Same windowing scheme as build_windows(), reshaped to
    (samples, window, n_features) for a Keras LSTM.
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        keras.Input(shape=(window, n_features)),
        layers.LSTM(32, return_sequences=False),
        layers.Dense(16, activation="relu"),
        layers.Dense(len(TARGETS)),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


if __name__ == "__main__":
    df = simulate_hospital(n_steps=2000, anomaly_rate=0.03)
    bundle, metrics = train_forecaster(df)

    # Demo: forecast next reading for the ICU from its most recent window
    icu = df[df["ward"] == "ICU"].sort_values("timestamp")
    last_window = icu.iloc[-WINDOW - 1:-1]
    forecast = forecast_next(bundle, last_window)
    actual = icu.iloc[-1][TARGETS].to_dict()
    print(f"\nICU next-step forecast: {forecast}")
    print(f"ICU actual next reading: {actual}")
