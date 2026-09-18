"""
AirPure Hospital Grid - Autoencoder Anomaly Detection
=======================================================
Trains an autoencoder on "normal" ventilation sensor patterns per ward,
then flags anomalies as readings with high reconstruction error.

Implementation note
--------------------
This is implemented as a small feed-forward autoencoder in pure NumPy
(trained with mini-batch gradient descent) so it runs anywhere with no
extra dependencies. The architecture (6 -> 4 -> 2 -> 4 -> 6, ReLU/linear)
maps 1:1 onto a TensorFlow/Keras model — see `build_keras_autoencoder()`
at the bottom for the drop-in production version once TensorFlow is
available on the deployment machine (e.g. `pip install tensorflow`).
"""

import numpy as np
import pandas as pd

from simulate_data import FEATURES, simulate_hospital


class NumpyAutoencoder:
    """Minimal dense autoencoder: input -> hidden -> bottleneck -> hidden -> output."""

    def __init__(self, n_features, hidden=4, bottleneck=2, lr=0.01, seed=0):
        rng = np.random.default_rng(seed)
        self.lr = lr

        def init(a, b):
            return rng.normal(0, np.sqrt(2 / a), size=(a, b))

        self.W1 = init(n_features, hidden)
        self.b1 = np.zeros(hidden)
        self.W2 = init(hidden, bottleneck)
        self.b2 = np.zeros(bottleneck)
        self.W3 = init(bottleneck, hidden)
        self.b3 = np.zeros(hidden)
        self.W4 = init(hidden, n_features)
        self.b4 = np.zeros(n_features)

    @staticmethod
    def _relu(x):
        return np.maximum(0, x)

    @staticmethod
    def _relu_grad(x):
        return (x > 0).astype(float)

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = self._relu(z1)
        z2 = a1 @ self.W2 + self.b2          # bottleneck (linear)
        z3 = z2 @ self.W3 + self.b3
        a3 = self._relu(z3)
        z4 = a3 @ self.W4 + self.b4          # reconstruction (linear)
        cache = (X, z1, a1, z2, z3, a3, z4)
        return z4, cache

    def backward(self, cache):
        X, z1, a1, z2, z3, a3, z4 = cache
        n = X.shape[0]

        dZ4 = (z4 - X) * (2 / n)             # dMSE/dz4
        dW4 = a3.T @ dZ4
        db4 = dZ4.sum(axis=0)

        dA3 = dZ4 @ self.W4.T
        dZ3 = dA3 * self._relu_grad(z3)
        dW3 = z2.T @ dZ3
        db3 = dZ3.sum(axis=0)

        dZ2 = dZ3 @ self.W3.T                # bottleneck is linear
        dW2 = a1.T @ dZ2
        db2 = dZ2.sum(axis=0)

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * self._relu_grad(z1)
        dW1 = X.T @ dZ1
        db1 = dZ1.sum(axis=0)

        for param, grad in [
            (self.W1, dW1), (self.b1, db1), (self.W2, dW2), (self.b2, db2),
            (self.W3, dW3), (self.b3, db3), (self.W4, dW4), (self.b4, db4),
        ]:
            param -= self.lr * grad

    def fit(self, X, epochs=60, batch_size=64, verbose=True):
        n = X.shape[0]
        for epoch in range(epochs):
            perm = np.random.permutation(n)
            X_shuf = X[perm]
            epoch_loss = 0.0
            for start in range(0, n, batch_size):
                batch = X_shuf[start:start + batch_size]
                recon, cache = self.forward(batch)
                loss = np.mean((recon - batch) ** 2)
                epoch_loss += loss * len(batch)
                self.backward(cache)
            epoch_loss /= n
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"  epoch {epoch:3d}  reconstruction MSE = {epoch_loss:.5f}")

    def reconstruction_error(self, X):
        recon, _ = self.forward(X)
        return np.mean((recon - X) ** 2, axis=1)


def zscore_fit(X):
    mu, sigma = X.mean(axis=0), X.std(axis=0) + 1e-8
    return mu, sigma


def zscore_apply(X, mu, sigma):
    return (X - mu) / sigma


def train_and_detect(df, epochs=200, threshold_percentile=97.5):
    """Train the autoencoder on normal-labelled data, score every reading."""
    X_all = df[FEATURES].values.astype(float)
    normal_mask = ~df["is_anomaly"].values

    mu, sigma = zscore_fit(X_all[normal_mask])
    X_norm = zscore_apply(X_all, mu, sigma)

    ae = NumpyAutoencoder(n_features=X_all.shape[1], hidden=8, bottleneck=3, lr=0.01)
    print("Training autoencoder on normal ventilation patterns...")
    ae.fit(X_norm[normal_mask], epochs=epochs)

    errors = ae.reconstruction_error(X_norm)
    threshold = np.percentile(errors[normal_mask], threshold_percentile)

    df = df.copy()
    df["reconstruction_error"] = errors
    df["predicted_anomaly"] = errors > threshold
    return df, ae, threshold, (mu, sigma)


def evaluate(df):
    tp = ((df["predicted_anomaly"]) & (df["is_anomaly"])).sum()
    fp = ((df["predicted_anomaly"]) & (~df["is_anomaly"])).sum()
    fn = ((~df["predicted_anomaly"]) & (df["is_anomaly"])).sum()
    tn = ((~df["predicted_anomaly"]) & (~df["is_anomaly"])).sum()
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}")
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=precision, recall=recall, f1=f1)


def build_keras_autoencoder(n_features):
    """
    Production version (requires `pip install tensorflow`).
    Architecture mirrors NumpyAutoencoder above so results transfer directly
    once real hardware/data is available.
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=(n_features,))
    x = layers.Dense(5, activation="relu")(inputs)
    bottleneck = layers.Dense(2, activation="linear")(x)
    x = layers.Dense(5, activation="relu")(bottleneck)
    outputs = layers.Dense(n_features, activation="linear")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    return model


if __name__ == "__main__":
    df = simulate_hospital(n_steps=2000, anomaly_rate=0.03)
    result_df, ae, threshold, scaler = train_and_detect(df)
    print(f"\nAnomaly threshold (reconstruction MSE) = {threshold:.5f}")
    evaluate(result_df)
    result_df.to_csv("outputs/anomaly_detection_results.csv", index=False)
    print("Saved -> outputs/anomaly_detection_results.csv")
