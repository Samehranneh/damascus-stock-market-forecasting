"""Compare forecasting models on one DSE company.

Models: Holt-Winters, LSTM, XGBoost, CatBoost, Random Forest, SVR, and a
naive baseline (tomorrow = today). Prints R2, RMSE and MAE on validation and
test sets and saves a chart to results/.

Usage:
    python src/compare_models.py "data/Syriatel Mobile Telecom S.A 2010-01-01 to 2024-09-01 .csv"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from xgboost import XGBRegressor

from data import chronological_split, load_prices, make_supervised

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None


def metrics(y_true, y_pred) -> dict:
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": mean_absolute_error(y_true, y_pred),
    }


def build_lstm(n_features: int):
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.models import Sequential

    model = Sequential([
        LSTM(50, activation="relu", input_shape=(1, n_features), return_sequences=True),
        Dropout(0.2),
        LSTM(50, activation="relu"),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def run(csv_path: str, out_dir: str = "results", use_lstm: bool = True) -> pd.DataFrame:
    df = load_prices(csv_path)
    X, y = make_supervised(df)
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = chronological_split(X, y)

    scaler = MinMaxScaler().fit(X_tr)  # fit on training data only
    Xs_tr, Xs_va, Xs_te = (scaler.transform(a) for a in (X_tr, X_va, X_te))

    preds: dict[str, tuple] = {}

    # Baseline: tomorrow's close = today's close
    preds["Naive (last close)"] = (X_va["close"].values, X_te["close"].values)

    regressors = {
        "XGBoost": XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "SVR": SVR(kernel="rbf"),
    }
    if CatBoostRegressor is not None:
        regressors["CatBoost"] = CatBoostRegressor(
            iterations=100, learning_rate=0.1, depth=6, loss_function="RMSE", verbose=0
        )
    for name, model in regressors.items():
        model.fit(Xs_tr, y_tr)
        preds[name] = (model.predict(Xs_va), model.predict(Xs_te))

    # Holt-Winters on the closing-price series itself
    hw = ExponentialSmoothing(
        y_tr.values, trend="add", seasonal="add", seasonal_periods=12
    ).fit()
    hw_fc = hw.forecast(len(y_va) + len(y_te))
    preds["Holt-Winters"] = (hw_fc[: len(y_va)], hw_fc[len(y_va):])

    if use_lstm:
        try:
            lstm = build_lstm(Xs_tr.shape[1])
            r = lambda a: a.reshape(a.shape[0], 1, a.shape[1])
            lstm.fit(r(Xs_tr), y_tr.values, epochs=50, batch_size=32, verbose=0)
            preds["LSTM"] = (lstm.predict(r(Xs_va), verbose=0).ravel(),
                             lstm.predict(r(Xs_te), verbose=0).ravel())
        except ImportError:
            print("TensorFlow not installed, skipping LSTM.")

    rows = []
    for name, (pv, pt) in preds.items():
        rows.append({"model": name, "split": "validation", **metrics(y_va, pv)})
        rows.append({"model": name, "split": "test", **metrics(y_te, pt)})
    table = pd.DataFrame(rows)

    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    stem = Path(csv_path).stem.split(" 2010")[0].strip().replace(" ", "_")
    table.to_csv(out / f"{stem}_scores.csv", index=False)

    test = table[table.split == "test"].sort_values("RMSE")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(test.model, test.RMSE, color="#4B5A2A")
    ax.invert_yaxis()
    ax.set_xlabel("Test RMSE (SYP, lower is better)")
    ax.set_title(f"Next-day closing price: {stem.replace('_', ' ')}")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / f"{stem}_test_rmse.png", dpi=150)

    print(table.round(4).to_string(index=False))
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--no-lstm", action="store_true")
    a = ap.parse_args()
    run(a.csv, use_lstm=not a.no_lstm)
