"""REST API: next-day price forecast and a text recommendation for a DSE company.

GET /api/companies            -> list of available companies
GET /api/predict/<company>    -> forecast, test scores, trend and recommendation

A Random Forest model is trained on request for the chosen company. The 30-day vs
60-day moving average gives the trend direction, which is passed as a prompt to
GPT-Neo to generate a short plain-language recommendation. GPT-Neo is optional:
set ENABLE_LLM=0 to skip it (it is a 2.7B-parameter download).
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

from flask import Flask, jsonify
from flask_cors import CORS
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

import pandas as pd

from data import chronological_split, list_companies, load_prices, make_supervised

DATA_DIR = Path(os.environ.get("DSE_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
ENABLE_LLM = os.environ.get("ENABLE_LLM", "1") == "1"
LLM_NAME = os.environ.get("LLM_NAME", "EleutherAI/gpt-neo-2.7B")

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
COMPANIES = list_companies(DATA_DIR)


@lru_cache(maxsize=1)
def load_llm():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(LLM_NAME)
    tok.pad_token = tok.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return tok, AutoModelForCausalLM.from_pretrained(LLM_NAME).to(device)


def recommendation(trend: str) -> str | None:
    if not ENABLE_LLM:
        return None
    try:
        tok, model = load_llm()
        prompt = (f"The current trend direction is {trend}. "
                  "Based on this trend, the recommendation for the stock market is:")
        inputs = tok(prompt, return_tensors="pt", padding=True).to(model.device)
        out = model.generate(**inputs, max_new_tokens=60, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0], skip_special_tokens=True).split("\n\n")[0]
    except Exception as e:  # keep the API usable without the LLM
        log.warning("LLM unavailable: %s", e)
        return None


@app.get("/api/companies")
def companies():
    return jsonify(sorted(COMPANIES))


@app.get("/api/predict/<company>")
def predict(company: str):
    path = COMPANIES.get(company)
    if path is None:
        return jsonify(status="error", message="Unknown company"), 404

    df = load_prices(path)
    X, y, close_t = make_supervised(df)
    (X_tr, y_tr), _, (X_te, y_te) = chronological_split(X, y)
    scaler = MinMaxScaler().fit(X_tr)
    # Random Forest was the most reliable model across companies (see README)
    model = RandomForestRegressor(n_estimators=200, min_samples_leaf=20, random_state=42)
    model.fit(scaler.transform(X_tr), y_tr)

    c_te = close_t.loc[X_te.index]
    actual = c_te * (1 + y_te)
    test_pred = c_te * (1 + model.predict(scaler.transform(X_te)))
    naive_mae = float(mean_absolute_error(actual, c_te))

    # the latest row has features but no target yet
    latest_X, _, _ = make_supervised(pd.concat([df, df.iloc[[-1]]]))
    next_ret = float(model.predict(scaler.transform(latest_X.iloc[[-1]]))[0])
    next_close = float(df["close"].iloc[-1]) * (1 + next_ret)

    trend = "upward" if df["close"].tail(30).mean() > df["close"].tail(60).mean() else "downward"
    return jsonify(
        status="success",
        company=company,
        last_date=str(df.index[-1].date()),
        last_close=float(df["close"].iloc[-1]),
        next_close_forecast=round(next_close, 2),
        test_r2=round(float(r2_score(actual, test_pred)), 4),
        test_mae=round(float(mean_absolute_error(actual, test_pred)), 2),
        naive_test_mae=round(naive_mae, 2),
        trend=trend,
        recommendation=recommendation(trend),
    )


if __name__ == "__main__":
    app.run(debug=False)
