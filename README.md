# Damascus Stock Market Forecasting

Forecasting next-day closing prices for companies on the **Damascus Securities Exchange (DSE)**, comparing classical time-series, machine-learning and deep-learning models, and serving the forecast through a REST API with an LLM-generated recommendation.

Master's project, M.Sc. Computer Science, Syrian Virtual University (2024).

## What it does

1. **Data:** 14 years of daily OHLCV prices (2010–2024) for 27 DSE-listed companies.
2. **Model comparison:** trains and scores seven approaches on each company:

   | Family | Models |
   |---|---|
   | Baseline | Naive (tomorrow = today) |
   | Time series | Holt-Winters exponential smoothing |
   | Machine learning | XGBoost, CatBoost, Random Forest, SVR |
   | Deep learning | Stacked LSTM (2 × 50 units, dropout 0.2) |

3. **API:** a Flask service trains XGBoost for a chosen company, returns the next-day forecast with its test error, detects the trend from 30- vs 60-day moving averages, and asks GPT-Neo 2.7B to phrase a short recommendation.

## Method

- Features on day *t*: open, high, low, close, change %, volume. Target: closing price on day *t + 1*.
- **Chronological split** 70 / 15 / 15 (train / validation / test). No shuffling, so the test period is genuinely in the future.
- Scaler fitted on the training set only.
- Metrics: R², RMSE and MAE, always reported next to the naive baseline. On daily stock prices, beating "tomorrow equals today" is the real test.

## Project structure

```
src/
  data.py            loading, cleaning, target construction, time-based split
  compare_models.py  trains all models for one company, saves scores and a chart
  app.py             Flask REST API with optional GPT-Neo recommendation
data/                CSV files (see data/README.md)
results/             generated scores and charts
```

## Run

```bash
pip install -r requirements.txt

# compare models for one company
python src/compare_models.py "data/Syriatel Mobile Telecom S.A 2010-01-01 to 2024-09-01 .csv"

# start the API (ENABLE_LLM=0 skips the 2.7B GPT-Neo download)
ENABLE_LLM=0 python src/app.py
curl http://127.0.0.1:5000/api/companies
curl http://127.0.0.1:5000/api/predict/syriatel-mobile-telecom-s-a
```

The API returns the company, last trading date and close, the next-day forecast, test R² and MAE, the trend direction, and the generated recommendation.

## Notes and limitations

- The DSE is a thin market: many stocks trade rarely, prices sit flat for long stretches, and there are regime shifts around currency changes. Expect large differences between companies.
- The LLM recommendation is a demonstration of combining a forecast with generated text. It is not financial advice.

## License

MIT
