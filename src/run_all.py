"""Run the model comparison for every company in data/ and write docs/results_summary.csv."""
import contextlib
import glob
import io
import warnings
from pathlib import Path

import pandas as pd

from compare_models import run
from data import load_prices

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
rows = []
for f in sorted(glob.glob(str(ROOT / "data" / "*.csv"))):
    name = Path(f).name.split(" 2010")[0].strip()
    n = len(load_prices(f))
    if n < 200:
        print(f"skip {name}: only {n} trading days")
        continue
    with contextlib.redirect_stdout(io.StringIO()):
        t = run(f, out_dir=str(ROOT / "results"), use_lstm=False)
    t = t[t.split == "test"].set_index("model")
    naive = t.loc["Naive (last close)", "RMSE"]
    best = t.drop("Naive (last close)").RMSE.idxmin()
    rows.append(dict(company=name, trading_days=n, naive_rmse=round(naive, 3),
                     best_model=best, best_rmse=round(t.loc[best, "RMSE"], 3),
                     improvement_pct=round((1 - t.loc[best, "RMSE"] / naive) * 100, 1) if naive else None))
df = pd.DataFrame(rows)
df.to_csv(ROOT / "docs" / "results_summary.csv", index=False)
print(df.to_string(index=False))
