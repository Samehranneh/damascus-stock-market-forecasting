# Data

Daily trading data for 27 companies listed on the Damascus Securities Exchange, January 2010 to September 2024: banks, insurers, telecoms (Syriatel, MTN), industry and services.

Each file is semicolon-separated:

```
date;openning;up;down;closing_price;changging;t_size
```

| Column | Meaning |
|---|---|
| `date` | Trading day, `dd/mm/YYYY` |
| `openning` | Opening price (SYP) |
| `up` / `down` | Daily high / low |
| `closing_price` | Closing price |
| `changging` | Daily change, % |
| `t_size` | Trading volume |

Place the CSV files in this folder. The code picks them up automatically.
