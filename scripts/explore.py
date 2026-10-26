"""Day 1 data exploration: fail rate, missing data, constant sensors, fails over time.

Run: uv run python scripts/explore.py
"""
import pandas as pd

from fabpqagent.data import sensor_columns
from fabpqagent.db import load_lots

MAX_MISSING_FRACTION = 0.5

df = load_lots()
sensors = df[sensor_columns()]
print("shape (lots, columns):", df.shape)

# --- 1a. Fail rate -----------------------------------------------------------
n_lots = len(df)
n_fails = int(df["failed"].sum())          # failed is 0/1, so the sum counts the 1s
fail_rate = df["failed"].mean()            # mean of a 0/1 column = fraction of 1s
print("\n1a. Fail rate")
print(f"  lots={n_lots}  fails={n_fails}  fail rate={fail_rate:.1%}")
print(f"  accuracy of 'always pass' = {1 - fail_rate:.1%}")

# --- 1b. Missing data per sensor ---------------------------------------------
missing_frac = sensors.isna().mean()       # per column: fraction of lots where it is NaN
print("\n1b. Missing data")
print(f"  sensors missing in > 50% of lots: {(missing_frac > MAX_MISSING_FRACTION).sum()}")
print(f"  sensors with no missing values:   {(missing_frac == 0).sum()}")
print("  5 worst sensors:")
print(missing_frac.sort_values(ascending=False).head(5).to_string())

# --- 1c. Constant sensors ----------------------------------------------------
std = sensors.std()                        # NaN for an all-NaN column, 0 for a constant one
constant = std == 0
print("\n1c. Constant sensors")
print(f"  std == 0: {constant.sum()}   std is NaN: {std.isna().sum()}")
keep = (missing_frac <= MAX_MISSING_FRACTION) & (std > 0)   # NaN > 0 is False -> dropped
print(f"  features left after both filters: {keep.sum()} of {len(keep)}")

# --- 1d. Fails over time -----------------------------------------------------
df["ts"] = pd.to_datetime(df["ts"])
weekly = df.set_index("ts").resample("W")["failed"].agg(["size", "sum", "mean"])
weekly.columns = ["lots", "fails", "fail_rate"]
print("\n1d. Fails per week (weeks end on Sunday)")
print(weekly.to_string(float_format="{:.1%}".format))
print(f"  week with most fails:     {weekly['fails'].idxmax().date()} ({weekly['fails'].max()} fails)")
busy = weekly[weekly["lots"] >= 20]        # ignore near-empty weeks, where the rate is noise
print(f"  highest rate (>=20 lots): {busy['fail_rate'].idxmax().date()} ({busy['fail_rate'].max():.1%})")

aug = df[(df["ts"] >= "2008-08-01") & (df["ts"] < "2008-09-01")]
print(f"\n  August 2008: lots={len(aug)}  fails={int(aug['failed'].sum())}")
