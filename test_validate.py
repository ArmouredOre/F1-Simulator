import numpy as np, pandas as pd
from validate import add_acc, compare, one_lap

# 30 m/s on a 100 m circle, counter-clockwise: a_lat = v^2/r = 9 m/s2 to the left, a_long = 0
t = np.arange(0, 2 * np.pi * 100 / 30, 0.1)
a = 30 * t / 100
df = pd.DataFrame(dict(Time=t, X=100 * np.cos(a), Y=100 * np.sin(a), Speed=30 * 3.6, Distance=30 * t, Driver="T", LapNumber=1))
df = add_acc(df)
assert abs(df.AccLat.median() - 9) < 0.1 and abs(df.AccLong).max() < 1e-6
slow = df.assign(Time=df.Time * 1.1)
r = compare(one_lap(slow), one_lap(df))
assert abs(r["lap_err_pct"] - 10) < 1e-6 and abs(sum(r["sector_delta_s"]) - r["lap_err_s"]) < 0.05
print("ok")
