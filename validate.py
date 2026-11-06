"""Overlay a sim log on a real FastF1 lap: speed vs distance, g-g, sector deltas, lap-time error.

python validate.py logs/sim.parquet --real 2024 Monza Q [--driver LEC]
python validate.py logs/sim.parquet --ref logs/real.parquet
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def add_acc(df):
    """Fill AccLong/AccLat (m/s2) from Speed, Time, X, Y when a log lacks them."""
    v = df.Speed.to_numpy() / 3.6
    t = df.Time.to_numpy()
    dx, dy = np.gradient(df.X.to_numpy()), np.gradient(df.Y.to_numpy())
    heading = np.unwrap(np.arctan2(dy, dx))
    heading = pd.Series(heading).rolling(9, center=True, min_periods=1).mean().to_numpy()  # ~1 s smoothing, raw X/Y is noisy
    df["AccLong"] = np.gradient(v, t)
    df["AccLat"] = v * np.gradient(heading, t)  # a_lat = v * yaw rate, + = left
    return df


def fastf1_lap(year, gp, session, driver=None, cache="ff1cache"):
    """Fastest lap of a FastF1 session, converted to the log format at 10 Hz."""
    import os
    import fastf1
    os.makedirs(cache, exist_ok=True)  # FastF1 errors if the cache folder is missing
    fastf1.Cache.enable_cache(cache)
    s = fastf1.get_session(year, gp, session)
    s.load(weather=False, messages=False)
    lap = (s.laps.pick_drivers(driver) if driver else s.laps).pick_fastest()
    tel = lap.get_telemetry()
    t = tel.Time.dt.total_seconds().to_numpy()
    t10 = np.arange(0, t[-1], 0.1)
    df = pd.DataFrame({c: np.interp(t10, t, tel[c].astype(float)) for c in
                       ["Speed", "RPM", "nGear", "Throttle", "Brake", "X", "Y", "Distance"]})
    df["X"], df["Y"] = df.X / 10, df.Y / 10  # FastF1 X/Y are in 1/10 m
    df["Brake"] = df.Brake > 0.5
    df["nGear"] = df.nGear.round().astype(int)
    df["Time"], df["Driver"], df["LapNumber"] = t10, lap.Driver, int(lap.LapNumber)
    return add_acc(df)


def one_lap(df):
    """First car, first lap in the log, with Time and Distance starting at 0."""
    df = df[df.Driver == df.Driver.iloc[0]]
    df = df[df.LapNumber == df.LapNumber.iloc[0]].copy()
    df["Time"] -= df.Time.iloc[0]
    df["Distance"] -= df.Distance.iloc[0]
    if "AccLat" not in df:
        add_acc(df)
    return df


def compare(sim, ref):
    """Lap-time error and per-sector delta (sim - real, s). Sectors are distance thirds."""
    # ponytail: equal-distance sectors, use the official sector distances once track.py exports them
    L = min(sim.Distance.iloc[-1], ref.Distance.iloc[-1])
    edges = np.linspace(0, L, 4)
    ts = np.interp(edges, sim.Distance, sim.Time)
    tr = np.interp(edges, ref.Distance, ref.Time)
    sectors = np.diff(ts) - np.diff(tr)
    lap_sim, lap_ref = sim.Time.iloc[-1], ref.Time.iloc[-1]
    return dict(lap_sim=lap_sim, lap_ref=lap_ref, lap_err_s=lap_sim - lap_ref,
                lap_err_pct=100 * (lap_sim - lap_ref) / lap_ref, sector_delta_s=sectors.tolist())


def plot(sim, ref, res, out=None):
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw=dict(width_ratios=[2, 1]))
    for df, name in [(ref, "real"), (sim, "sim")]:
        a.plot(df.Distance, df.Speed, label=name)
        b.scatter(df.AccLat / 9.81, df.AccLong / 9.81, s=3, label=name)
    a.set(xlabel="Distance (m)", ylabel="Speed (km/h)",
          title=f"Lap: sim {res['lap_sim']:.3f}s  real {res['lap_ref']:.3f}s  "
                f"err {res['lap_err_s']:+.3f}s ({res['lap_err_pct']:+.2f}%)  sectors "
                + " ".join(f"{d:+.3f}" for d in res["sector_delta_s"]))
    b.set(xlabel="Lateral g", ylabel="Longitudinal g", title="g-g", xlim=(-6, 6), ylim=(-6, 3))
    a.legend(), b.legend()
    fig.tight_layout()
    plt.savefig(out) if out else plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("sim")
    p.add_argument("--ref", help="log parquet to compare against")
    p.add_argument("--real", nargs=3, metavar=("YEAR", "GP", "SESSION"))
    p.add_argument("--driver")
    p.add_argument("--save-ref", help="also write the real lap as a log parquet")
    p.add_argument("--out", help="save the plot instead of showing it")
    a = p.parse_args()
    ref = pd.read_parquet(a.ref) if a.ref else fastf1_lap(int(a.real[0]), a.real[1], a.real[2], a.driver)
    if a.save_ref:
        ref.to_parquet(a.save_ref)
    sim, ref = one_lap(pd.read_parquet(a.sim)), one_lap(ref)
    res = compare(sim, ref)
    print(res)
    plot(sim, ref, res, a.out)
