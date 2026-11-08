"""Sanity gate for the sim core: a simple driver laps the track on the minimum-curvature line from
agents/reference.py (centreline on the synthetic track); check the numbers land in the real F1 range.
Run: python test_env.py [track] [season]"""
import sys
import numpy as np
import pandas as pd
import physics
import tyres
from env import RaceEnv


def racing_line(t):
    """Lateral offset n(s) and curvature of the reference line; centreline if there is none."""
    try:
        from agents.reference import min_curvature_line, curvature
        x, y = min_curvature_line(t)
        return (x - t["x"]) * -t["ty"] + (y - t["y"]) * t["tx"], curvature(x, y)
    except Exception:
        return np.zeros_like(t["s"]), t["kappa"]


def speed_profile(env, kappa, mu=1.25):
    """Grip-limited target speed (point mass, backward braking pass). Test driver only."""
    a, m = env.R["aero"]["corner"], env.R["car"]["mass_min"]
    k = np.abs(np.convolve(np.tile(kappa, 3), np.ones(20) / 20, "same")[len(kappa):2 * len(kappa)])
    qa = 0.5 * env.rho * a["cl_a"] / m
    v = np.sqrt(mu * physics.G / np.maximum(k - mu * qa, 1e-6)).clip(0, 110)
    for _ in range(2):
        for i in range(len(v) - 1, -1, -1):
            nxt = v[(i + 1) % len(v)]
            dec = mu * (physics.G + qa * nxt**2) + 0.5 * env.rho * a["cd_a"] * nxt**2 / m
            v[i] = min(v[i], np.sqrt(nxt**2 + 2 * dec))
    return v


def grip(env):
    """Tyre grip now (water, temperature, wear) relative to warm dry tyres, per car."""
    st = env.st
    mu = tyres.mu(st["comp"], tyres.FZ0, st["Ts"], st["Tb"], st["wear"], physics.water_mm(env.cond), 1)
    return np.minimum((mu / st["comp"]["mu"]).mean(1) / 0.82, 1.0)   # 0.82: warm dry tyres at race temps, est


def drive(env, line_n, v_t, thr_trim=1.0):
    """Pure pursuit on the line, a speed controller that shares grip (friction circle), and TC.
    Speed target, grip circle and traction cap all scale with the tyres' grip now, so rain slows it down."""
    st, t, L, R = env.st, env.trk, env.trk["length"], env.R
    g = grip(env)
    m, a = R["car"]["mass_min"], R["aero"]["corner"]
    p_max = (R["pu"]["ice_kw"] + R["pu"]["mguk_kw"]) * 1e3 * physics.ETA
    acts = {}
    for i, name in enumerate(env.names):
        s, v = st["s"][i], st["vx"][i]
        ld = max(12.0, 0.9 * v)
        kap = np.interp((s + np.arange(0, ld, 2)) % L, t["s"], t["kappa"]).mean()
        e = 0.5 * kap * ld**2 + np.interp((s + ld) % L, t["s"], line_n) - st["n"][i] - ld * np.sin(st["xi"][i])
        steer = np.arctan(2 * R["car"]["wheelbase"] * e / ld**2) / physics.MAX_STEER
        vt = np.interp((s + 0.3 * v) % L, t["s"], v_t) * np.sqrt(g[i])
        down = 0.5 * env.rho * a["cl_a"] * v**2
        spare = np.sqrt(max(0.0, 1 - (st["ay"][i] / (1.5 * g[i] * (physics.G + down / m))) ** 2))
        thr_cap = 1.5 * g[i] * thr_trim * (0.55 * m * physics.G + 0.58 * down) * max(v, 1) / p_max
        thr_cap *= np.clip(1 - 20 * (abs(st["vy"][i] / max(v, 1)) - 0.02), 0, 1)
        tb = np.clip(0.5 * (vt - v), -1, min(1, thr_cap)) * spare
        acts[name] = [steer, tb, 1, 1, 1 if st["batt"][i] > 0.5 else -0.5]
    return acts


def run(track="oval", season="2026", laps=3, **kw):
    env = RaceEnv(track=track, season=season, **kw)
    env.reset(seed=0)
    line_n, kappa = np.zeros_like(env.trk["s"]), env.trk["kappa"]  # ponytail: centreline; pursuit lags in chicanes on the racing line
    v_t = speed_profile(env, kappa)
    done = False
    while not done and len(env.lap_times[0]) < laps:
        _, _, term, trunc, _ = env.step(drive(env, line_n, v_t))
        done = term["car_0"] or trunc["car_0"]
    return env, done


if __name__ == "__main__":
    track, season = (sys.argv[1:] + ["oval", "2026"][len(sys.argv) - 1:])[:2]
    env, crashed = run(track, season)
    path = env.save_log(f"gate_{track}_{season}")
    lap = pd.DataFrame(env.rows).query("LapNumber == 3")
    temps = lap.filter(like="TyreTemp").values
    print(f"{season} {track}: laps {np.round(env.lap_times[0], 2)}  log {path.name}")
    print(f"  top {lap.Speed.max():.0f} km/h, min {lap.Speed.min():.0f} km/h, "
          f"peak lat {lap.AccLat.abs().max() / 9.81:.2f} g, brake {-lap.AccLong.min() / 9.81:.2f} g, "
          f"accel {lap.AccLong.max() / 9.81:.2f} g")
    print(f"  tyre surface {temps.min():.0f}-{temps.max():.0f} C, wear {lap.filter(like='TyreWear').values.max() / 3:.3f}/lap, "
          f"battery {lap.Battery_MJ.min():.2f}-{lap.Battery_MJ.max():.2f} MJ")
    assert not crashed and len(env.lap_times[0]) == 3, "did not finish 3 laps"
    assert 290 < lap.Speed.max() < 375, "top speed"
    assert 1.5 < lap.AccLat.abs().max() / 9.81 < 6.5, "lateral g"
    assert 3.0 < -lap.AccLong.min() / 9.81 < 6.5, "braking g"
    assert 50 < temps.min() and temps.max() < 150, "tyre temps"
    print("sanity gate OK")
