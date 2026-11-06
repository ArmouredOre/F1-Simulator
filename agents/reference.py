"""Reference lap: minimum-curvature line + forward-backward point-mass speed profile."""
import sys
from pathlib import Path
import numpy as np
import osqp
from scipy.interpolate import CubicSpline
from scipy.sparse import diags, identity, vstack

TRACKS = Path(__file__).resolve().parent.parent / "tracks"
G, RHO = 9.81, 1.2
# ponytail: one generic 2024-style car, aero level per track (teams do change wings); swap for physics.py once B ships
CAR = dict(m=805, mu=1.8, p=750e3, vmax=100)
AERO = dict(monza=(3.0, 0.8), silverstone=(4.5, 1.1), monaco=(5.0, 1.3), spa=(3.5, 0.9), suzuka=(4.5, 1.1), bahrain=(4.0, 1.0))  # (ClA, CdA) in m2
R_MIN = dict(monaco=12.0)  # m, line limit; default 15. Monaco's hairpin is too tight for 15 (QP infeasible)
MU = dict(monaco=1.5, suzuka=1.6, bahrain=1.5)  # default 1.8; Monaco is bumpy, low-grip street tarmac and nobody commits near the walls (fitted to pole)
POLE = dict(monza=79.327, silverstone=85.819, monaco=70.270, spa=100.562, suzuka=88.197, bahrain=89.165)  # dry pole laps, s (2024; Spa is 2025, as 2024 qualifying was damp)


def min_curvature_line(t, step=2, margin=1.0, r_min=15.0):
    """Lateral offsets a (left +) minimising sum of line curvature^2, with |curvature| <= 1/r_min.

    Line curvature is linearised in a: k ~ kc + kc^2 a + a'' (offset-curve formula), so it's one sparse QP.
    """
    # ponytail: linearised curvature, one QP; TUM iterates the nonlinear version if this proves too loose
    i = np.arange(0, len(t["s"]), step)
    kc, n = t["kappa"][i], len(i)
    D = diags([1.0, 1.0, -2.0, 1.0, 1.0], [-(n - 1), -1, 0, 1, n - 1], shape=(n, n)) / step**2
    A = (D + diags(kc**2)).tocsc()
    qp = osqp.OSQP()
    qp.setup(P=(2 * A.T @ A).tocsc(), q=2 * A.T @ kc, A=vstack([identity(n), A]).tocsc(),
             l=np.r_[-t["w_right"][i] + margin, -1 / r_min - kc], u=np.r_[t["w_left"][i] - margin, 1 / r_min - kc],
             verbose=False, eps_abs=1e-6, eps_rel=1e-6, max_iter=100000)
    res = qp.solve()
    assert res.info.status in ("solved", "solved inaccurate"), f"no line within the track at r_min={r_min} m: {res.info.status}"
    a = res.x
    alpha = CubicSpline(np.r_[t["s"][i], t["s"][-1] + t["s"][1]], np.r_[a, a[0]], bc_type="periodic")(t["s"])
    tx, ty = np.roll(t["x"], -1) - np.roll(t["x"], 1), np.roll(t["y"], -1) - np.roll(t["y"], 1)
    h = np.hypot(tx, ty)
    return t["x"] - ty / h * alpha, t["y"] + tx / h * alpha


def curvature(x, y):
    d = lambda v: (np.roll(v, -1) - np.roll(v, 1)) / 2
    dx, dy = d(x), d(y)
    return (dx * d(dy) - dy * d(dx)) / (dx * dx + dy * dy) ** 1.5


def speed_profile(kappa, ds, m, mu, p, vmax, cla, cda):
    q, k = 0.5 * RHO, np.abs(kappa)
    den = m * k - mu * q * cla  # cornering limit: m k v^2 = mu (m g + q ClA v^2)
    v = np.minimum(np.where(den > 0, np.sqrt(mu * m * G / np.maximum(den, 1e-12)), vmax), vmax)
    i0 = np.argmin(v)  # start at the slowest corner so the loop closes
    v, k, ds = [np.r_[np.roll(a, -i0), np.roll(a, -i0)[0]] for a in (v, k, ds)]
    grip = lambda vi, ki: np.sqrt(max((mu * (G + q * cla * vi * vi / m)) ** 2 - (vi * vi * ki) ** 2, 0))
    drag = lambda vi: q * cda * vi * vi / m
    for i in range(len(v) - 1):  # forward: power and grip limited
        a = min(grip(v[i], k[i]), p / (m * max(v[i], 1))) - drag(v[i])
        v[i + 1] = min(v[i + 1], np.sqrt(max(v[i] ** 2 + 2 * a * ds[i], 0)))
    for i in range(len(v) - 1, 0, -1):  # backward: braking, drag helps
        v[i - 1] = min(v[i - 1], np.sqrt(v[i] ** 2 + 2 * (grip(v[i], k[i]) + drag(v[i])) * ds[i - 1]))
    return np.roll(v[:-1], i0)


def reference_lap(name):
    t = dict(np.load(TRACKS / f"{name}.npz"))
    x, y = min_curvature_line(t, r_min=R_MIN.get(name, 15.0))
    ds = np.hypot(np.roll(x, -1) - x, np.roll(y, -1) - y)
    cla, cda = AERO[name]
    v = speed_profile(curvature(x, y), ds, **{**CAR, "mu": MU.get(name, CAR["mu"])}, cla=cla, cda=cda)
    dt = 2 * ds / (v + np.roll(v, -1))
    return dict(x=x, y=y, v=v, s=np.r_[0, np.cumsum(ds)[:-1]], t=np.r_[0, np.cumsum(dt)[:-1]], lap_time=dt.sum())


if __name__ == "__main__":
    ok = True
    for name in sys.argv[1:] or [p.stem for p in sorted(TRACKS.glob("*.npz"))]:
        r = reference_lap(name)
        err = (r["lap_time"] - POLE[name]) / POLE[name]
        r_min = 1 / np.abs(curvature(r["x"], r["y"])).max()
        ok &= abs(err) < 0.05 and r_min > R_MIN.get(name, 15.0) - 2  # QP holds the limit on a 2 m grid; allow a little between grid points
        print(f"{name:12s} ref {r['lap_time']:7.3f} s  pole {POLE[name]:7.3f} s  err {err:+.1%}  vmax {r['v'].max()*3.6:.0f} km/h  vmin {r['v'].min()*3.6:.0f} km/h  min R {r_min:.1f} m")
    assert ok, "gate failed: a lap is more than 5% off pole or its line is tighter than its minimum radius"
    print("gate passed")
