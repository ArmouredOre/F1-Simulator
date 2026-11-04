"""Gate 1 check: measure the car's g-g-v envelope by running physics.step (braking, traction and
constant-speed ramp-steer tests), then drive a quasi-steady lap with it on the reference line.
Run: python gate.py [season] [tracks...]"""
import sys
import numpy as np
import physics
import rules
import tyres
from env import WING
from agents.reference import TRACKS, POLE, min_curvature_line, curvature

V = np.arange(8.0, 106.0, 4.0)    # m/s
COND = dict(air_c=26, track_c=42)
FLAT = dict(s=np.arange(2.0), kappa=np.zeros(2), tx=np.ones(2), ty=np.zeros(2), ds=1.0, length=1e9)


def hold(R, v, steer=0.0, thr=0.0, brk=0.0, straight=False, steps=30):
    """Run physics at a held speed until loads settle; return the state."""
    st = physics.init_state([0.0], [0.0], [v], ["soft"], COND, R)
    st["Ts"][:] = st["Tb"][:] = tyres.COMPOUNDS["soft"]["t_opt"]      # qualifying: tyres in the window
    st["batt"][:] = R["pu"]["battery_mj"]
    dep, harv = rules.mguk(R, np.array([v]), np.ones(1), np.zeros(1, bool), st)
    rho = physics.air_density(COND)
    for _ in range(steps):
        physics.step(st, np.array([steer]), np.array([thr]), np.array([brk]), np.array([straight]),
                     dep, harv, np.zeros(1), R, FLAT, COND, rho, 0.01)
        st["vx"][:], st["s"][:], st["n"][:], st["xi"][:] = v, 0.0, 0.0, 0.0
    return st


def lateral_max(R, v):
    """Constant-speed ramp steer: highest steady |ay| before the car slides (body slip > 6 deg)."""
    best = 0.0
    for steer in np.linspace(0.005, 1.0, 80):
        st = hold(R, v, steer=steer, thr=0.0, steps=60)
        st["vx"][:] = v
        if abs(st["vy"][0] / v) > 0.1:
            break
        best = max(best, abs(st["ay"][0]))
    return best


def ggv(R):
    acc = np.array([hold(R, v, thr=1.0, straight=True)["ax"][0] for v in V])
    brk = np.array([-hold(R, v, brk=1.0)["ax"][0] for v in V])
    lat = np.array([lateral_max(R, v) for v in V])
    return acc, brk, lat


def qss_lap(R, name, env_ggv):
    acc, brk, lat = env_ggv
    t = dict(np.load(TRACKS / f"{name}.npz"))
    x, y = min_curvature_line(t)
    # ponytail: 15 m smoothing, the QP line has a few unphysically tight spots (R < 10 m, past full lock)
    k = np.convolve(np.abs(np.tile(curvature(x, y), 3)), np.ones(15) / 15, "same")[len(x):2 * len(x)]
    ds = np.hypot(np.roll(x, -1) - x, np.roll(y, -1) - y)
    f = lambda tab, v: np.interp(v, V, tab)
    vg = np.linspace(V[0], V[-1], 400)
    ok = f(lat, vg)[None, :] >= vg[None, :] ** 2 * k[:, None]          # corner speed limit
    v = np.where(ok.any(1), vg[np.where(ok, np.arange(400), 0).max(1)], V[0])
    use = lambda vi, ki: np.sqrt(max(0.0, 1 - (vi * vi * ki / f(lat, vi)) ** 2))
    for _ in range(2):                                                  # loop twice so the lap closes
        for i in range(len(v)):
            j = (i + 1) % len(v)
            a = f(acc, v[i]) * use(v[i], k[i])
            v[j] = min(v[j], np.sqrt(max(v[i] ** 2 + 2 * a * ds[i], 0)))
        for i in range(len(v) - 1, -1, -1):
            j = (i + 1) % len(v)
            v[i] = min(v[i], np.sqrt(v[j] ** 2 + 2 * f(brk, v[j]) * use(v[j], k[j]) * ds[i]))
    return (2 * ds / (v + np.roll(v, -1))).sum(), v


if __name__ == "__main__":
    season = sys.argv[1] if len(sys.argv) > 1 else "2025"
    env_ggv = ggv(rules.load(season))
    print(f"season {season} envelope at medium wing (g):")
    for kmh in (100, 200, 300):
        i = np.argmin(abs(V * 3.6 - kmh))
        print(f"  {V[i] * 3.6:3.0f} km/h  accel {env_ggv[0][i] / 9.81:.2f}  brake {env_ggv[1][i] / 9.81:.2f}  lateral {env_ggv[2][i] / 9.81:.2f}")
    errs = []
    for name in sys.argv[2:] or ["monza", "silverstone"]:
        R = rules.load(season, WING.get(name, 1.0))
        lt, v = qss_lap(R, name, ggv(R))
        errs.append((lt - POLE[name]) / POLE[name])
        print(f"  {name:12s} wing {WING.get(name, 1.0)}  qss lap {lt:6.2f} s  2024 pole {POLE[name]:6.2f} s  err {errs[-1]:+.1%}  "
              f"vmax {v.max() * 3.6:.0f} km/h  vmin {v.min() * 3.6:.0f} km/h")
    if season == "2025":
        assert all(abs(e) < 0.05 for e in errs), "gate failed: more than 5% off pole"
        print("gate passed")
