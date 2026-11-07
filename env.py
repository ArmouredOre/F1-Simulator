"""RaceEnv: multi-car environment. Physics at 100 Hz, actions at 10 Hz, logs in FastF1 channel names.
Action per car: [steer -1..1 (+left), throttle_brake -1..1, straight_mode 0/1, overtake 0/1,
deploy_bias -1..1 (+ deploy MGU-K, - harvest on throttle)]."""
import pathlib
import numpy as np
import pandas as pd
import physics
import rules

HERE = pathlib.Path(__file__).parent
DT, SUBSTEPS = 0.01, 10
LOOK = np.arange(1, 25) * 20.0   # obs lookahead: 24 points, 20..480 m (about 6 s at speed)
WING = dict(silverstone=1.2, monaco=1.3)   # default wing per track (1.0 = medium, Monza-like trim), est


def synthetic_track():
    """Rounded rectangle (1400 x 600 m) with 40/120/250/80 m corners, in the tracks/*.npz format."""
    W, H, rad = 1400.0, 600.0, [40.0, 120.0, 250.0, 80.0]
    sides = [W - rad[3] - rad[0], H - rad[0] - rad[1], W - rad[1] - rad[2], H - rad[2] - rad[3]]
    kappa, zones, s = [], [], 0.0
    for side, rc in zip(sides, rad):
        zones.append([s + 50, s + side - 150])
        kappa += [0.0] * int(side) + [1 / rc] * int(round(rc * np.pi / 2))
        s += int(side) + int(round(rc * np.pi / 2))
    kappa = np.array(kappa)
    psi = np.concatenate([[0], np.cumsum(kappa[:-1])])
    x, y = np.cumsum(np.cos(psi)), np.cumsum(np.sin(psi))
    n = len(kappa)
    return dict(s=np.arange(n, dtype=float), x=x, y=y, w_left=np.full(n, 6.0), w_right=np.full(n, 6.0),
                kappa=kappa, zones=np.array([z for z in zones if z[1] > z[0]]))


def load_track(name):
    f = HERE / "tracks" / f"{name}.npz"
    t = dict(np.load(f)) if f.exists() else synthetic_track()
    dx, dy = np.gradient(t["x"]), np.gradient(t["y"])
    norm = np.hypot(dx, dy)
    t["tx"], t["ty"] = dx / norm, dy / norm
    t["ds"] = t["s"][1] - t["s"][0]
    t["length"] = t["s"][-1] + t["ds"]
    return t


class RaceEnv:
    def __init__(self, track="oval", season="2026", n_cars=1, conditions=None, compound="medium",
                 max_time_s=600.0, log=True, wing=None):
        self.trk, self.R = load_track(track), rules.load(season, WING.get(track, 1.0) if wing is None else wing)
        self.n = n_cars
        self.cond = {**dict(air_c=26, track_c=42, rain_mm_h=0, wind_ms=0, wind_dir=0), **(conditions or {})}
        self.compounds = [compound] * n_cars if isinstance(compound, str) else list(compound)
        self.max_time, self.logging = max_time_s, log
        self.names = [f"car_{i}" for i in range(n_cars)]

    def reset(self, seed=None, v0=50.0):
        self.rng = np.random.default_rng(seed)
        L = self.trk["length"]
        s = (-8.0 * np.arange(self.n)) % L                     # 8 m apart, car_0 in front
        n = np.where(np.arange(self.n) % 2, -2.0, 2.0) if self.n > 1 else np.zeros(1)
        self.st = physics.init_state(s, n, np.full(self.n, v0), self.compounds, self.cond, self.R)
        self.rho = physics.air_density(self.cond)
        self.t, self.lap = 0.0, np.where(s > L / 2, 0, 1)        # cars behind the line start lap 1 on crossing
        self.lap_start = np.zeros(self.n)
        self.lap_times = [[] for _ in range(self.n)]
        self.done = np.zeros(self.n, bool)
        self.rows = []
        self._gaps()
        return self._obs(), {"lap_times": self.lap_times}

    def _gaps(self):
        self.wk, d, self.dn_ahead, self.ahead = physics.wake(self.st, self.trk["length"])
        self.d_ahead = d
        self.gap = d / np.maximum(self.st["vx"], 1.0)        # seconds to the car ahead
        self.gap[~np.isfinite(self.gap)] = np.inf

    def step(self, actions):
        a = np.array([actions.get(k, [0, -1, 0, 0, 0]) for k in self.names], float).reshape(self.n, 5)
        a[self.done] = [0, -1, 0, 0, 0]                        # retired cars brake to a stop
        steer, tb = np.clip(a[:, 0], -1, 1), np.clip(a[:, 1], -1, 1)
        thr, brk = np.maximum(tb, 0), np.maximum(-tb, 0)
        st, R = self.st, self.R
        mode = rules.straight_mode(R, a[:, 2] > 0.5, st["s"], self.trk["zones"], self.gap, brk > 0)
        ot = rules.overtake(R, a[:, 3] > 0.5, self.gap)
        st["ot_lap"] |= ot
        dep, harv = rules.mguk(R, st["vx"], np.clip(a[:, 4], -1, 1), ot, st)
        s0 = st["s"].copy()
        for _ in range(SUBSTEPS):
            physics.step(st, steer, thr, brk, mode, dep, harv, self.wk, R, self.trk, self.cond, self.rho, DT)
        hit, wall = physics.collide(st, self.trk, DT * SUBSTEPS)
        self.t += DT * SUBSTEPS
        progress = st["s"] - s0

        L = self.trk["length"]
        for i in np.where(st["s"] >= L)[0]:                    # crossed the line
            if self.lap[i] > 0:
                self.lap_times[i].append(self.t - self.lap_start[i])
            self.lap_start[i], self.lap[i] = self.t, self.lap[i] + 1
            st["dep_lap"][i] = st["harv_lap"][i] = 0
            st["ot_lap"][i] = False
        st["s"] %= L
        self._gaps()

        w = np.where(st["n"] > 0, physics.at(self.trk, "w_left", st["s"]), physics.at(self.trk, "w_right", st["s"]))
        off = np.abs(st["n"]) > w + 1.0                        # all four wheels past the line
        crash = (st["dmg"] >= 1.0) | (np.abs(st["xi"]) > 1.2)       # wrecked or spun
        rew = progress / 10.0 - 1.0 * off - 5.0 * (hit | wall) - 50.0 * (crash & ~self.done)
        self.done |= crash
        if self.logging:
            self._log(a, mode, ot)
        trunc = self.t >= self.max_time
        k = self.names
        return (self._obs(), dict(zip(k, rew)), dict(zip(k, self.done.copy())),
                dict(zip(k, [trunc] * self.n)), {"lap_times": self.lap_times, "lap": self.lap.copy()})

    def _obs(self):
        st, t, L = self.st, self.trk, self.trk["length"]
        ahead = (st["s"][:, None] + LOOK[None, :]) % L
        look = [physics.at(t, k, ahead) for k in ("kappa", "w_left", "w_right")]
        behind = ((st["s"][None, :] - st["s"][:, None]) % L).T  # how far i is ahead of j, by row j
        np.fill_diagonal(behind, np.inf)
        gap_b = behind.min(1) / np.maximum(st["vx"], 1.0)
        R = self.R
        zone = rules.in_zones(st["s"], t["zones"]).astype(float)
        ot_ok = (self.gap <= R["rules"]["overtake"]["gap_s"]) if R["rules"]["overtake"] else np.zeros(self.n)
        c = self.cond
        o = np.column_stack([
            look[0] * 100, look[1] / 10, look[2] / 10,
            st["vx"] / 100, st["vy"] / 10, st["r"], st["ax"] / 50, st["ay"] / 50, st["n"] / 10, st["xi"],
            st["delta"], (st["Ts"] - 100) / 50, st["wear"], st["batt"] / R["pu"]["battery_mj"],
            ot_ok, zone,
            np.full((self.n, 4), [c["air_c"] / 40, c["track_c"] / 60, c["rain_mm_h"] / 10, c["wind_ms"] / 10]),
            np.minimum(self.gap, 5) / 5, np.minimum(gap_b, 5) / 5, self.dn_ahead.clip(-10, 10) / 10,
            (st["vx"][self.ahead] - st["vx"]) / 20 * (self.n > 1),
        ]).astype(np.float32)
        return dict(zip(self.names, o))

    def _log(self, a, mode, ot):
        st, t = self.st, self.trk
        x = np.interp(st["s"], t["s"], t["x"]) - physics.at(t, "ty", st["s"]) * st["n"]
        y = np.interp(st["s"], t["s"], t["y"]) + physics.at(t, "tx", st["s"]) * st["n"]
        gear, rpm = physics.gear_rpm(st["vx"])
        for i, name in enumerate(self.names):
            row = dict(Time=self.t, Driver=name, Speed=st["vx"][i] * 3.6, RPM=rpm[i], nGear=int(gear[i]),
                       Throttle=st["thr"][i] * 100, Brake=bool(st["brk"][i] > 0), X=x[i], Y=y[i],
                       Distance=st["s"][i], LapNumber=int(self.lap[i]), Steer=st["delta"][i],
                       AccLat=st["ay"][i], AccLong=st["ax"][i], Battery_MJ=st["batt"][i],
                       Damage=st["dmg"][i], AeroMode=int(mode[i]), Overtake=bool(ot[i]), Compound=st["compounds"][i])
            for j, wh in enumerate(("FL", "FR", "RL", "RR")):
                row[f"TyreTemp_{wh}"], row[f"TyreWear_{wh}"] = st["Ts"][i, j], st["wear"][i, j]
            self.rows.append(row)

    def save_log(self, run):
        path = HERE / "logs" / f"{run}.parquet"
        path.parent.mkdir(exist_ok=True)
        pd.DataFrame(self.rows).to_parquet(path)
        return path
