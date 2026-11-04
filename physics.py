"""Car physics: dynamic single-track car in track (Frenet) coordinates, with per-wheel loads from
longitudinal and lateral load transfer, aero downforce/drag with wake and tow, and an energy store.
State is a dict of arrays shaped [n_cars] (tyre arrays [n_cars, 4], wheel order FL, FR, RL, RR)."""
import numpy as np
import tyres

G, ETA, C_RR = 9.81, 0.92, 0.015   # gravity, drivetrain efficiency, rolling resistance
MAX_STEER = 0.35                   # rad at the front wheels
GEAR_KMH = np.array([95, 130, 165, 200, 235, 270, 305, 360.0])  # top speed per gear, cosmetic
WAKE_LEN, WAKE_W = 50.0, 1.5       # m: wake decays over this distance behind / lateral offset


def air_density(cond):
    return cond.get("pressure_hpa", 1013) * 100 / (287.05 * (cond["air_c"] + 273.15))


def water_mm(cond):
    return min(0.1 * cond.get("rain_mm_h", 0), 3.0)  # ponytail: film depth straight from rain rate


def init_state(s, n, v, compounds, cond, R):
    k = len(s)
    z = np.zeros(k)
    return dict(s=np.array(s, float), n=np.array(n, float), xi=z.copy(), vx=np.array(v, float),
                vy=z.copy(), r=z.copy(), ax=z.copy(), ay=z.copy(), delta=z.copy(), thr=z.copy(),
                brk=z.copy(), Ts=np.full((k, 4), 80.0),
                Tb=np.full((k, 4), 80.0), wear=np.zeros((k, 4)),
                batt=np.full(k, R["pu"]["battery_mj"] / 2), dep_lap=z.copy(), harv_lap=z.copy(),
                ot_lap=np.zeros(k, bool), comp=tyres.comp(compounds), compounds=list(compounds))


def at(trk, key, s):
    """Track channel at distance s (nearest 1 m sample; np.interp with period re-sorts every call)."""
    return trk[key][(np.asarray(s) / trk["ds"]).astype(int) % len(trk[key])]


def wake(st, L):
    """Wake strength 0..1 per car from the nearest car ahead, plus gap to it in s and m."""
    ds = (st["s"][None, :] - st["s"][:, None]) % L   # [i, j]: how far j is ahead of i
    np.fill_diagonal(ds, np.inf)
    j = ds.argmin(1)
    d = ds[np.arange(len(j)), j]
    dn = st["n"][j] - st["n"]
    w = np.exp(-d / WAKE_LEN) * np.exp(-((dn / WAKE_W) ** 2))
    return np.where(np.isfinite(d), w, 0.0), d, dn, j


def step(st, steer, thr, brk, straight, deploy_kw, harvest_kw, wk, R, trk, cond, rho, dt):
    """Advance every car by dt seconds. thr and brk are 0..1; straight is the active aero mode."""
    c, a = R["car"], R["aero"]
    m, Lw, h, tw = c["mass_min"], c["wheelbase"], c["cg_height"], c["track_width"]
    lf = Lw * (1 - c["weight_front"])
    lr = Lw - lf
    vx = np.maximum(st["vx"], 1.0)
    vy, r = st["vy"], st["r"]
    delta = steer * MAX_STEER
    kap = at(trk, "kappa", st["s"])

    # aero, with wind along the car heading (wind_dir is where it blows from, 0 = +y)
    tx, ty = at(trk, "tx", st["s"]), at(trk, "ty", st["s"])
    hx, hy = tx * np.cos(st["xi"]) - ty * np.sin(st["xi"]), tx * np.sin(st["xi"]) + ty * np.cos(st["xi"])
    wd = np.radians(cond.get("wind_dir", 0.0))
    w = cond.get("wind_ms", 0.0)
    v_air = vx + w * (np.sin(wd) * hx + np.cos(wd) * hy)
    q = 0.5 * rho * v_air * np.abs(v_air)
    mode = lambda key: np.where(straight, a["straight"][key], a["corner"][key])
    df_f = q * mode("cl_a") * a["balance"] * (1 - a["wake"]["df_loss"] * wk)
    df_r = q * mode("cl_a") * (1 - a["balance"])
    drag = q * mode("cd_a") * (1 - a["wake"]["drag_cut"] * wk)

    # wheel loads with load transfer from last step's accelerations
    Wf = m * G * lr / Lw + df_f - m * st["ax"] * h / Lw
    Wr = m * G * lf / Lw + df_r + m * st["ax"] * h / Lw
    dlf = m * st["ay"] * h / tw * c["roll_front"]
    dlr = m * st["ay"] * h / tw * (1 - c["roll_front"])
    Fz = np.maximum(np.stack([Wf / 2 - dlf, Wf / 2 + dlf, Wr / 2 - dlr, Wr / 2 + dlr], 1), 50.0)
    wat = water_mm(cond)
    mu = tyres.mu(st["comp"], Fz, st["Ts"], st["Tb"], st["wear"], wat, R["tyres"]["grip_scale"])
    Fmax = mu * Fz

    # longitudinal: power-limited drive on the rears, brake split by bias, both capped by grip
    # ponytail: no wheel-spin states, so lock-up/wheelspin are clipped at the grip limit (ideal ABS/TC)
    P = thr * (R["pu"]["ice_kw"] + deploy_kw) * 1e3 * ETA
    drive = np.minimum(P / vx / 2, 0.98 * Fmax[:, 2:].min(1))
    fb = brk * 0.98 * Fmax.sum(1)
    bf = np.minimum(fb * c["brake_front"] / 2, 0.98 * Fmax[:, :2].min(1))
    br = np.minimum(fb * (1 - c["brake_front"]) / 2, 0.98 * Fmax[:, 2:].min(1))
    Fx = np.stack([-bf, -bf, drive - br, drive - br], 1)

    # lateral: Pacejka at the axle slip angle, reduced by the friction ellipse
    af = delta - np.arctan((vy + lf * r) / vx)
    ar = -np.arctan((vy - lr * r) / vx)
    alpha = np.stack([af, af, ar, ar], 1)
    Fy = tyres.lateral(alpha, Fz, mu) * np.sqrt(np.clip(1 - (Fx / Fmax) ** 2, 0, 1))

    Fxf, Fxr = Fx[:, :2].sum(1), Fx[:, 2:].sum(1)
    Fyf, Fyr = Fy[:, :2].sum(1), Fy[:, 2:].sum(1)
    cd, sd = np.cos(delta), np.sin(delta)
    ax = (Fxr + Fxf * cd - Fyf * sd - drag - C_RR * Fz.sum(1) * (st["vx"] > 0.1)) / m
    ay = (Fyr + Fyf * cd + Fxf * sd) / m
    yaw = (lf * (Fyf * cd + Fxf * sd) - lr * Fyr) / c["yaw_inertia"]

    # semi-implicit Euler: velocities first, then positions with the new velocities
    st["vx"] = np.maximum(st["vx"] + (ax + vy * r) * dt, 0.0)
    st["vy"] = vy + (ay - vx * r) * dt
    st["r"] = r + yaw * dt
    sx, sv = np.cos(st["xi"]), np.sin(st["xi"])
    sdot = (st["vx"] * sx - st["vy"] * sv) / (1 - st["n"] * kap)
    st["s"] = st["s"] + sdot * dt
    st["n"] = st["n"] + (st["vx"] * sv + st["vy"] * sx) * dt
    st["xi"] = st["xi"] + (st["r"] - kap * sdot) * dt

    # tyre heat and wear from sliding power
    P_slide = np.abs(Fy) * vx[:, None] * np.abs(np.sin(alpha)) + Fx**2 * vx[:, None] / (20 * Fz)
    st["Ts"], st["Tb"] = tyres.thermal(st["Ts"], st["Tb"], P_slide, Fz, vx, cond["air_c"],
                                       cond["track_c"], wat, dt)
    st["wear"] = st["wear"] + st["comp"]["wear"] * P_slide * dt / 1e6

    # energy store: deploy or harvest on throttle, regen under braking
    used = np.minimum(thr * deploy_kw * dt / 1e3, np.maximum(st["batt"], 0))   # < 0 = harvest on throttle
    regen = np.minimum(fb * vx, harvest_kw * 1e3) * dt / 1e6 + np.maximum(-used, 0)
    mguh = thr * (R["pu"].get("mguh_kw") or 0) * dt / 1e3                     # 2025 only, uncapped
    st["batt"] = np.minimum(st["batt"] + regen + mguh - np.maximum(used, 0), R["pu"]["battery_mj"])
    st["dep_lap"] += np.maximum(used, 0)
    st["harv_lap"] += regen
    st.update(ax=ax, ay=ay, delta=delta, thr=thr, brk=brk, Fz=Fz, sdot=sdot)


def gear_rpm(v):
    kmh = v * 3.6
    g = np.minimum(np.searchsorted(GEAR_KMH, kmh), 7)
    return g + 1, np.clip(11800 * kmh / GEAR_KMH[g], 4000, 12000)
