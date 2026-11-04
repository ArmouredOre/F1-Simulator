"""Tyres: Pacejka lateral force, 2-node (surface/bulk) thermal model, sliding-energy wear.
Arrays are shaped [n_cars, 4] in wheel order FL, FR, RL, RR. All coefficients are fitted guesses."""
import numpy as np

# mu: peak friction at nominal load; t_opt/t_win: grip window (C); wear: life lost per MJ of sliding
# energy; d0: water depth (mm) that halves grip.
COMPOUNDS = {
    "soft":   dict(mu=1.86, t_opt=95,  t_win=22, wear=0.0060, d0=0.3),
    "medium": dict(mu=1.80, t_opt=100, t_win=28, wear=0.0040, d0=0.3),
    "hard":   dict(mu=1.74, t_opt=105, t_win=34, wear=0.0028, d0=0.3),
    "inter":  dict(mu=1.55, t_opt=70,  t_win=25, wear=0.0060, d0=1.5),
    "wet":    dict(mu=1.45, t_opt=60,  t_win=25, wear=0.0060, d0=4.0),
}
B, C, E = 15.0, 1.9, 0.97      # Magic Formula shape: peak near 7 deg slip
FZ0, LOAD_SENS = 5000.0, 0.12  # mu falls 12% for each +100% load over FZ0 (N)
C_S, C_B = 2500.0, 15000.0     # heat capacity of surface / bulk node (J/K)
H_SB, H_TRACK = 150.0, 60.0    # conductance surface-bulk, surface-track (W/K)
FRIC_IN, HYST = 0.20, 0.025    # share of sliding power into the tyre; bulk heating per (N m/s)


def comp(names):
    """Per-car compound coefficients as [n, 1] arrays."""
    return {k: np.array([[COMPOUNDS[c][k]] for c in names]) for k in COMPOUNDS["soft"]}


def mu(c, Fz, Ts, Tb, wear, water_mm, scale):
    T = 0.7 * Ts + 0.3 * Tb
    temp = 0.7 + 0.3 * np.exp(-(((T - c["t_opt"]) / c["t_win"]) ** 2))
    load = 1 - LOAD_SENS * (Fz / FZ0 - 1)
    return c["mu"] * scale * load * temp * (1 - 0.25 * np.clip(wear, 0, 1)) / (1 + water_mm / c["d0"])


def lateral(alpha, Fz, mu_):
    x = B * alpha
    return mu_ * Fz * np.sin(C * np.arctan(x - E * (x - np.arctan(x))))


def thermal(Ts, Tb, P_slide, Fz, v, air_c, track_c, water_mm, dt):
    v = v[:, None]
    h_air = 10 + 0.6 * v + 200 * water_mm   # forced convection, plus spray cooling in the wet
    dTs = FRIC_IN * P_slide - H_SB * (Ts - Tb) - h_air * (Ts - air_c) - H_TRACK * (Ts - track_c)
    dTb = HYST * Fz * v + H_SB * (Ts - Tb) - (3 + 0.3 * v) * (Tb - air_c)
    return Ts + dTs / C_S * dt, Tb + dTb / C_B * dt


if __name__ == "__main__":
    a = np.linspace(0, 0.3, 301)
    f = lateral(a, 5000.0, 1.0)
    print("peak slip deg", np.degrees(a[f.argmax()]).round(1), "peak/Fz", (f.max() / 5000).round(3))
    assert 4 < np.degrees(a[f.argmax()]) < 10 and abs(f.max() / 5000 - 1) < 0.01
