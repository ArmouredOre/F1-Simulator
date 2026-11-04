"""Season rules: load seasons/<year>.yaml and gate what drivers ask for."""
import pathlib
import numpy as np
import yaml

HERE = pathlib.Path(__file__).parent


def load(season, wing=1.0):
    """Season rules. wing scales downforce (drag by half as much): a set-up choice, 1.0 = medium trim."""
    R = yaml.safe_load((HERE / "seasons" / f"{season}.yaml").read_text())
    for mode in ("corner", "straight"):
        R["aero"][mode]["cl_a"] *= wing
        R["aero"][mode]["cd_a"] *= (1 + wing) / 2
    return R


def in_zones(s, zones):
    """True where distance s lies in any [start, end] zone (zones may wrap past the line)."""
    s = np.asarray(s, float)[:, None]
    z = np.asarray(zones, float).reshape(-1, 2)
    a, b = z[:, 0], z[:, 1]
    return np.where(a <= b, (s >= a) & (s <= b), (s >= a) | (s <= b)).any(1)


def straight_mode(R, press, s, zones, gap_ahead, braking):
    ok = in_zones(s, zones) & ~braking
    if R["rules"]["drs"]:
        ok &= gap_ahead <= R["rules"]["drs"]["gap_s"]
    return press & ok


def overtake(R, press, gap_ahead):
    ot = R["rules"]["overtake"]
    return press & (gap_ahead <= ot["gap_s"]) if ot else np.zeros_like(press)


def _taper(kmh, band):
    return np.ones_like(kmh) if band is None else np.clip((band[1] - kmh) / (band[1] - band[0]), 0, 1)


def mguk(R, v, bias, ot_active, st):
    """MGU-K power on throttle (kW, + deploy / - harvest) and max regen under braking (kW) per car."""
    pu, ot = R["pu"], R["rules"]["overtake"]
    kmh = v * 3.6
    t = _taper(kmh, pu["taper_kmh"])
    if ot:
        t = np.where(ot_active, _taper(kmh, ot["taper_kmh"]), t)
    can_deploy = (st["batt"] > 0) & (st["dep_lap"] < (pu["deploy_mj_lap"] or np.inf))
    cap = pu["harvest_mj_lap"] + (ot["extra_mj"] * st["ot_lap"] if ot else 0)
    can_harvest = (st["batt"] < pu["battery_mj"]) & (st["harv_lap"] < cap)
    # bias > 0 deploys that share of MGU-K power; bias < 0 harvests on throttle (costs ICE power)
    k = pu["mguk_kw"] * np.where(bias > 0, bias * t * can_deploy, bias * can_harvest)
    return k, pu["mguk_kw"] * can_harvest
