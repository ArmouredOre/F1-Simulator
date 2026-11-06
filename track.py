"""Build tracks/<name>.npz (contract format) from the TUM racetrack database or FastF1 X/Y."""
import io, sys, urllib.request
from pathlib import Path
import numpy as np
from scipy.interpolate import splprep, splev

TRACKS = Path(__file__).parent / "tracks"
TUM = "https://raw.githubusercontent.com/TUMFTM/racetrack-database/master/tracks/{}.csv"
TUM_NAMES = {"monza": "Monza", "silverstone": "Silverstone"}  # Monaco isn't in TUM, so it comes from FastF1


def from_tum(name):
    raw = urllib.request.urlopen(TUM.format(TUM_NAMES[name])).read().decode()
    x, y, wr, wl = np.loadtxt(io.StringIO(raw), delimiter=",", comments="#").T
    return x, y, wl, wr, len(x) * 0.25


GP = {"spa": "Belgium"}  # FastF1 fuzzy search needs 4+ letters
FIGURE_EIGHT = {"suzuka"}  # crosses itself, so total turning is 0 rather than 2 pi


def from_fastf1(name, year=2024, half_width=4.0):
    # ponytail: pole lap's driven line stands in for the centreline, constant width; swap for a surveyed map if found
    from data import fastest_lap
    _, tel, _ = fastest_lap(year, GP.get(name, name))
    x, y = tel.X.values, tel.Y.values
    keep = np.r_[True, np.hypot(np.diff(x), np.diff(y)) > 0.5]
    x, y = x[keep], y[keep]
    w = np.full(len(x), half_width)
    return x, y, w, w, len(x) * 1.0


def build(x, y, wl, wr, smooth):
    tck, _ = splprep([x, y, wl, wr], s=smooth, per=1)
    uf = np.linspace(0, 1, 50000, endpoint=False)
    xf, yf = splev(uf, tck)[:2]
    ds = np.hypot(np.diff(xf, append=xf[0]), np.diff(yf, append=yf[0]))
    L = ds.sum()
    n = int(round(L))
    s = np.arange(n) * L / n
    u = np.interp(s, np.r_[0, np.cumsum(ds)[:-1]], uf)
    x, y, wl, wr = splev(u, tck)
    dx, dy = splev(u, tck, der=1)[:2]
    ddx, ddy = splev(u, tck, der=2)[:2]
    kappa = (dx * ddy - dy * ddx) / (dx * dx + dy * dy) ** 1.5
    return dict(s=s, x=x, y=y, w_left=wl, w_right=wr, kappa=kappa, zones=zones(s, kappa))


def zones(s, kappa, min_len=300):
    """Rows of (start, end) in m; end < start means the zone wraps over the line."""
    # ponytail: radius > 1 km for 300 m+ counts as a straight; swap for FIA-published zones
    st = np.abs(kappa) < 1e-3
    k = np.argmin(st)  # a non-straight index, so no run is split by the roll
    d = np.diff(np.r_[0, np.roll(st, -k).astype(int), 0])
    a, b = np.where(d == 1)[0], np.where(d == -1)[0]
    keep = (b - a) * (s[1] - s[0]) >= min_len
    return np.c_[s[(a[keep] + k) % len(s)], s[(b[keep] - 1 + k) % len(s)]]


def make(name):
    t = build(*(from_tum(name) if name in TUM_NAMES else from_fastf1(name)))
    turn = abs(np.sum(t["kappa"]) * (t["s"][1] - t["s"][0]))
    assert abs(turn - (0 if name in FIGURE_EIGHT else 2 * np.pi)) < 0.1, "not one closed loop"
    TRACKS.mkdir(exist_ok=True)
    np.savez(TRACKS / f"{name}.npz", **t)
    print(f"{name}: {t['s'][-1]:.0f} m, {len(t['zones'])} straight zones")


if __name__ == "__main__":
    for name in sys.argv[1:] or ["monza", "silverstone", "monaco", "spa", "suzuka", "bahrain"]:
        make(name)
