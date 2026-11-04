"""Top-down replay of a telemetry log.

python gui.py logs/run.parquet [--track tracks/monza.npz]
Keys: space pause, up/down playback speed, left/right skip 5 s, esc quit.
python gui.py logs/run.parquet --shot out.png --at 30   # headless frame grab
"""
import argparse
import numpy as np
import pandas as pd
import pygame

W, H, PANEL = 1280, 800, 360
COLORS = [(230, 40, 40), (40, 160, 255), (250, 200, 40), (60, 220, 120), (200, 90, 255), (255, 140, 30)]
SHOW = ["Speed", "nGear", "Throttle", "Brake", "AeroMode", "Overtake", "Battery_MJ",
        "TyreTemp_FL", "TyreTemp_FR", "TyreTemp_RL", "TyreTemp_RR",
        "TyreWear_FL", "TyreWear_FR", "TyreWear_RL", "TyreWear_RR",
        "AirTemp", "TrackTemp", "Rainfall", "WindSpeed"]  # shown when present in the log


def track_lines(log, npz):
    """(centreline + edges, straight-mode zone paths) from a track npz, or just the first car's path."""
    if not npz:
        d = log[log.Driver == log.Driver.iloc[0]]
        return [np.c_[d.X, d.Y]], []
    t = np.load(npz)
    c, s = np.c_[t["x"], t["y"]], t["s"]
    tan = np.gradient(c, axis=0)
    n = np.c_[-tan[:, 1], tan[:, 0]] / np.linalg.norm(tan, axis=1)[:, None]  # left normal
    # a zone with end < start wraps over the start/finish line
    zones = [c[np.r_[np.flatnonzero(s >= a), np.flatnonzero(s <= b)] if b < a else (s >= a) & (s <= b)]
             for a, b in t["zones"]]
    return [c, c + n * t["w_left"][:, None], c - n * t["w_right"][:, None]], zones


def lap_times(log):
    """{driver: [lap seconds]} for laps that look complete (not the first partial or last partial)."""
    g = log.groupby(["Driver", "LapNumber"]).Time.agg(["min", "max"])
    out = {}
    for (drv, _), r in g.iterrows():
        out.setdefault(drv, []).append(r["max"] - r["min"])
    return out


def main(a):
    log = pd.read_parquet(a.log).sort_values("Time")
    drivers = list(dict.fromkeys(log.Driver))
    cars = {d: log[log.Driver == d] for d in drivers}
    lines, zones = track_lines(log, a.track)
    pts = np.vstack(lines)
    lo, hi = pts.min(0), pts.max(0)
    scale = 0.92 * min((W - PANEL) / (hi - lo)[0], H / (hi - lo)[1])
    off = np.array([(W - PANEL), H]) / 2 - scale * (lo + hi) / 2

    def px(xy):
        p = off + scale * np.asarray(xy)
        return np.c_[p[..., 0], H - p[..., 1]] if p.ndim > 1 else (p[0], H - p[1])

    laps = lap_times(log)
    pygame.display.init(), pygame.font.init()
    screen = pygame.display.set_mode((W, H))
    font = pygame.font.SysFont("monospace", 15)
    clock = pygame.time.Clock()
    t, t_end, rate, paused = a.at, log.Time.max(), 1.0, False

    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                return
            if e.type == pygame.KEYDOWN:
                paused ^= e.key == pygame.K_SPACE
                rate *= {pygame.K_UP: 2, pygame.K_DOWN: 0.5}.get(e.key, 1)
                t += {pygame.K_RIGHT: 5, pygame.K_LEFT: -5}.get(e.key, 0)

        screen.fill((20, 20, 24))
        for i, line in enumerate(lines):
            pygame.draw.lines(screen, (90, 90, 100) if i else (150, 150, 160), True, px(line).tolist(), 2 if i else 1)
        for z in zones:  # straight-mode aero zones
            pygame.draw.lines(screen, (60, 200, 110), False, px(z).tolist(), 4)

        y = 10
        for i, d in enumerate(drivers):
            c = cars[d]
            r = c.iloc[min(np.searchsorted(c.Time.to_numpy(), t), len(c) - 1)]
            col = COLORS[i % len(COLORS)]
            pygame.draw.circle(screen, col, px((r.X, r.Y)), 7)
            screen.blit(font.render(str(d), True, col), np.add(px((r.X, r.Y)), 9))
            if i == 0:  # readouts for the lead car of the log
                screen.blit(font.render(f"{d}  lap {int(r.LapNumber)}  t={t:7.1f}s  x{rate:g}", True, col), (W - PANEL + 10, y))
                y += 24
                for k in SHOW:
                    if k in c:
                        v = r[k]
                        txt = f"{v:.2f}" if isinstance(v, float) else str(v)
                        screen.blit(font.render(f"{k:<12}{txt:>10}", True, (220, 220, 220)), (W - PANEL + 10, y))
                        y += 19

        # lap-time chart: one bar per lap, per driver
        y += 15
        screen.blit(font.render("Lap times (s)", True, (220, 220, 220)), (W - PANEL + 10, y))
        y += 22
        allt = [x for v in laps.values() for x in v]
        tmin, tmax = min(allt), max(allt)
        for i, d in enumerate(drivers):
            for n, lt in enumerate(laps[d][:12]):
                w = 20 + 120 * (lt - tmin) / (tmax - tmin + 1e-9)
                pygame.draw.rect(screen, COLORS[i % len(COLORS)], (W - PANEL + 10, y, w, 12))
                screen.blit(font.render(f"{d} L{n + 1} {lt:.3f}", True, (220, 220, 220)), (W - PANEL + 20 + w, y - 2))
                y += 16
                if y > H - 20:
                    break

        if a.shot:
            pygame.image.save(screen, a.shot)
            return
        pygame.display.flip()
        dt = clock.tick(60) / 1000
        if not paused:
            t = (t + dt * rate) % t_end


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("log")
    p.add_argument("--track", help="tracks/<name>.npz for edges")
    p.add_argument("--at", type=float, default=0.0, help="start time (s)")
    p.add_argument("--shot", help="save one frame to this png and exit")
    main(p.parse_args())
