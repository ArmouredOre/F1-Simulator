"""Top-down replay of a telemetry log.

python gui.py logs/run.parquet [--track tracks/monza.npz]
Keys: space pause, up/down playback speed, left/right skip 5 s, esc quit.
python gui.py logs/run.parquet --shot out.png --at 30   # headless frame grab

python gui.py --drive monza   # drive a car yourself through RaceEnv
Keys: up throttle, down brake, left/right steer, Z hold straight mode, X hold overtake, R restart, esc quit.
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
    t = npz if isinstance(npz, dict) else np.load(npz)  # a path or an already loaded track dict
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


def fit(lines):
    """Function mapping track metres to screen pixels, scaled to fill the track area."""
    pts = np.vstack(lines)
    lo, hi = pts.min(0), pts.max(0)
    scale = 0.92 * min((W - PANEL) / (hi - lo)[0], H / (hi - lo)[1])
    off = np.array([(W - PANEL), H]) / 2 - scale * (lo + hi) / 2

    def px(xy):
        p = off + scale * np.asarray(xy)
        return np.c_[p[..., 0], H - p[..., 1]] if p.ndim > 1 else (p[0], H - p[1])
    return px


def drive(a):
    """Keyboard-drive one car in RaceEnv: the sim steps at 10 Hz in real time, the window redraws at 60 fps."""
    from env import RaceEnv
    env = RaceEnv(track=a.drive, max_time_s=1e9)
    env.reset(seed=0)
    lines, zones = track_lines(None, env.trk)
    px = fit(lines)
    pygame.display.init(), pygame.font.init()
    screen = pygame.display.set_mode((W, H))
    font = pygame.font.SysFont("monospace", 15)
    clock = pygame.time.Clock()
    steer, frame, white = 0.0, 0, (220, 220, 220)

    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                return
            if e.type == pygame.KEYDOWN and e.key == pygame.K_r:
                env.reset(seed=0), env.rows.clear()
        k = pygame.key.get_pressed()
        steer += np.clip(k[pygame.K_LEFT] - k[pygame.K_RIGHT] - steer, -0.08, 0.08)  # ramp so taps are gentle
        if frame % 6 == 0 and not env.done[0]:
            lock = min(1.0, 340 / env.st["vx"][0] ** 2)  # full lock only at low speed, a real car steers ~0.1 of it at 200 km/h
            env.step({"car_0": [steer * lock, k[pygame.K_UP] - k[pygame.K_DOWN], k[pygame.K_z], k[pygame.K_x], 0]})
        # ponytail: the car jumps 10 times a second, interpolate between steps if it looks choppy

        screen.fill((20, 20, 24))
        for i, line in enumerate(lines):
            pygame.draw.lines(screen, (90, 90, 100) if i else (150, 150, 160), True, px(line).tolist(), 2 if i else 1)
        for z in zones:
            pygame.draw.lines(screen, (60, 200, 110), False, px(z).tolist(), 4)
        y = 10
        if env.rows:
            r = env.rows[-1]
            pygame.draw.circle(screen, COLORS[0], px((r["X"], r["Y"])), 7)
            screen.blit(font.render(f"{a.drive}  lap {int(env.lap[0])}  lap time {env.t - env.lap_start[0]:6.1f}s", True, COLORS[0]), (W - PANEL + 10, y))
            y += 24
            for key in SHOW:
                if key in r:
                    v = r[key]
                    screen.blit(font.render(f"{key:<12}{(f'{v:.2f}' if isinstance(v, float) else str(v)):>10}", True, white), (W - PANEL + 10, y))
                    y += 19
        if env.done[0]:
            screen.blit(font.render("CRASHED - press R to restart", True, (255, 80, 80)), (W - PANEL + 10, y + 10))
        y += 40
        screen.blit(font.render("Lap times (s)", True, white), (W - PANEL + 10, y))
        for n, lt in enumerate(env.lap_times[0][-12:]):
            screen.blit(font.render(f"L{len(env.lap_times[0]) - len(env.lap_times[0][-12:]) + n + 1}  {lt:.3f}", True, white), (W - PANEL + 10, y + 22 + 16 * n))

        frame += 1
        if a.shot and frame == 120:  # headless frame grab after 2 s
            pygame.image.save(screen, a.shot)
            return
        pygame.display.flip()
        clock.tick(60)


def main(a):
    if a.drive:
        return drive(a)
    log = pd.read_parquet(a.log).sort_values("Time")
    drivers = list(dict.fromkeys(log.Driver))
    cars = {d: log[log.Driver == d] for d in drivers}
    lines, zones = track_lines(log, a.track)
    px = fit(lines)
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
    p.add_argument("log", nargs="?", help="telemetry parquet to replay (not needed with --drive)")
    p.add_argument("--drive", metavar="TRACK", help="drive this track yourself, e.g. monza")
    p.add_argument("--track", help="tracks/<name>.npz for edges")
    p.add_argument("--at", type=float, default=0.0, help="start time (s)")
    p.add_argument("--shot", help="save one frame to this png and exit")
    a = p.parse_args()
    if not (a.log or a.drive):
        p.error("give a log to replay, or --drive TRACK")
    main(a)
