"""Top-down 2D lap viewer: follow camera, kerbs and run-off, lap and sector times, optional ghost lap.

python gui.py logs/rl_monza_2026.parquet --track tracks/monza.npz [--ghost logs/gate_monza_2026.parquet]
Keys: space pause, up/down playback speed, left/right skip 5 s, tab switch car.
python gui.py logs/run.parquet --track tracks/monza.npz --shot out.png --at 30   # headless frame grab

python gui.py --drive monza [--ghost logs/rl_monza_2026.parquet]   # drive yourself, race the ghost
Keys: up throttle, down brake, left/right steer, Z hold straight mode, X hold overtake, R restart.
Your laps are saved to logs/player_<track>_<time>.parquet on R or esc.
Weather (live): 1/2 air temp, 3/4 track temp, 5/6 rain, 7/8 wind (down/up).
Both modes: mouse wheel or -/= zoom, C full-track view, esc quit.
"""
import argparse
import time
import numpy as np
import pandas as pd
import pygame

W, H = 1280, 800
COLORS = [(230, 40, 40), (40, 160, 255), (250, 200, 40), (60, 220, 120), (200, 90, 255), (255, 140, 30)]
GRASS, RUNOFF, ASPHALT, WHITE = (38, 92, 48), (150, 140, 120), (62, 62, 68), (235, 235, 235)
CAR = np.array([(2.9, 0), (2.7, .9), (2.3, .9), (2.3, .3), (.5, .4), (.3, .95), (-1.2, .95), (-1.4, .5),
                (-2.3, .5), (-2.4, .9), (-2.9, .9)])  # half an F1 outline in metres, x forward
CAR = np.r_[CAR, CAR[::-1] * [1, -1]]
WX = {pygame.K_1: ("air_c", -2), pygame.K_2: ("air_c", 2), pygame.K_3: ("track_c", -2), pygame.K_4: ("track_c", 2),
      pygame.K_5: ("rain_mm_h", -0.5), pygame.K_6: ("rain_mm_h", 0.5), pygame.K_7: ("wind_ms", -1), pygame.K_8: ("wind_ms", 1)}


class Track:
    """Track geometry in metres from a tracks/*.npz (path or loaded dict)."""

    def __init__(self, npz):
        t = npz if isinstance(npz, dict) else np.load(npz)
        self.c, self.L, s = np.c_[t["x"], t["y"]], float(t["s"][-1]), t["s"]
        tan = np.gradient(self.c, axis=0)
        n = np.c_[-tan[:, 1], tan[:, 0]] / np.linalg.norm(tan, axis=1)[:, None]  # left normal
        wl, wr = t["w_left"][:, None], t["w_right"][:, None]
        self.width = float(np.mean(wl + wr))
        # ponytail: one road width per track (thick polyline); per-point widths need quad strips
        kerb = np.abs(t["kappa"]) > 0.006  # kerbs only in corners
        self.kerbs = []  # (colour, edge points) in ~5 m stripes
        for edge in (self.c + n * wl, self.c - n * wr):
            for i in range(0, len(s) - 6, 5):
                if kerb[i]:
                    self.kerbs.append(((220, 30, 30) if (i // 5) % 2 else WHITE, edge[i:i + 6]))
        self.zones = [self.c[np.r_[np.flatnonzero(s >= a), np.flatnonzero(s <= b)] if b < a else (s >= a) & (s <= b)]
                      for a, b in t["zones"]]  # straight-mode zones, end < start wraps the line
        self.start = (self.c[0] + n[0] * wl[0], self.c[0] - n[0] * wr[0])
        lo, hi = self.c.min(0), self.c.max(0)
        self.mid, self.fit = (lo + hi) / 2, 0.9 * min(W / (hi - lo)[0], H / (hi - lo)[1])


class Cam:
    def __init__(self, trk):
        self.trk, self.zoom, self.overview, self.at = trk, 6.0, False, trk.mid

    def scale(self):
        return self.trk.fit if self.overview else self.zoom

    def px(self, xy):
        """World metres -> screen pixels (works on one point or an (n, 2) array)."""
        c = self.trk.mid if self.overview else self.at
        p = (np.asarray(xy) - c) * self.scale()
        return np.c_[W / 2 + p[..., 0], H / 2 - p[..., 1]] if p.ndim > 1 else (W / 2 + p[0], H / 2 - p[1])

    def key(self, e):
        if e.type == pygame.MOUSEWHEEL:
            self.zoom = np.clip(self.zoom * 1.15 ** e.y, 0.5, 20)
        elif e.type == pygame.KEYDOWN:
            self.zoom = np.clip(self.zoom * {pygame.K_EQUALS: 1.25, pygame.K_MINUS: 0.8}.get(e.key, 1), 0.5, 20)
            self.overview ^= e.key == pygame.K_c


def draw_track(screen, cam):
    t, s = cam.trk, cam.scale()
    screen.fill(GRASS)
    c = cam.px(t.c)
    vis = c[::2][(np.abs(c[::2] - (W / 2, H / 2)) < (W, H)).all(1)]  # on-screen points, to round off line joints
    for col, w in ((RUNOFF, t.width + 16), (ASPHALT, t.width)):
        pygame.draw.lines(screen, col, True, c.tolist(), max(1, int(w * s)))
        for q in vis if s > 1 else ():
            pygame.draw.circle(screen, col, q, w * s / 2)
    for col, pts in t.kerbs:
        pygame.draw.lines(screen, col, False, cam.px(pts).tolist(), max(1, int(1.2 * s)))
    for z in t.zones:  # straight-mode zones as a thin green centre stripe
        pygame.draw.lines(screen, (60, 200, 110), False, cam.px(z).tolist(), max(1, int(0.5 * s)))
    pygame.draw.line(screen, WHITE, cam.px(t.start[0]), cam.px(t.start[1]), max(2, int(1.5 * s)))


def draw_car(screen, cam, xy, hdg, col, ghost=False):
    rot = np.array([[np.cos(hdg), np.sin(hdg)], [-np.sin(hdg), np.cos(hdg)]])
    pts = cam.px(CAR @ rot + xy).tolist()
    if ghost:
        pygame.draw.polygon(screen, (200, 200, 255), pts, 2)
    else:
        pygame.draw.polygon(screen, col, pts)
        pygame.draw.polygon(screen, (10, 10, 10), pts, 1)
    if cam.scale() < 2:  # too small to see the outline
        pygame.draw.circle(screen, (200, 200, 255) if ghost else col, cam.px(xy), 5, 2 if ghost else 0)


def minimap(trk):
    """Small pre-drawn track outline and a function placing a point on it."""
    surf = pygame.Surface((220, 220), pygame.SRCALPHA)
    surf.fill((0, 0, 0, 140))
    lo, hi = trk.c.min(0), trk.c.max(0)
    k = 200 / max(hi - lo)

    def mp(xy):
        p = (np.asarray(xy) - (lo + hi) / 2) * k
        return np.c_[110 + p[..., 0], 110 - p[..., 1]] if p.ndim > 1 else (W - 230 + 110 + p[0], 10 + 110 - p[1])
    pygame.draw.lines(surf, WHITE, True, mp(trk.c).tolist(), 2)
    return surf, mp


def heading(x, y):
    return np.unwrap(np.arctan2(np.gradient(y), np.gradient(x)))


class Lap:
    """One recorded lap as arrays against lap time: used for the ghost and for sector splits."""

    def __init__(self, d):
        self.t = d.Time.to_numpy() - d.Time.iloc[0]
        self.x, self.y, self.dist = d.X.to_numpy(), d.Y.to_numpy(), d.Distance.to_numpy()
        self.h = heading(self.x, self.y)

    def at(self, tl):
        return np.array([np.interp(tl, self.t, self.x), np.interp(tl, self.t, self.y)]), np.interp(tl, self.t, self.h)

    def time_at(self, dist):
        return np.interp(dist, self.dist, self.t)

    def splits(self, cuts):
        return np.diff(np.r_[0, self.time_at(cuts), self.t[-1]])


def laps_of(log, L):
    """{driver: {lap number: Lap}} for complete laps (covered almost the full distance)."""
    out = {}
    for (drv, n), d in log.groupby(["Driver", "LapNumber"]):
        if len(d) > 10 and d.Distance.min() < 0.05 * L and d.Distance.max() > 0.95 * L:
            out.setdefault(drv, {})[n] = Lap(d)
    return out


def best_lap(path, L):
    laps = [lp for v in laps_of(pd.read_parquet(path).sort_values("Time"), L).values() for lp in v.values()]
    return min(laps, key=lambda lp: lp.t[-1]) if laps else None


def text(screen, font, s, xy, col=WHITE):
    screen.blit(font.render(s, True, col), xy)


def hud(screen, fonts, lines, x=10, y=10):
    """Panel of (text, colour) lines on a translucent box."""
    box = pygame.Surface((300, 12 + 22 * len(lines)), pygame.SRCALPHA)
    box.fill((0, 0, 0, 150))
    screen.blit(box, (x, y))
    for i, (s, col) in enumerate(lines):
        text(screen, fonts[0], s, (x + 10, y + 6 + 22 * i), col)


def dash(screen, fonts, kmh, gear, thr, brk):
    """Speed, gear and pedal bars, bottom left."""
    box = pygame.Surface((220, 90), pygame.SRCALPHA)
    box.fill((0, 0, 0, 150))
    screen.blit(box, (10, H - 100))
    text(screen, fonts[1], f"{kmh:3.0f}", (20, H - 96))
    text(screen, fonts[0], f"km/h   gear {gear}", (20, H - 50))
    pygame.draw.rect(screen, (60, 220, 90), (130, H - 90, 80 * thr, 12))
    pygame.draw.rect(screen, (230, 50, 50), (130, H - 72, 80 * brk, 12))


def fmt(t):
    return f"{int(t // 60)}:{t % 60:06.3f}"


def delta_line(lap_t, dist, ghost):
    if not ghost:
        return []
    d = lap_t - ghost.time_at(dist)
    return [(f"vs ghost {d:+.2f}s", (255, 90, 90) if d > 0 else (90, 255, 120))]


def setup(trk):
    pygame.display.init(), pygame.font.init()
    screen = pygame.display.set_mode((W, H))
    fonts = pygame.font.SysFont("monospace", 17, bold=True), pygame.font.SysFont("monospace", 40, bold=True)
    return screen, fonts, Cam(trk), minimap(trk), pygame.time.Clock()


def drive(a):
    """Keyboard-drive one car in RaceEnv: the sim steps at 10 Hz in real time, the window redraws at 60 fps."""
    from env import RaceEnv
    from physics import air_density
    env = RaceEnv(track=a.drive, max_time_s=1e9)
    env.reset(seed=0)
    trk = Track(env.trk)
    ghost = best_lap(a.ghost, trk.L) if a.ghost else None
    screen, fonts, cam, (mm, mp), clock = setup(trk)
    steer, frame, prev = 0.0, 0, None
    # keep the player's laps for the rival's player clone (#11), one file per run
    keep = lambda: env.lap_times[0] and print("saved", env.save_log(f"player_{a.drive}_{time.strftime('%Y%m%d_%H%M%S')}"))

    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                return keep()
            cam.key(e)
            if e.type == pygame.KEYDOWN and e.key == pygame.K_r:
                keep(), env.reset(seed=0), env.rows.clear()
            if e.type == pygame.KEYDOWN and e.key in WX:  # weather keys, applied live
                key, d = WX[e.key]
                env.cond[key] = max(0, env.cond[key] + d)
                env.rho = air_density(env.cond)
        k = pygame.key.get_pressed()
        steer += np.clip(k[pygame.K_LEFT] - k[pygame.K_RIGHT] - steer, -0.08, 0.08)  # ramp so taps are gentle
        if frame % 6 == 0 and not env.done[0]:
            prev = env.rows[-1] if env.rows else None
            lock = min(1.0, 340 / env.st["vx"][0] ** 2)  # full lock only at low speed, a real car steers ~0.1 of it at 200 km/h
            env.step({"car_0": [steer * lock, k[pygame.K_UP] - k[pygame.K_DOWN], k[pygame.K_z], k[pygame.K_x], 0]})
        # ponytail: the car jumps 10 times a second, interpolate between steps if it looks choppy

        r = env.rows[-1]
        xy = np.array([r["X"], r["Y"]])
        hdg = np.arctan2(r["Y"] - prev["Y"], r["X"] - prev["X"]) if prev else 0.0
        cam.at = xy
        lap_t = env.t - env.lap_start[0]
        draw_track(screen, cam)
        if ghost and lap_t < ghost.t[-1]:
            draw_car(screen, cam, *ghost.at(lap_t), None, ghost=True)
        draw_car(screen, cam, xy, hdg, COLORS[0])
        lt = env.lap_times[0]
        lines = [(f"{a.drive.upper()}  lap {int(env.lap[0])}", COLORS[0]), (f"time  {fmt(lap_t)}", WHITE)]
        lines += delta_line(lap_t, r["Distance"], ghost) if "Distance" in r else []
        lines += [(f"last  {fmt(lt[-1])}", WHITE), (f"best  {fmt(min(lt))}", (200, 120, 255))] if lt else []
        lines += [(f"air {env.cond['air_c']:.0f}C  track {env.cond['track_c']:.0f}C", (120, 180, 255)),
                  (f"rain {env.cond['rain_mm_h']:.1f}mm/h  wind {env.cond['wind_ms']:.0f}m/s", (120, 180, 255))]
        if env.st["dmg"][0] > 0:
            lines.append((f"damage {min(env.st['dmg'][0], 1):.0%}", (255, 160, 60)))
        if env.done[0]:
            lines.append(("CRASHED - press R", (255, 80, 80)))
        hud(screen, fonts, lines)
        dash(screen, fonts, r["Speed"], r["nGear"], r["Throttle"] / 100, float(r["Brake"]))
        screen.blit(mm, (W - 230, 10))
        pygame.draw.circle(screen, COLORS[0], mp(xy), 5)

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
    trk = Track(a.track)
    drivers = list(dict.fromkeys(log.Driver))
    cars = {d: log[log.Driver == d] for d in drivers}
    hdgs = {d: heading(c.X.to_numpy(), c.Y.to_numpy()) for d, c in cars.items()}
    laps = laps_of(log, trk.L)
    ghost = best_lap(a.ghost, trk.L) if a.ghost else None
    screen, fonts, cam, (mm, mp), clock = setup(trk)
    t, t_end, rate, paused, follow = a.at, log.Time.max(), 1.0, False, 0

    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                return
            cam.key(e)
            if e.type == pygame.KEYDOWN:
                paused ^= e.key == pygame.K_SPACE
                rate *= {pygame.K_UP: 2, pygame.K_DOWN: 0.5}.get(e.key, 1)
                t += {pygame.K_RIGHT: 5, pygame.K_LEFT: -5}.get(e.key, 0)
                follow = (follow + (e.key == pygame.K_TAB)) % len(drivers)

        idx = {d: min(np.searchsorted(c.Time.to_numpy(), t), len(c) - 1) for d, c in cars.items()}
        rows = {d: cars[d].iloc[i] for d, i in idx.items()}
        d = drivers[follow]
        r = rows[d]
        cam.at = np.array([r.X, r.Y])
        c = cars[d]
        lap_t = t - c.Time[c.LapNumber == r.LapNumber].min()
        draw_track(screen, cam)
        if ghost and lap_t < ghost.t[-1]:
            draw_car(screen, cam, *ghost.at(lap_t), None, ghost=True)
        for i, (drv, row) in enumerate(rows.items()):
            draw_car(screen, cam, np.array([row.X, row.Y]), hdgs[drv][idx[drv]], COLORS[i % len(COLORS)])

        # sectors = thirds of the lap; colour green when it matches this car's best, else yellow
        mine = laps.get(d, {})
        done = {n: lp for n, lp in mine.items() if n < r.LapNumber}
        cuts = np.array([trk.L / 3, 2 * trk.L / 3])
        best = np.min([lp.splits(cuts) for lp in done.values()], 0) if done else None
        col = COLORS[follow % len(COLORS)]
        lines = [(f"{d}  lap {int(r.LapNumber)}   x{rate:g}{'  PAUSED' if paused else ''}", col), (f"time  {fmt(lap_t)}", WHITE)]
        lines += delta_line(lap_t, r.Distance, ghost)
        if r.LapNumber in mine:  # sector splits so far this lap
            sp = mine[r.LapNumber].splits(cuts)
            for j in range(3):
                if r.Distance > (cuts[j] if j < 2 else 0.99 * trk.L):
                    good = best is None or sp[j] <= best[j] + 1e-6
                    lines.append((f"S{j + 1}    {sp[j]:6.3f}", (90, 255, 120) if good else (250, 200, 40)))
        lap_list = [lp.t[-1] for n, lp in sorted(done.items())]
        lines += [(f"last  {fmt(lap_list[-1])}", WHITE), (f"best  {fmt(min(lap_list))}", (200, 120, 255))] if lap_list else []
        lines += [(f"ghost {fmt(ghost.t[-1])}", (200, 200, 255))] if ghost else []
        hud(screen, fonts, lines)
        dash(screen, fonts, r.Speed, r.nGear, r.Throttle / 100, float(r.Brake))
        screen.blit(mm, (W - 230, 10))
        for i, row in enumerate(rows.values()):
            pygame.draw.circle(screen, COLORS[i % len(COLORS)], mp((row.X, row.Y)), 5)

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
    p.add_argument("--track", help="tracks/<name>.npz (required for replay)")
    p.add_argument("--ghost", help="parquet log whose best lap is shown as a ghost car")
    p.add_argument("--at", type=float, default=0.0, help="start time (s)")
    p.add_argument("--shot", help="save one frame to this png and exit")
    a = p.parse_args()
    if not (a.log or a.drive):
        p.error("give a log to replay, or --drive TRACK")
    if a.log and not a.drive and not a.track:
        p.error("replay needs --track tracks/<name>.npz")
    main(a)
