"""3D game server: runs RaceEnv for the Godot client (game/) over a local TCP socket, one JSON object per line.
python server.py [track] [--rain MM_H] [--no-ai] [--pro]     then press Play in Godot.
Beginner assists are on unless --pro: left/right move the car across the track and the test driver does the wheel
work, it brakes for you when you are too fast for the corner, there is no damage, and a spin puts you back on track.
Godot sends the player's controls 50 times a second; each message steps the sim once and gets the new state back.
The player is car_0, the AI car (scripted test driver) is car_1 and starts 40 m ahead.
Your laps are saved to logs/player_<track>_<time>.parquet on restart or quit, like gui.py --drive."""
import argparse
import json
import socket
import time
import numpy as np
import physics
import test_env
from env import RaceEnv, DT

EDGE = 2.5                 # m: beginner assist keeps the car's centre this far inside the white lines
PORT, SUBSTEPS = 5555, 2   # 2 physics steps per message = 50 Hz, so the controls feel direct


def guide(line_n, v_t, ds):
    """Beginner guide per metre: the racing line's offset and what to do there (0 throttle, 1 lift, 2 brake),
    from how the test driver's target speed changes. Braking is shown 20 m early to allow for reaction time."""
    acc = v_t * np.gradient(np.r_[v_t[-1], v_t, v_t[0]])[1:-1] / ds
    return line_n, np.maximum(np.where(acc < -0.5, 1, 0), 2 * np.roll(acc < -4, -int(20 / ds)))


def track_msg(env, name, plan, assist=True, every=2):
    """Track edges and the guide line for the client to build the meshes from, one point every 2 m."""
    t = env.trk
    r = lambda k: np.round(t[k][::every], 2).tolist()
    line, pace = guide(*plan, t["ds"])
    return dict(track=name, x=r("x"), y=r("y"), wl=r("w_left"), wr=r("w_right"), wall=physics.WALL, assist=assist,
                line=np.round(line[::every], 2).tolist(), pace=pace[::every].tolist(),
                kerb=(np.abs(t["kappa"][::every]) > 0.006).astype(int).tolist(), length=float(t["length"]))


def state_msg(env, v_t):
    """Car poses (x, y, heading, front wheel angle) and the player's dashboard, from the last log rows.
    target = the guide's speed 0.3 s up the road, so the client can say brake or go."""
    st, t = env.st, env.trk
    rows = env.rows[-env.n:]
    hdg = np.arctan2(physics.at(t, "ty", st["s"]), physics.at(t, "tx", st["s"])) + st["xi"]
    r, lt = rows[0], env.lap_times[0]
    w = t["w_left" if st["n"][0] > 0 else "w_right"][int(st["s"][0] / t["ds"]) % len(t["s"])]
    return dict(cars=[[round(q["X"], 3), round(q["Y"], 3), round(float(h), 4), round(q["Steer"], 3)] for q, h in zip(rows, hdg)],
                kmh=round(r["Speed"], 1), gear=r["nGear"], rpm=round(r["RPM"]), thr=r["Throttle"] / 100, brk=float(st["brk"][0]),
                lap=int(env.lap[0]), t=round(env.t - env.lap_start[0], 3), last=lt[-1] if lt else 0, best=min(lt) if lt else 0,
                dmg=round(min(float(st["dmg"][0]), 1), 3), done=bool(env.done[0]), drs=r["AeroMode"], ot=bool(r["Overtake"]),
                batt=round(float(st["batt"][0] / env.R["pu"]["battery_mj"]), 3), off=bool(abs(st["n"][0]) > w + 1.0),
                gap=round(float(env.gap[0]), 2) if np.isfinite(env.gap[0]) else 0, rain=env.cond["rain_mm_h"],
                target=round(float(np.interp((st["s"][0] + 0.3 * st["vx"][0]) % t["length"], t["s"], v_t)) * 3.6))


class Game:
    def __init__(self, track, ai=True, rain=0.0, assist=True):
        self.track, self.ai, self.assist = track, ai, assist
        self.env = RaceEnv(track=track, n_cars=1 + ai, conditions=dict(rain_mm_h=rain), max_time_s=1e9)
        self.env.substeps = SUBSTEPS
        self.reset()
        self.plan = test_env.plan(self.env, track)   # AI's line and speeds, also the player's guide; the QP takes a few seconds

    def reset(self):
        env = self.env
        if getattr(env, "lap_times", None) and env.lap_times[0]:
            print("saved", env.save_log(f"player_{self.track}_{time.strftime('%Y%m%d_%H%M%S')}"))
        env.reset(seed=0, v0=25.0 if self.assist else 50.0)   # beginners roll off at 90 km/h, not 180
        self.rescues = 0
        self.off = 0.0   # beginner: how far the player has moved off the racing line, m (+left)
        env.st["n"][:] = 0
        if env.n > 1:
            env.st["s"][1] = 40.0   # AI ahead of the player, so it never rams you from behind at the start
            env.lap[1] = 1
        env._gaps()
        env._log(np.zeros((env.n, 5)), np.zeros(env.n, bool), np.zeros(env.n, bool))   # a row to draw before the first step

    def rescue(self, i):
        """Put car i back on the racing line, pointing straight, at 100 km/h at most, with no damage."""
        env, (line, v_t) = self.env, self.plan
        st, t = env.st, env.trk
        st["n"][i] = np.interp(st["s"][i], t["s"], line)
        st["xi"][i] = st["vy"][i] = st["r"][i] = st["dmg"][i] = 0
        st["vx"][i] = min(st["vx"][i], 28.0, 0.8 * np.interp(st["s"][i], t["s"], v_t))
        env.done[i] = False
        self.rescues += i == 0
        self.off *= i != 0

    def step(self, c):
        """c: steer -1..1 (+left, already cut for speed), x (the raw stick/keys), tb -1..1 (throttle - brake),
        drs, ot, reset, rescue."""
        env, (line, v_t) = self.env, self.plan
        if c.get("reset"):
            self.reset()
        elif self.assist or not env.done[0]:
            if c.get("rescue"):
                self.rescue(0)
            acts = test_env.drive(env, line, v_t, look=0.5) if self.ai else {}   # 0.5 s lookahead: 0.9 crashes at 50 Hz
            steer, tb, x = c.get("steer", 0), c.get("tb", 0), c.get("x", 0)
            if self.assist:
                # left/right slide the player's own line across the track at 5 m/s; let go and it drifts back to the racing line
                t, dt = env.trk, DT * SUBSTEPS
                self.off = float(np.clip(self.off + (5 * x if x else -0.5 * self.off) * dt, -10, 10))
                mine = np.clip(line + self.off, -t["w_right"] + EDGE, t["w_left"] - EDGE)
                steer, safe_tb = test_env.drive(env, mine, 0.92 * v_t, look=0.5)["car_0"][:2]   # 8% slower than the AI: room for weaving
                hardest = test_env.drive(env, mine, 1e-3 * v_t, look=0.5)["car_0"][1]   # the driver asked to stop: its safest hard brake
                tb = max(min(tb, safe_tb), hardest)   # no more throttle or brake than grip allows, and it brakes when you don't
            # ponytail: battery deploys by itself whenever there is charge; add an ERS mode button if wanted
            acts["car_0"] = [steer, tb, c.get("drs", 0), c.get("ot", 0), float(env.st["batt"][0] > 0.5)]
            env.step(acts)
            if self.assist:
                env.st["dmg"][:] = 0
                for i in np.where(env.done)[0]:   # spun: back on track instead of race over (the AI car too)
                    self.rescue(i)
        return state_msg(env, v_t)


def serve(game):
    srv = socket.create_server(("127.0.0.1", PORT))
    print(f"{game.track}: waiting for Godot on port {PORT} (ctrl+C quits)")
    while True:
        conn, _ = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print("Godot connected")
        f = conn.makefile("rw")
        send = lambda m: (f.write(json.dumps(m) + "\n"), f.flush())
        try:
            game.reset()
            send(track_msg(game.env, game.track, game.plan, game.assist))
            for line in f:
                send(game.step(json.loads(line)))
        except (ConnectionError, json.JSONDecodeError) as e:
            print("client error:", e)
        print("Godot disconnected")
        game.reset()   # saves the laps


def demo():
    """Self-check: a clumsy beginner still laps Monza, pro mode is the raw car, and the messages are valid JSON."""
    g = Game("monza")
    t = track_msg(g.env, "monza", g.plan)
    assert len(t["x"]) == len(t["line"]) == len(t["pace"]) > 1000 and 0.03 < np.mean(np.array(t["pace"]) == 2) < 0.3, "brake zones"
    # beginner: flat out, yanking left and right at random, never reaches the gravel (1 m past the line) or needs a rescue
    e, rng, x, worst, t0 = g.env, np.random.default_rng(0), 0, 0.0, time.time()
    for i in range(50 * 130):
        x = rng.choice([-1, 0, 0, 1]) if i % 50 == 0 else x
        s = g.step(dict(tb=1, x=x))
        w = e.trk["w_left" if e.st["n"][0] > 0 else "w_right"][int(e.st["s"][0]) % len(e.trk["s"])]
        worst = max(worst, abs(e.st["n"][0]) - w)
    ms = (time.time() - t0) / (50 * 130) * 1e3
    json.dumps(s)
    assert e.lap_times[0] and worst < 1.0 and g.rescues == 0 and len(s["cars"]) == 2, (e.lap_times[0], worst, g.rescues)
    print(f"ok: clumsy beginner laps in {e.lap_times[0][0]:.1f} s, widest {worst:+.1f} m past the white line, {ms:.2f} ms per step (budget 20 ms)")
    g.assist = False   # pro: full lock at speed spins the car and ends the run
    g.reset()
    for _ in range(100):
        s = g.step(dict(steer=0, tb=1))
    assert s["kmh"] > 200 and s["target"] > 100 and not s["done"], s
    for _ in range(150):
        s = g.step(dict(steer=1, tb=1))
    assert s["done"], "pro mode should let you crash"
    print("ok: pro mode is the raw car")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("track", nargs="?", default="monza")
    p.add_argument("--rain", type=float, default=0.0, help="rain in mm/h")
    p.add_argument("--no-ai", action="store_true", help="drive alone")
    p.add_argument("--pro", action="store_true", help="no assists: you steer the wheel, crashes end the run")
    p.add_argument("--demo", action="store_true", help="run the self-check and exit")
    a = p.parse_args()
    demo() if a.demo else serve(Game(a.track, not a.no_ai, a.rain, not a.pro))
