"""Player clone, "ghost you" (#11 step 2): learns your speed at every metre of track (braking points, corner and
exit speeds) from your saved laps (logs/player_<track>_*.parquet, written by gui.py --drive) and drives it with the
test_env.py driver in RaceEnv. It practises: after each run it compares its speed with yours and corrects its
speed targets where it was off (iterative learning), until its lap matches yours.
python clone.py monza      # writes logs/clone_monza_2026.parquet; watch it against you:
python gui.py logs/clone_monza_2026.parquet logs/player_monza_<time>.parquet --track tracks/monza.npz
ponytail: drives the half racing line (like train.py), not your line: the pursuit driver cuts chicanes when it copies a
human path and crashes. Copy your line once the steering is better (#20). One track's speed per metre, no reaction to rain or traffic."""
import glob
import sys
import numpy as np
import pandas as pd
from env import RaceEnv
import test_env


def laps(t, log):
    """[(lap number, lap time, speed in m/s on the track's 1 m grid)] per full lap of a log."""
    out = [(n, d.Time.iloc[-1] - d.Time.iloc[0], np.interp(t["s"], d.Distance, d.Speed / 3.6)) for n, d in log.groupby("LapNumber")
           if d.Distance.min() < 20 and d.Distance.max() > t["length"] - 20]   # full laps, not a crash or the unfinished last one
    return [lp for lp in out if lp[0] > 1] or out   # lap 1 is a rolling start: skip it if there are others


def run(track, line, v, laps=2, season="2026"):
    env = RaceEnv(track=track, season=season, max_time_s=200.0 * laps)
    env.reset(seed=0)
    done = False
    while not done and len(env.lap_times[0]) < laps:
        _, _, term, trunc, _ = env.step(test_env.drive(env, line, v))
        done = term["car_0"] or trunc["car_0"]
    return env


def clone(track, files, season="2026", practice=5):
    t = RaceEnv(track=track, log=False).trk
    yours = [lp for f in files for lp in laps(t, pd.read_parquet(f))]
    assert yours, f"no complete laps in {files}"
    v_p = np.mean([lp[2] for lp in yours], 0)
    line, v = 0.5 * test_env.racing_line(t)[0], v_p.copy()
    for i in range(practice):
        env = run(track, line, v, season=season)
        d = pd.DataFrame(env.rows)
        d = d[d.LapNumber == (2 if len(env.lap_times[0]) > 1 else d.LapNumber.max())]   # flying lap, or the one it crashed on
        seen = (t["s"] >= d.Distance.min()) & (t["s"] <= d.Distance.max())   # after a crash, only correct the part it drove
        v[seen] = (v + 0.7 * (v_p - np.interp(t["s"], d.Distance, d.Speed / 3.6)))[seen]
        print(f"  practice {i + 1}: laps {np.round(env.lap_times[0], 2)}{'  crashed' if env.done[0] else ''}", flush=True)
    env = run(track, line, v, season=season)
    path = env.save_log(f"clone_{track}_{season}")
    mine, theirs = env.lap_times[0][1] if len(env.lap_times[0]) > 1 else np.inf, np.mean([lp[1] for lp in yours])
    print(f"{track}: your laps {np.round([lp[1] for lp in yours], 2)}  clone {np.round(env.lap_times[0], 2)}  "
          f"off by {100 * (mine / theirs - 1):+.1f}%  log {path.name}")
    return mine, theirs


if __name__ == "__main__":
    if len(sys.argv) > 1:
        tr = sys.argv[1]
        clone(tr, sorted(glob.glob(f"logs/player_{tr}_*.parquet")))
    else:   # self-check: a fake "player" (test driver, own line, 0.9x speed) is cloned to within 1% (the #11 gate)
        env = RaceEnv(track="monza")
        env.reset(seed=0)
        line_n, k = test_env.racing_line(env.trk)
        v_t, line_n = 0.9 * test_env.speed_profile(env, k), 0.3 * line_n
        while len(env.lap_times[0]) < 3:
            env.step(test_env.drive(env, line_n, v_t))
        mine, theirs = clone("monza", [env.save_log("player_monza_selfcheck")])
        assert abs(mine / theirs - 1) < 0.01, "clone not within 1% of the player"
        print("clone OK")
