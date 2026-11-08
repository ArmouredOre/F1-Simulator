"""RL driver (brief D): SAC on RaceEnv, one car, random track, start point and conditions per episode. CPU.
Train: python train.py train [steps]      Eval: python train.py eval monza [rain_mm_h]   (writes logs/rl_monza_2026.parquet)
Needs: pip install stable-baselines3 (pulls in torch; kept out of requirements.txt to keep CI light)."""
import functools
import pathlib
import sys
import numpy as np
import gymnasium as gym
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from agents.reference import curvature
from env import RaceEnv, LOOK, load_track
import test_env

TRACKS, MODEL, CKPT = ["monza", "silverstone"], "models/sac_driver", "models/run4"
BLEND, OFFSET = 0.5, 2.5   # base line = half the min-curvature line (the test driver crashes on the full one); RL shifts it +-2.5 m


def conditions(rng):
    air = rng.uniform(10, 35)
    return dict(air_c=air, track_c=air + rng.uniform(5, 25), wind_ms=rng.uniform(0, 8),
                wind_dir=rng.uniform(0, 360), rain_mm_h=rng.uniform(0, 3) * (rng.random() < 0.3))


@functools.lru_cache   # the QP takes ~4 s, so solve once per track
def base_line(track):
    """Half the min-curvature line as offsets n(s), and the target speeds' curvature of that line."""
    t = load_track(track)
    n = BLEND * test_env.racing_line(t)[0]
    return n, curvature(t["x"] - t["ty"] * n, t["y"] + t["tx"] * n)


class Drive(gym.Env):
    """Single-car wrapper. The policy drives through the test_env.py driver (pure pursuit + speed profile) and
    picks its own line: a lateral offset of the base line, a target-speed scale, deploy_bias and a throttle trim.
    Learning from scratch parked or crashed the car, so RL starts from a driver that already laps.
    Straight mode and overtake are always requested. Obs = RaceEnv obs + the base line at the lookahead points."""
    observation_space = gym.spaces.Box(-np.inf, np.inf, (123,), np.float32)
    action_space = gym.spaces.Box(-1, 1, (4,), np.float32)

    def __init__(self, track=None, season="2026", episode_s=60.0, cond=None, log=False):
        self.track, self.season, self.episode_s, self.cond, self.log = track, season, episode_s, cond, log

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        track = self.track or str(rng.choice(TRACKS))
        self.env = RaceEnv(track=track, season=self.season, log=self.log,
                           conditions=self.cond or conditions(rng), max_time_s=self.episode_s)
        obs, _ = self.env.reset(seed=int(rng.integers(2**31)))
        t = self.env.trk
        self.line_n, k = base_line(track)
        self.v_t = test_env.speed_profile(self.env, k)
        self.lo, self.hi = -t["w_right"] + 1.5, t["w_left"] - 1.5
        if not self.log:   # training: drop in anywhere on track at a random speed, so every corner gets seen
            st = self.env.st
            st["s"][0] = rng.uniform(0, t["length"])
            st["vx"][0] = min(rng.uniform(20, 60), 0.8 * np.interp(st["s"][0], t["s"], self.v_t))   # never faster than the corner allows
            self.env._gaps()
            obs = self.env._obs()
        return self._obs(obs), {}

    def _obs(self, o):
        t = self.env.trk
        return np.r_[o["car_0"], np.interp((self.env.st["s"][0] + LOOK) % t["length"], t["s"], self.line_n) / 10].astype(np.float32)

    def step(self, a):
        # a[0] shifts the whole line (kept 1.5 m inside the track), a[1] scales the target speed (0.7-1.3x)
        # instead of raw throttle so the policy can't learn to park, a[3] < 0 trims the traction cap (0.5-1.0x); a = 0 is the plain test driver
        line = np.clip(self.line_n + OFFSET * a[0], self.lo, self.hi)
        steer, tb, _, _, dep = test_env.drive(self.env, line, self.v_t * (1 + 0.3 * a[1]), 1 + 0.5 * min(a[3], 0))["car_0"]
        o, r, term, trunc, info = self.env.step({"car_0": [steer, tb, 1, 1, np.clip(dep + a[2], -1, 1)]})
        return self._obs(o), float(r["car_0"]), bool(term["car_0"]), bool(trunc["car_0"]), info


def train(steps, n_envs=4):
    torch.set_num_threads(1)   # small MLP: extra torch threads only fight the env processes for the 4 CPUs
    # resume from the newest checkpoint if a run was cut off (replay buffer is not saved, so it refills)
    env = make_vec_env(Drive, n_envs, vec_env_cls=SubprocVecEnv)
    ckpts = sorted(pathlib.Path(CKPT).glob("sac_ckpt_*_steps.zip"), key=lambda p: int(p.stem.split("_")[2]))
    model = SAC.load(ckpts[-1], env=env) if ckpts else SAC(
        "MlpPolicy", env, learning_rate=3e-4, buffer_size=300_000, batch_size=256,
        gamma=0.99, policy_kwargs=dict(net_arch=[256, 256]), verbose=1)
    model.learn(steps - model.num_timesteps, log_interval=10, reset_num_timesteps=False,
                callback=CheckpointCallback(20_000 // n_envs, CKPT, "sac_ckpt"))
    model.save(MODEL)


def evaluate(track, season="2026", laps=3, rain=0.0):
    model = SAC.load(MODEL)
    env = Drive(track, season, episode_s=100.0 * laps, cond=dict(air_c=26, track_c=42, rain_mm_h=rain), log=True)
    o, _ = env.reset(seed=0)
    done = False
    while not done and len(env.env.lap_times[0]) < laps:
        o, _, term, trunc, _ = env.step(model.predict(o, deterministic=True)[0])
        done = term or trunc
    path = env.env.save_log(f"rl_{track}_{season}")
    print(f"{track} {season}: laps {np.round(env.env.lap_times[0], 2)}  crashed {term}  log {path.name}")
    return env.env.lap_times[0]


if __name__ == "__main__":   # guard needed: SubprocVecEnv re-imports this file
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    if cmd == "train":
        train(int(sys.argv[2]) if len(sys.argv) > 2 else 200_000)
    else:
        evaluate(sys.argv[2] if len(sys.argv) > 2 else "monza", rain=float(sys.argv[3]) if len(sys.argv) > 3 else 0.0)
