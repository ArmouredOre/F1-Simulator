"""RL driver (brief D): SAC on RaceEnv, one car, random track, start point and conditions per episode. CPU.
Train: python train.py train [steps]      Eval: python train.py eval monza   (writes logs/rl_monza_2026.parquet)
Needs: pip install stable-baselines3 (pulls in torch; kept out of requirements.txt to keep CI light)."""
import pathlib
import sys
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from env import RaceEnv
import test_env

TRACKS, MODEL = ["monza", "silverstone"], "models/sac_driver"


def conditions(rng):
    air = rng.uniform(10, 35)
    return dict(air_c=air, track_c=air + rng.uniform(5, 25), wind_ms=rng.uniform(0, 8),
                wind_dir=rng.uniform(0, 360), rain_mm_h=rng.uniform(0, 3) * (rng.random() < 0.3))


class Drive(gym.Env):
    """Single-car wrapper. The policy corrects the test_env.py driver (pure pursuit + speed profile): steer,
    target-speed scale and deploy_bias. Learning from scratch parked or crashed the car,
    so RL starts from a driver that already laps. Straight mode and overtake are always requested."""
    observation_space = gym.spaces.Box(-np.inf, np.inf, (99,), np.float32)
    action_space = gym.spaces.Box(-1, 1, (3,), np.float32)

    def __init__(self, track=None, season="2026", episode_s=60.0, cond=None, log=False):
        self.track, self.season, self.episode_s, self.cond, self.log = track, season, episode_s, cond, log

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        self.env = RaceEnv(track=self.track or str(rng.choice(TRACKS)), season=self.season, log=self.log,
                           conditions=self.cond or conditions(rng), max_time_s=self.episode_s)
        obs, _ = self.env.reset(seed=int(rng.integers(2**31)))
        self.line_n = np.zeros_like(self.env.trk["s"])     # centreline, as in test_env.run
        self.v_t = test_env.speed_profile(self.env, self.env.trk["kappa"])
        if not self.log:   # training: drop in anywhere on track at a random speed, so every corner gets seen
            st = self.env.st
            st["s"][0], st["vx"][0] = rng.uniform(0, self.env.trk["length"]), rng.uniform(20, 60)
            self.env._gaps()
            obs = self.env._obs()
        return obs["car_0"], {}

    def step(self, a):
        # a[1] scales the target speed (0.8-1.2x) instead of raw throttle, so the policy can't learn to park
        steer, tb, _, _, dep = test_env.drive(self.env, self.line_n, self.v_t * (1 + 0.2 * a[1]))["car_0"]
        act = [np.clip(steer + 0.1 * a[0], -1, 1), tb, 1, 1, np.clip(dep + a[2], -1, 1)]
        o, r, term, trunc, info = self.env.step({"car_0": act})
        return o["car_0"], float(r["car_0"]), bool(term["car_0"]), bool(trunc["car_0"]), info


def train(steps):
    # ponytail: one env process (~400 steps/s physics); SubprocVecEnv if wall-clock matters
    # resume from the newest checkpoint if a run was cut off (replay buffer is not saved, so it refills)
    ckpts = sorted(pathlib.Path("models").glob("sac_ckpt_*_steps.zip"), key=lambda p: int(p.stem.split("_")[2]))
    model = SAC.load(ckpts[-1], env=Drive()) if ckpts else SAC(
        "MlpPolicy", Drive(), learning_rate=3e-4, buffer_size=300_000, batch_size=256,
        gamma=0.99, policy_kwargs=dict(net_arch=[256, 256]), verbose=1)
    model.learn(steps - model.num_timesteps, log_interval=10, reset_num_timesteps=False,
                callback=CheckpointCallback(20_000, "models", "sac_ckpt"))
    model.save(MODEL)


def evaluate(track, season="2026", laps=3):
    model = SAC.load(MODEL)
    env = Drive(track, season, episode_s=100.0 * laps, cond=dict(air_c=26, track_c=42), log=True)
    o, _ = env.reset(seed=0)
    done = False
    while not done and len(env.env.lap_times[0]) < laps:
        o, _, term, trunc, _ = env.step(model.predict(o, deterministic=True)[0])
        done = term or trunc
    path = env.env.save_log(f"rl_{track}_{season}")
    print(f"{track} {season}: laps {np.round(env.env.lap_times[0], 2)}  crashed {term}  log {path.name}")
    return env.env.lap_times[0]


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    if cmd == "train":
        train(int(sys.argv[2]) if len(sys.argv) > 2 else 200_000)
    else:
        evaluate(sys.argv[2] if len(sys.argv) > 2 else "monza")
