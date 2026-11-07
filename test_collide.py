"""Self-check for collisions and damage: solid cars, barriers, gravel, and damage costing grip.
Run: python test_collide.py"""
import numpy as np
import physics
from env import RaceEnv

env = RaceEnv(n_cars=2, log=False)

# nose to tail: rear car 20 m/s faster and 3 m into the gearbox ahead -> pushed apart, share speed, both damaged
env.reset(seed=0)
st = env.st
st["s"][:], st["n"][:], st["vx"][:] = [103.0, 100.0], 0.0, [50.0, 70.0]
hit, _ = physics.collide(st, env.trk, 0.1)
assert hit.all() and st["s"][0] - st["s"][1] >= physics.CAR_L - 1e-9, "cars still overlap"
assert st["vx"][0] == st["vx"][1] == 60.0 and (st["dmg"] > 0.6).all(), (st["vx"], st["dmg"])

# side by side: pushed apart across the track
env.reset(seed=0)
st = env.st
st["s"][:], st["n"][:] = 100.0, [0.5, -0.5]
physics.collide(st, env.trk, 0.1)
assert st["n"][0] - st["n"][1] >= physics.CAR_W - 1e-9, "side by side still overlap"

# barrier: clamped to the wall, slides along it, damage from the speed into it
env.reset(seed=0)
st = env.st
st["s"][:], st["n"][:], st["vx"][:], st["xi"][:] = [100.0, 300.0], [20.0, 0.0], 60.0, [0.3, 0.0]
_, wall = physics.collide(st, env.trk, 0.1)
w = physics.at(env.trk, "w_left", 100.0) + physics.WALL
assert wall[0] and not wall[1] and abs(st["n"][0] - w) < 1e-9 and st["xi"][0] == 0, "wall clamp"
assert 0.5 < st["dmg"][0] < 0.7 and st["dmg"][1] == 0, st["dmg"]   # (60 sin(0.3) - 3) / 27 = 0.54

# gravel: slows the car and chips damage
env.reset(seed=0)
st = env.st
st["s"][:], st["n"][:], st["vx"][:] = [100.0, 300.0], [-8.0, 0.0], 50.0
physics.collide(st, env.trk, 0.1)
assert st["vx"][0] < 50 and st["vx"][1] == 50 and 0 < st["dmg"][0] < 0.01 and st["dmg"][1] == 0, "gravel"

# damage costs grip: same car, same steering, the damaged one corners slower
env.reset(seed=0)
st = env.st
st["s"][:], st["n"][:], st["vx"][:], st["dmg"][:] = [100.0, 300.0], 0.0, 60.0, [0.0, 0.8]
for _ in range(100):
    physics.step(st, np.full(2, 0.5), np.zeros(2), np.zeros(2), np.zeros(2, bool), np.zeros(2), np.zeros(2),
                 np.zeros(2), env.R, env.trk, env.cond, env.rho, 0.01)
assert abs(st["ay"][1]) < 0.85 * abs(st["ay"][0]), st["ay"]

# full speed into the wall wrecks the car and ends its race
env = RaceEnv(n_cars=1, log=False)
env.reset(seed=0, v0=80.0)
env.st["xi"][:] = 0.6
for _ in range(30):
    _, _, done, _, _ = env.step({"car_0": [0, 1, 0, 0, 0]})
    if done["car_0"]:
        break
assert done["car_0"] and env.st["dmg"][0] >= 1, env.st["dmg"]
print("collisions OK")
