# F1-Simulator

F1-Simulator is a Formula 1 simulator in which reinforcement learning drivers learn to minimise lap times on real circuits. The conditions change between runs: weather, track temperature and season regulations. The physics is checked against real FastF1 telemetry before any AI trains on it. An adaptive AI rival learns to overtake you from your own driving.

> **Status (2026-11-06):** Phases 0 and 1 are merged, most of Phase 3 (viewer, keyboard driving) is merged, and the RL driver is training. The rival, driver styles and the 3D game are not started.

## What makes it different

- **Conditions change the physics.** Weather, rain, track and air temperature, wind and track rubber all change tyre grip, tyre temperature and aero.
- **Regulations live in data files.** Each season is one YAML file, and the 2026 rules are the default. Those rules cover the 768 kg minimum mass, the 350 kW MGU-K, 8.5 MJ of energy recovered per lap, active aero and overtake mode.
- **Telemetry matches real F1.** The sim logs the same channels as FastF1, so a sim lap can be overlaid on a real lap.
- **The AI rival learns from you.** It clones your driving from your laps, then trains to exploit your weak spots.

## Try it

```
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt pygame gymnasium
python gui.py --drive monza
```

Keys: up throttle, down brake, left/right steer, Z straight mode, X overtake, R restart, Esc quit. Steering is limited at speed, so use gentle taps above 200 km/h.

Replay a scripted lap: `python test_env.py monza 2026`, then `python gui.py logs/gate_monza_2026.parquet --track tracks/monza.npz`. `train.py` also needs `stable-baselines3`.

## Architecture

```
FastF1 data ─┐
Track geometry ├─> Sim core (car, tyres, aero, conditions, rules)
Season rules ─┘            │
                  Environment API  step(actions)
                           │
   Reference line · RL driver · Player clone · Rival AI · Human player
                           │
          Telemetry logs ─> Validation · GUI
```

Every driver goes through the same `step()` call, whether it's the AI or a human, so the game mode needs no rewrite.

## Roadmap

| Phase | Scope | Status | Gate |
| --- | --- | --- | --- |
| 0. Data and reference lap | Track builder, minimum-curvature racing line | Merged. Monza, Silverstone and Monaco built | Reference lap within 5% of pole: passed (Monza +2.4%, Silverstone +3.0%, Monaco -1.4%) |
| 1. Physics and conditions | Single-track car, Pacejka tyres with temperature and wear, aero with wake and tow, weather, 2025 and 2026 rules | Merged | Sanity gate passes; overlay on real laps needs FastF1 |
| 2. RL driver | SAC with randomised conditions. Run 3 is residual RL on top of the scripted gate driver | Training | Beats the reference lap with no crashes across 100 conditions |
| 3. Multi-car and GUI | Wake and tow, 2-car env, pygame replay viewer, validation plots, keyboard driving | Partly done | Playable at 60 fps; RL overtakes a reference car |
| 4. Adaptive rival | Player clone (behaviour cloning, then GAIL), rival trained on a pool of driving styles | Not started | Rival's pass rate rises over sessions |
| 5. Real-driver styles | Modern drivers: FastF1 style features, fine-tune with a driver id. Legends: hand-tuned action profiles | Not started | Styles are distinct in lap shape and racing |
| 6. 3D game, built last | Godot 4 client for the Python sim over a local socket: 3D track from the `.npz`, Blender car, cameras, HUD, AI cars, weather | Not started | Playable 3D race against the trained rival at 60 fps |

## Layout

```
data.py  track.py  physics.py  tyres.py  rules.py  env.py   # data and sim core
agents/reference.py  train.py  gate.py                      # reference line, RL driver, scripted gate driver
gui.py  validate.py                                         # viewer, keyboard driving, sim vs real overlay
seasons/  tracks/  test_*.py                                # rules, track files, tests
```

## Stack

Python 3.12, FastF1, NumPy, SciPy, OSQP, pandas, Gymnasium, stable-baselines3, PyTorch, pygame.

## Background

The design draws on GT Sophy (Wurman et al., *Nature* 2022), minimum-curvature racing lines (Heilmeier et al., 2019), the TRT tyre thermal model (Farroni et al., 2014) and the Virtual Strategy Engineer (Heilmeier et al., 2020). The full literature review and system design: [design doc](https://claude.ai/artifact/AQCKEww9qnjiayFLmV7pe6).
