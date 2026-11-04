# F1-Simulator

F1-Simulator is a Formula 1 simulator in which reinforcement learning drivers learn to minimise lap times on real circuits. The conditions change between runs: weather, track temperature and season regulations. The physics is checked against real FastF1 telemetry before any AI trains on it. An adaptive AI rival learns to overtake you from your own driving.

> **Status:** early development. The planning and design phase is done, and Phase 0 (data and reference lap) and Phase 1 (physics) are in progress.

## What makes it different

- **Conditions change the physics.** Weather, rain, track and air temperature, wind and track rubber all change tyre grip, tyre temperature and aero.
- **Regulations live in data files.** Each season is one YAML file, and the 2026 rules are the default. Those rules cover the 768 kg minimum mass, the 350 kW MGU-K, 8.5 MJ of energy recovered per lap, active aero and overtake mode.
- **Telemetry matches real F1.** The sim logs the same channels as FastF1, so a sim lap can be overlaid on a real lap.
- **The AI rival learns from you.** It clones your driving from your laps, then trains to exploit your weak spots.

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

| Phase | Scope | Gate |
| --- | --- | --- |
| 0. Data and reference lap | FastF1 data download, track builder, minimum-curvature racing line | Reference lap within 5% of the real pole time |
| 1. Physics and conditions | Single-track car, Pacejka tyres with temperature and wear, aero, weather, rules engine | Speed trace and grip envelope match real laps; lap time within 2% |
| 2. RL driver | SAC with randomised conditions, learned energy deployment | Beats the reference lap with no crashes across 100 conditions |
| 3. Multi-car and GUI | Wake and slipstream, multi-agent API, pygame viewer, keyboard driving | Playable at 60 fps |
| 4. Adaptive rival | Player clone (behaviour cloning, then GAIL), rival trained on a pool of driving styles | Rival's pass rate rises over sessions |

## Planned layout

```
data.py  track.py  physics.py  tyres.py  rules.py  env.py
agents/  gui.py  validate.py  seasons/
```

## Stack

Python 3.12, FastF1, NumPy, Gymnasium / PettingZoo, stable-baselines3, PyTorch, pygame.

## Background

The design draws on GT Sophy (Wurman et al., *Nature* 2022), minimum-curvature racing lines (Heilmeier et al., 2019), the TRT tyre thermal model (Farroni et al., 2014) and the Virtual Strategy Engineer (Heilmeier et al., 2020). The full literature review and system design: [design doc](https://claude.ai/artifact/AQCKEww9qnjiayFLmV7pe6).
