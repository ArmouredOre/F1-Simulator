"""Runs each self-checking script; a script that isn't in the repo yet is skipped. No network needed."""
import pathlib, subprocess, sys
import pytest

ROOT = pathlib.Path(__file__).parent


@pytest.mark.parametrize("cmd", [
    ["test_validate.py"],        # validator maths on a synthetic circle
    ["test_env.py"],             # sim sanity gate: 3 clean laps, speeds/g/tyre temps in F1 range
    ["test_collide.py"],         # solid cars, barriers, gravel, damage costs grip
    ["agents/reference.py"],     # reference lap within 5% of 2024 pole, line no tighter than R 15 m
    ["gate.py", "2025"],         # physics gate: reference lap within 5% of 2024 pole
])
def test_script(cmd):
    if not (ROOT / cmd[0]).exists():
        pytest.skip(f"{cmd[0]} not in repo yet")
    subprocess.run([sys.executable, *cmd], cwd=ROOT, check=True)
