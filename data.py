"""FastF1 access with a local cache: laps, telemetry and weather."""
from pathlib import Path
import fastf1

CACHE = Path(__file__).parent / "cache"
CACHE.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE))


def session(year, gp, kind="Q"):
    s = fastf1.get_session(year, gp, kind)
    s.load()
    return s


def fastest_lap(year, gp, kind="Q"):
    """(lap, telemetry, weather) for the fastest lap of a session. Telemetry has X/Y in m."""
    s = session(year, gp, kind)
    lap = s.laps.pick_fastest()
    tel = lap.get_telemetry()
    tel["X"], tel["Y"] = tel.X / 10, tel.Y / 10  # FastF1 positions are 1/10 m
    return lap, tel, s.weather_data
