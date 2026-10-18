from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str) and not value.strip():
            return default
        out = float(value)
        if not np.isfinite(out):
            return default
        return out
    except Exception:
        return default


def _clip_factor(x: float, lo: float = 0.90, hi: float = 1.05) -> float:
    return float(np.clip(x, lo, hi))


@dataclass
class WeatherConfig:
    min_factor: float = 0.90
    max_factor: float = 1.05
    heat_threshold_c: float = 30.0
    humidity_threshold: float = 70.0
    altitude_threshold_m: float = 1000.0
    heat_penalty: float = 0.03
    humidity_penalty: float = 0.015
    altitude_penalty: float = 0.025
    wind_penalty: float = 0.010


class WeatherEngine:
    """
    Produces a venue/weather factor around 1.0.
    Useful when venue context contains:
    - temperature_c
    - humidity_pct
    - altitude_m
    - wind_kph
    """

    def __init__(self, config: WeatherConfig | None = None):
        self.config = config or WeatherConfig()

    def compute_factor(self, fixture_context: dict | None = None, team_profile: dict | None = None) -> float:
        fixture_context = fixture_context or {}
        team_profile = team_profile or {}

        temperature_c = _safe_float(fixture_context.get("temperature_c", np.nan), np.nan)
        humidity_pct = _safe_float(fixture_context.get("humidity_pct", np.nan), np.nan)
        altitude_m = _safe_float(fixture_context.get("altitude_m", np.nan), np.nan)
        wind_kph = _safe_float(fixture_context.get("wind_kph", np.nan), np.nan)

        factor = 1.0

        if np.isfinite(temperature_c) and temperature_c > self.config.heat_threshold_c:
            factor -= self.config.heat_penalty * min((temperature_c - self.config.heat_threshold_c) / 10.0, 1.0)

        if np.isfinite(humidity_pct) and humidity_pct > self.config.humidity_threshold:
            factor -= self.config.humidity_penalty * min((humidity_pct - self.config.humidity_threshold) / 20.0, 1.0)

        if np.isfinite(altitude_m) and altitude_m > self.config.altitude_threshold_m:
            factor -= self.config.altitude_penalty * min((altitude_m - self.config.altitude_threshold_m) / 1500.0, 1.0)

        if np.isfinite(wind_kph) and wind_kph > 25.0:
            factor -= self.config.wind_penalty * min((wind_kph - 25.0) / 20.0, 1.0)

        # Optional team-specific tolerance
        # Example: a warm-climate team may lose less in heat
        climate_tolerance = _safe_float(team_profile.get("climate_tolerance", 0.0), 0.0)
        factor += 0.01 * np.tanh(climate_tolerance)

        return _clip_factor(factor, self.config.min_factor, self.config.max_factor)


def get_weather_factor(fixture_context: dict | None = None, team_profile: dict | None = None) -> float:
    return WeatherEngine().compute_factor(fixture_context=fixture_context, team_profile=team_profile)