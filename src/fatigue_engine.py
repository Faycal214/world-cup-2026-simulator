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


def _clip_factor(x: float, lo: float = 0.85, hi: float = 1.00) -> float:
    return float(np.clip(x, lo, hi))


@dataclass
class FatigueConfig:
    min_factor: float = 0.85
    max_factor: float = 1.00
    minutes_scale: float = 1800.0
    age_scale: float = 30.0
    rest_scale: float = 4.0
    depth_scale: float = 25.0


class FatigueEngine:
    """
    Produces a fatigue factor in [0.85, 1.00].

    Intended usage:
        lambda_final = lambda_base * fatigue_factor
    """

    def __init__(self, config: FatigueConfig | None = None):
        self.config = config or FatigueConfig()

    def compute_factor(
        self,
        cumulative_minutes: float | None = None,
        average_age: float | None = None,
        squad_depth: float | None = None,
        days_rest: float | None = None,
        team_profile: dict | None = None,
    ) -> float:
        team_profile = team_profile or {}

        cumulative_minutes = _safe_float(
            cumulative_minutes if cumulative_minutes is not None else team_profile.get("cumulative_minutes", 0.0),
            0.0,
        )
        average_age = _safe_float(
            average_age if average_age is not None else team_profile.get("average_age", 27.0),
            27.0,
        )
        squad_depth = _safe_float(
            squad_depth if squad_depth is not None else team_profile.get("squad_depth", 23.0),
            23.0,
        )
        days_rest = _safe_float(
            days_rest if days_rest is not None else team_profile.get("days_rest", 4.0),
            4.0,
        )

        # Higher cumulative minutes => more fatigue
        minutes_term = np.tanh(cumulative_minutes / self.config.minutes_scale)

        # Older teams tend to fatigue more in a tournament setting
        age_term = np.tanh((average_age - self.config.age_scale) / 6.0)

        # Better depth reduces fatigue
        depth_term = np.tanh((self.config.depth_scale - squad_depth) / 8.0)

        # More rest => less fatigue
        rest_bonus = np.tanh((days_rest - self.config.rest_scale) / 2.0)

        # Combine into a small penalty
        penalty = 0.06 * minutes_term + 0.03 * max(age_term, 0.0) + 0.04 * max(depth_term, 0.0) - 0.05 * max(rest_bonus, 0.0)

        factor = 1.0 - penalty
        return _clip_factor(factor, self.config.min_factor, self.config.max_factor)


def get_fatigue_factor(
    cumulative_minutes: float | None = None,
    average_age: float | None = None,
    squad_depth: float | None = None,
    days_rest: float | None = None,
    team_profile: dict | None = None,
) -> float:
    return FatigueEngine().compute_factor(
        cumulative_minutes=cumulative_minutes,
        average_age=average_age,
        squad_depth=squad_depth,
        days_rest=days_rest,
        team_profile=team_profile,
    )