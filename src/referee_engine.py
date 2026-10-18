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


def _clip_factor(x: float, lo: float = 0.97, hi: float = 1.03) -> float:
    return float(np.clip(x, lo, hi))


@dataclass
class RefereeConfig:
    min_factor: float = 0.97
    max_factor: float = 1.03
    yellow_tolerance_scale: float = 0.015
    red_tolerance_scale: float = 0.020
    foul_tolerance_scale: float = 0.010


class RefereeEngine:
    """
    Very small referee layer.

    This should remain a mild multiplier because referee effects are easy to overfit.
    """

    def __init__(self, config: RefereeConfig | None = None):
        self.config = config or RefereeConfig()

    def compute_factor(
        self,
        referee_profile: dict | None = None,
        team_profile: dict | None = None,
        opponent_profile: dict | None = None,
    ) -> float:
        referee_profile = referee_profile or {}
        team_profile = team_profile or {}
        opponent_profile = opponent_profile or {}

        factor = 1.0

        yellows_per_game = _safe_float(referee_profile.get("yellows_per_game", 0.0), 0.0)
        reds_per_game = _safe_float(referee_profile.get("reds_per_game", 0.0), 0.0)
        fouls_per_game = _safe_float(referee_profile.get("fouls_per_game", 0.0), 0.0)

        team_discipline = _safe_float(team_profile.get("discipline", 0.0), 0.0)
        opp_discipline = _safe_float(opponent_profile.get("discipline", 0.0), 0.0)

        # Disciplined teams benefit slightly from strict referees
        discipline_diff = team_discipline - opp_discipline

        factor += self.config.yellow_tolerance_scale * np.tanh((2.5 - yellows_per_game) / 2.0) * discipline_diff
        factor += self.config.red_tolerance_scale * np.tanh((1.0 - reds_per_game) / 1.0) * discipline_diff
        factor += self.config.foul_tolerance_scale * np.tanh((20.0 - fouls_per_game) / 10.0) * discipline_diff

        return _clip_factor(factor, self.config.min_factor, self.config.max_factor)


def get_referee_factor(
    referee_profile: dict | None = None,
    team_profile: dict | None = None,
    opponent_profile: dict | None = None,
) -> float:
    return RefereeEngine().compute_factor(
        referee_profile=referee_profile,
        team_profile=team_profile,
        opponent_profile=opponent_profile,
    )