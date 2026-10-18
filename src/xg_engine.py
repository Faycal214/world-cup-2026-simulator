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


def _clip(x: float, lo: float, hi: float) -> float:
    return float(np.clip(x, lo, hi))


@dataclass
class XGConfig:
    base_xg: float = 1.05
    min_xg: float = 0.15
    max_xg: float = 2.80
    shot_quality_scale: float = 0.55
    shot_volume_scale: float = 0.08
    opponent_defense_scale: float = 0.20
    player_quality_scale: float = 0.18
    finishing_scale: float = 0.10


class XGEngine:
    """
    Optional expected-goals layer.

    This can be used either:
    - as a richer pre-Poisson goal intensity proxy,
    - or as an alternate way to estimate lambda before simulation.
    """

    def __init__(self, config: XGConfig | None = None):
        self.config = config or XGConfig()

    def estimate_team_xg(
        self,
        shots: float | None = None,
        shot_quality: float | None = None,
        player_quality: float | None = None,
        opponent_defense: float | None = None,
        finishing: float | None = None,
        team_profile: dict | None = None,
    ) -> float:
        team_profile = team_profile or {}

        shots = _safe_float(shots if shots is not None else team_profile.get("shots", 10.0), 10.0)
        shot_quality = _safe_float(shot_quality if shot_quality is not None else team_profile.get("shot_quality", 0.0), 0.0)
        player_quality = _safe_float(player_quality if player_quality is not None else team_profile.get("player_quality", 0.0), 0.0)
        opponent_defense = _safe_float(opponent_defense if opponent_defense is not None else team_profile.get("opponent_defense", 0.0), 0.0)
        finishing = _safe_float(finishing if finishing is not None else team_profile.get("finishing", 0.0), 0.0)

        xg = (
            self.config.base_xg
            + self.config.shot_volume_scale * (shots - 10.0)
            + self.config.shot_quality_scale * shot_quality
            + self.config.player_quality_scale * player_quality
            - self.config.opponent_defense_scale * opponent_defense
            + self.config.finishing_scale * finishing
        )

        return _clip(xg, self.config.min_xg, self.config.max_xg)

    def xg_to_lambda(self, xg: float, conversion_bias: float = 0.95) -> float:
        """
        Convert xG to a goal intensity proxy.
        """
        xg = _safe_float(xg, self.config.base_xg)
        return _clip(xg * conversion_bias, self.config.min_xg, self.config.max_xg)


def get_xg_estimate(
    shots: float | None = None,
    shot_quality: float | None = None,
    player_quality: float | None = None,
    opponent_defense: float | None = None,
    finishing: float | None = None,
    team_profile: dict | None = None,
) -> float:
    return XGEngine().estimate_team_xg(
        shots=shots,
        shot_quality=shot_quality,
        player_quality=player_quality,
        opponent_defense=opponent_defense,
        finishing=finishing,
        team_profile=team_profile,
    )