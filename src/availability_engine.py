from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


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


def _clip_factor(x: float, lo: float = 0.80, hi: float = 1.00) -> float:
    return float(np.clip(x, lo, hi))


@dataclass
class AvailabilityConfig:
    baseline: float = 1.0
    injury_penalty_per_starting_xi_absence: float = 0.035
    suspension_penalty_per_player: float = 0.020
    missing_key_player_penalty: float = 0.050
    missing_goalkeeper_penalty: float = 0.030
    missing_defender_penalty: float = 0.020
    missing_midfielder_penalty: float = 0.018
    missing_forward_penalty: float = 0.022
    min_factor: float = 0.80
    max_factor: float = 1.00


class AvailabilityEngine:
    """
    Builds a squad availability factor in [0.80, 1.00].

    The idea:
    - 1.00 means full strength available
    - lower values mean injuries / suspensions / absences
    """

    def __init__(self, config: AvailabilityConfig | None = None):
        self.config = config or AvailabilityConfig()

    @staticmethod
    def _position_penalty(position: str) -> float:
        pos = str(position).strip().lower()
        if pos in {"gk", "goalkeeper"}:
            return 0.030
        if pos in {"cb", "lb", "rb", "lcb", "rcb", "defender", "df"}:
            return 0.020
        if pos in {"cm", "dm", "am", "midfielder", "mf"}:
            return 0.018
        if pos in {"lw", "rw", "st", "cf", "fw", "forward", "attacker"}:
            return 0.022
        return 0.015

    def compute_factor(
        self,
        team_profile: dict | pd.Series | None = None,
        squad_table: pd.DataFrame | None = None,
        injuries_table: pd.DataFrame | None = None,
        suspensions_table: pd.DataFrame | None = None,
    ) -> float:
        team_profile = team_profile or {}

        factor = self.config.baseline

        # Team-level indicators if available
        starting_xi_absences = _safe_float(team_profile.get("starting_xi_absences", 0.0), 0.0)
        key_player_absences = _safe_float(team_profile.get("key_player_absences", 0.0), 0.0)

        factor -= self.config.injury_penalty_per_starting_xi_absence * starting_xi_absences
        factor -= self.config.missing_key_player_penalty * key_player_absences

        # Squad table logic
        if squad_table is not None and not squad_table.empty:
            cols = {c.lower(): c for c in squad_table.columns}
            pos_col = cols.get("position", None)
            avail_col = cols.get("available", None)
            key_col = cols.get("is_key_player", None)

            if pos_col is not None:
                for _, row in squad_table.iterrows():
                    available = True
                    if avail_col is not None:
                        val = row[avail_col]
                        if isinstance(val, str):
                            available = val.strip().lower() in {"1", "true", "yes", "y"}
                        else:
                            available = bool(val)

                    if not available:
                        penalty = self._position_penalty(row[pos_col])
                        if key_col is not None and bool(row[key_col]):
                            penalty += 0.020
                        factor -= penalty

        # Injuries / suspensions tables
        if injuries_table is not None and not injuries_table.empty:
            if "team" in injuries_table.columns:
                team_mask = injuries_table["team"].astype(str).str.lower() == str(team_profile.get("team", "")).lower()
                team_injuries = injuries_table.loc[team_mask]
            else:
                team_injuries = injuries_table

            for _, row in team_injuries.iterrows():
                pos = row.get("position", "")
                penalty = self._position_penalty(pos)
                if bool(row.get("is_key_player", False)):
                    penalty += self.config.missing_key_player_penalty
                factor -= penalty

        if suspensions_table is not None and not suspensions_table.empty:
            if "team" in suspensions_table.columns:
                team_mask = suspensions_table["team"].astype(str).str.lower() == str(team_profile.get("team", "")).lower()
                team_suspensions = suspensions_table.loc[team_mask]
            else:
                team_suspensions = suspensions_table

            factor -= self.config.suspension_penalty_per_player * len(team_suspensions)

        return _clip_factor(factor, self.config.min_factor, self.config.max_factor)


def get_availability_factor(
    team_profile: dict | pd.Series | None = None,
    squad_table: pd.DataFrame | None = None,
    injuries_table: pd.DataFrame | None = None,
    suspensions_table: pd.DataFrame | None = None,
) -> float:
    return AvailabilityEngine().compute_factor(
        team_profile=team_profile,
        squad_table=squad_table,
        injuries_table=injuries_table,
        suspensions_table=suspensions_table,
    )