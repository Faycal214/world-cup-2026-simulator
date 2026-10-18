from __future__ import annotations

from math import radians, sin, cos, sqrt, asin
from typing import Any

import numpy as np
import pandas as pd

from config import HOST_TEAMS


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


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    if any(pd.isna(v) for v in [lat1, lon1, lat2, lon2]):
        return float("nan")
    r = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2.0 * r * asin(sqrt(a))


def enrich_fixture_context(row: pd.Series | dict) -> dict:
    if isinstance(row, pd.Series):
        data = row.to_dict()
    else:
        data = dict(row)

    ctx = {
        "match_id": data.get("match_id"),
        "matchday": data.get("matchday"),
        "group": data.get("group"),
        "venue_name": data.get("venue_stadium") or data.get("venue_name") or "",
        "venue_stadium": data.get("venue_stadium") or data.get("venue_name") or "",
        "venue_city": data.get("venue_city") or "",
        "venue_country": data.get("venue_country") or "",
        "match_date": pd.to_datetime(data.get("match_date"), errors="coerce") if data.get("match_date") is not None else pd.NaT,
        "latitude": _safe_float(data.get("latitude"), np.nan),
        "longitude": _safe_float(data.get("longitude"), np.nan),
        "capacity": _safe_float(data.get("capacity"), np.nan),
        "temperature_c": _safe_float(data.get("temperature_c"), np.nan),
        "humidity_pct": _safe_float(data.get("humidity_pct"), np.nan),
        "altitude_m": _safe_float(data.get("altitude_m"), np.nan),
        "wind_kph": _safe_float(data.get("wind_kph"), np.nan),
        "referee_name": data.get("referee_name"),
        "referee_country": data.get("referee_country"),
        "referee_profile": {
            "yellows_per_game": _safe_float(data.get("referee_yellows_per_game"), 0.0),
            "reds_per_game": _safe_float(data.get("referee_reds_per_game"), 0.0),
            "fouls_per_game": _safe_float(data.get("referee_fouls_per_game"), 0.0),
        },
    }

    # Neutral defaults if the fixture file does not provide weather/referee fields
    if pd.isna(ctx["temperature_c"]):
        ctx["temperature_c"] = np.nan
    if pd.isna(ctx["humidity_pct"]):
        ctx["humidity_pct"] = np.nan
    if pd.isna(ctx["altitude_m"]):
        ctx["altitude_m"] = np.nan
    if pd.isna(ctx["wind_kph"]):
        ctx["wind_kph"] = np.nan

    return ctx


def enrich_team_profile(
    base_profile: dict,
    fixture_context: dict | None = None,
    previous_match_date: pd.Timestamp | None = None,
) -> dict:
    """
    Adds matchup-specific values to the base team profile.
    Works even if some columns are missing; it falls back to neutral defaults.
    """
    profile = dict(base_profile)
    fixture_context = fixture_context or {}

    # Host flag if not already present
    team_name = str(profile.get("team", "")).strip()
    if "host_prior" not in profile or profile.get("host_prior") is None:
        profile["host_prior"] = 1.0 if team_name in HOST_TEAMS else 0.0

    # Travel proxy from team coordinates to venue coordinates if available
    team_lat = _safe_float(profile.get("latitude"), np.nan)
    team_lon = _safe_float(profile.get("longitude"), np.nan)
    venue_lat = _safe_float(fixture_context.get("latitude"), np.nan)
    venue_lon = _safe_float(fixture_context.get("longitude"), np.nan)

    travel_km = haversine_km(team_lat, team_lon, venue_lat, venue_lon)
    if np.isfinite(travel_km):
        # normalize to roughly [0, 1]
        profile["travel_prior"] = _clip(travel_km / 12000.0, 0.0, 1.0)
        profile["travel_km"] = travel_km
    else:
        profile["travel_prior"] = _safe_float(profile.get("travel_prior"), 0.0)
        profile["travel_km"] = np.nan

    # Availability proxy from squad availability columns if present
    starting_xi_absences = _safe_float(profile.get("starting_xi_absences"), 0.0)
    key_player_absences = _safe_float(profile.get("key_player_absences"), 0.0)
    suspensions = _safe_float(profile.get("suspensions"), 0.0)

    availability_penalty = (
        0.04 * starting_xi_absences
        + 0.05 * key_player_absences
        + 0.02 * suspensions
    )
    profile["availability_proxy"] = _clip(1.0 - availability_penalty, 0.80, 1.00)

    # Fatigue proxy from squad size / age / rest if available
    squad_depth = _safe_float(profile.get("squad_depth"), _safe_float(profile.get("squad_size"), 23.0))
    average_age = _safe_float(profile.get("average_age"), 27.0)

    if previous_match_date is not None and pd.notna(previous_match_date) and pd.notna(fixture_context.get("match_date")):
        days_rest = (pd.Timestamp(fixture_context["match_date"]) - pd.Timestamp(previous_match_date)).days
    else:
        days_rest = _safe_float(profile.get("days_rest"), 4.0)

    profile["squad_depth"] = squad_depth
    profile["average_age"] = average_age
    profile["days_rest"] = float(days_rest)

    fatigue_penalty = 0.0
    fatigue_penalty += 0.04 * np.tanh(max(0.0, (27.0 - squad_depth)) / 8.0)
    fatigue_penalty += 0.03 * max(0.0, (average_age - 28.0) / 8.0)
    fatigue_penalty -= 0.04 * max(0.0, (days_rest - 4.0) / 4.0)

    profile["fatigue_proxy"] = _clip(1.0 - fatigue_penalty, 0.85, 1.00)

    # Climate tolerance can be set team-by-team later; keep neutral if not present
    profile["climate_tolerance"] = _safe_float(profile.get("climate_tolerance"), 0.0)

    return profile