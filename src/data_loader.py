from __future__ import annotations

from pathlib import Path
import pandas as pd

from config import PROCESSED_DIR, RAW_DIR
from naming import normalize_team_name
from team_mentality import TEAM_CLIMATE_PROFILES, get_team_class


def _load_csv(path: Path, fallback: Path | None = None) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    if fallback and fallback.exists():
        return pd.read_csv(fallback)
    raise FileNotFoundError(f"Missing file: {path}")


def _to_numeric(df: pd.DataFrame, cols: list[str], default: float = 0.0) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(default)
    return out


def _ensure_columns(df: pd.DataFrame, defaults: dict) -> pd.DataFrame:
    out = df.copy()
    for col, value in defaults.items():
        if col not in out.columns:
            out[col] = value
    return out


def load_team_profiles() -> pd.DataFrame:
    team_features_path = PROCESSED_DIR / "team_features.csv"
    priors_path = PROCESSED_DIR / "team_strength_priors.csv"

    team_features = _load_csv(team_features_path).copy()
    priors = _load_csv(priors_path).copy()

    team_features["team"] = team_features["team"].apply(normalize_team_name)
    priors["team"] = priors["team"].apply(normalize_team_name)

    df = team_features.merge(priors, on="team", how="left", suffixes=("", "_prior"))

    # ------------------------------------------------------------------
    # Canonical aliases
    # ------------------------------------------------------------------
    rename_map = {
        "average_age": "avg_age",
        "squad_depth": "squad_depth",
        "squad_size": "squad_size",
        "median_age": "median_age",
        "age_std": "age_std",
        "avg_caps": "avg_caps",
        "total_caps": "total_caps",
        "avg_goals": "avg_goals",
        "total_goals": "total_goals",
        "avg_height_cm": "avg_height_cm",
        "height_std_cm": "height_std_cm",
        "gk_count": "gk_count",
        "df_count": "df_count",
        "mf_count": "mf_count",
        "fw_count": "fw_count",
        "gk_share": "gk_share",
        "df_share": "df_share",
        "mf_share": "mf_share",
        "fw_share": "fw_share",
        "attack_depth": "attack_depth",
        "defense_depth": "defense_depth",
        "experience_index": "experience_index",
        "tm_squad_size": "tm_squad_size",
        "tm_avg_age": "tm_avg_age",
        "wc_participations": "wc_participations",
        "foreigners_pct": "foreigners_pct",
        "market_value_eur": "market_value_eur",
        "avg_market_value_eur": "avg_market_value_eur",
        "form_gf": "form_gf",
        "form_ga": "form_ga",
        "form_pts": "form_pts",
        "elo": "elo",
        "host_flag": "host_flag",
        "starting_xi_absences": "starting_xi_absences",
        "key_player_absences": "key_player_absences",
        "injury_count": "injury_count",
        "suspension_count": "suspension_count",
        "cumulative_minutes": "cumulative_minutes",
        "days_rest": "days_rest",
        "discipline": "discipline",
        "climate_tolerance": "climate_tolerance",
    }
    for src, dst in rename_map.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]

    # ------------------------------------------------------------------
    # Make sure the model-facing columns exist
    # ------------------------------------------------------------------
    df = _ensure_columns(
        df,
        {
            "elo": 1500.0,
            "squad_size": 23,
            "avg_age": 27.0,
            "median_age": 27.0,
            "age_std": 2.5,
            "avg_caps": 0.0,
            "total_caps": 0.0,
            "avg_goals": 0.0,
            "total_goals": 0.0,
            "avg_height_cm": 180.0,
            "height_std_cm": 5.0,
            "gk_count": 3,
            "df_count": 8,
            "mf_count": 8,
            "fw_count": 4,
            "gk_share": 3 / 23,
            "df_share": 8 / 23,
            "mf_share": 8 / 23,
            "fw_share": 4 / 23,
            "attack_depth": 0.0,
            "defense_depth": 0.0,
            "experience_index": 0.0,
            "tm_squad_size": 23,
            "tm_avg_age": 27.0,
            "wc_participations": 0.0,
            "foreigners_pct": 0.0,
            "market_value_eur": 0.0,
            "avg_market_value_eur": 0.0,
            "form_gf": 0.0,
            "form_ga": 0.0,
            "form_pts": 0.0,
            "starting_xi_absences": 0.0,
            "key_player_absences": 0.0,
            "injury_count": 0.0,
            "suspension_count": 0.0,
            "cumulative_minutes": 0.0,
            "days_rest": 4.0,
            "discipline": 0.0,
            "climate_tolerance": 0.0,
            "host_flag": 0,
            "attack_prior": 0.0,
            "defense_prior": 0.0,
            "elo_prior": 0.0,
            "market_prior": 0.0,
            "form_prior": 0.0,
            "experience_prior": 0.0,
            "host_prior": 0.0,
            "travel_prior": 0.0,
            "scenario_bias_prior": 0.0,
        },
    )

    numeric_cols = [
        "elo",
        "squad_size",
        "avg_age",
        "median_age",
        "age_std",
        "avg_caps",
        "total_caps",
        "avg_goals",
        "total_goals",
        "avg_height_cm",
        "height_std_cm",
        "gk_count",
        "df_count",
        "mf_count",
        "fw_count",
        "gk_share",
        "df_share",
        "mf_share",
        "fw_share",
        "attack_depth",
        "defense_depth",
        "experience_index",
        "tm_squad_size",
        "tm_avg_age",
        "wc_participations",
        "foreigners_pct",
        "market_value_eur",
        "avg_market_value_eur",
        "form_gf",
        "form_ga",
        "form_pts",
        "starting_xi_absences",
        "key_player_absences",
        "injury_count",
        "suspension_count",
        "cumulative_minutes",
        "days_rest",
        "discipline",
        "climate_tolerance",
        "attack_prior",
        "defense_prior",
        "elo_prior",
        "market_prior",
        "form_prior",
        "experience_prior",
        "host_prior",
        "travel_prior",
        "scenario_bias_prior",
    ]
    df = _to_numeric(df, numeric_cols, default=0.0)

    # ------------------------------------------------------------------
    # Derived / normalized fields used by the simulator
    # ------------------------------------------------------------------
    df["country"] = df["team"]
    df["team_class"] = df["team"].apply(get_team_class)
    df["climate_profile"] = df["team"].map(TEAM_CLIMATE_PROFILES).fillna("Temperate")

    # current_elo: explicit elo if available, else a mild transform from the prior
    df["current_elo"] = df["elo"].fillna(1500.0 + 180.0 * df["elo_prior"].fillna(0.0))

    # raw market value helper
    df["market_value_raw"] = df["market_value_eur"].fillna(0.0)

    # host flag: keep it binary
    df["host_flag"] = df["host_flag"].fillna(0).astype(int)

    # ------------------------------------------------------------------
    # Convenience columns for the optional layers
    # ------------------------------------------------------------------
    # These keep the engines from relying on fallback defaults.
    df["available_players"] = (df["squad_size"] - df["starting_xi_absences"] - df["key_player_absences"]).clip(lower=0)
    df["squad_depth"] = df["squad_size"]
    df["average_age"] = df["avg_age"]
    df["referee_bias_tolerance"] = df["discipline"]
    df["fitness_load"] = df["cumulative_minutes"]
    df["rest_days"] = df["days_rest"]
    df["weather_resilience"] = df["climate_tolerance"]

    # Keep the object ready for fast .loc lookups
    df = df.sort_values("team").reset_index(drop=True)
    return df


def load_fixture_features() -> pd.DataFrame:
    processed_path = PROCESSED_DIR / "fixture_features.csv"
    raw_path = RAW_DIR / "fixtures_raw.csv"

    fixtures = _load_csv(processed_path, fallback=raw_path).copy()

    # Canonical fixture columns
    defaults = {
        "match_id": "",
        "matchday": 0,
        "group": "",
        "stage": "",
        "home_team": "",
        "away_team": "",
        "venue_city": "",
        "venue_country": "",
        "venue_stadium": "",
        "venue_name": "",
        "travel_intensity_proxy": 1.0,
        "stage_id": 99,
        "temperature_c": float("nan"),
        "humidity_pct": float("nan"),
        "altitude_m": float("nan"),
        "wind_kph": float("nan"),
        "referee_name": "",
        "referee_profile": {},
    }
    fixtures = _ensure_columns(fixtures, defaults)

    # Normalize team columns if present
    for col in ["home_team", "away_team"]:
        if col in fixtures.columns:
            fixtures[col] = fixtures[col].astype(str).map(normalize_team_name).fillna("")

    # Basic numeric cleanup
    for col in ["matchday", "travel_intensity_proxy", "stage_id", "temperature_c", "humidity_pct", "altitude_m", "wind_kph"]:
        if col in fixtures.columns:
            fixtures[col] = pd.to_numeric(fixtures[col], errors="coerce")

    if "travel_intensity_proxy" in fixtures.columns:
        fixtures["travel_intensity_proxy"] = fixtures["travel_intensity_proxy"].fillna(1.0)
    if "stage_id" in fixtures.columns:
        fixtures["stage_id"] = fixtures["stage_id"].fillna(99).astype(int)

    for col in ["venue_city", "venue_country", "venue_stadium", "venue_name", "stage", "group", "referee_name"]:
        if col in fixtures.columns:
            fixtures[col] = fixtures[col].astype(str).fillna("")

    # Make sure referee_profile is always present and usable
    if "referee_profile" not in fixtures.columns:
        fixtures["referee_profile"] = [{} for _ in range(len(fixtures))]
    else:
        fixtures["referee_profile"] = fixtures["referee_profile"].apply(
            lambda x: x if isinstance(x, dict) else {}
        )

    # Date/time fields if available
    if "date_et" in fixtures.columns:
        fixtures["date_et"] = pd.to_datetime(fixtures["date_et"], errors="coerce")
    if "time_et" in fixtures.columns:
        fixtures["time_et"] = fixtures["time_et"].astype(str).fillna("")

    return fixtures