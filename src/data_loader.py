from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.config import PROCESSED_DIR, RAW_DIR
from src.naming import normalize_team_name
from src.team_mentality import TEAM_CLIMATE_PROFILES, get_team_class

def _load_csv(path: Path, fallback: Path | None = None) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    if fallback and fallback.exists():
        return pd.read_csv(fallback)
    raise FileNotFoundError(f"Missing file: {path}")

def load_team_profiles() -> pd.DataFrame:
    team_features_path = PROCESSED_DIR / "team_features.csv"
    priors_path = PROCESSED_DIR / "team_strength_priors.csv"

    team_features = _load_csv(team_features_path)
    priors = _load_csv(priors_path)

    team_features = team_features.copy()
    priors = priors.copy()

    team_features["team"] = team_features["team"].apply(normalize_team_name)
    priors["team"] = priors["team"].apply(normalize_team_name)

    df = team_features.merge(priors, on="team", how="left", suffixes=("", "_prior"))

    numeric_cols = [
        "elo", "squad_size", "avg_age", "median_age", "age_std", "avg_caps", "total_caps",
        "avg_goals", "total_goals", "avg_height_cm", "height_std_cm", "gk_count",
        "df_count", "mf_count", "fw_count", "gk_share", "df_share", "mf_share",
        "fw_share", "attack_depth", "defense_depth", "experience_index",
        "tm_squad_size", "tm_avg_age", "wc_participations", "foreigners_pct",
        "market_value_eur", "avg_market_value_eur", "form_gf", "form_ga",
        "form_pts", "attack_prior", "defense_prior", "elo_prior", "market_prior",
        "form_prior", "host_prior", "travel_prior", "scenario_bias_prior",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["country"] = df["team"]
    df["team_class"] = df["team"].apply(get_team_class)
    df["climate_profile"] = df["team"].map(TEAM_CLIMATE_PROFILES).fillna("Temperate")
    df["current_elo"] = df["elo"].fillna(1500.0 + 180.0 * df.get("elo_prior", 0).fillna(0.0))
    df["market_value_raw"] = df["market_value_eur"].fillna(0.0)
    df["host_flag"] = df.get("host_flag", 0).fillna(0).astype(int)

    # Keep the object ready for fast .loc lookups
    df = df.sort_values("team").reset_index(drop=True)
    return df

def load_fixture_features() -> pd.DataFrame:
    processed_path = PROCESSED_DIR / "fixture_features.csv"
    raw_path = RAW_DIR / "fixtures_raw.csv"

    fixtures = _load_csv(processed_path, fallback=raw_path)
    fixtures = fixtures.copy()

    for col in ["home_team", "away_team", "group", "stage", "venue_city", "venue_country", "venue_stadium"]:
        if col in fixtures.columns:
            fixtures[col] = fixtures[col].astype(str).map(lambda x: normalize_team_name(x) if col in ["home_team", "away_team"] else x).fillna("")

    if "travel_intensity_proxy" in fixtures.columns:
        fixtures["travel_intensity_proxy"] = pd.to_numeric(fixtures["travel_intensity_proxy"], errors="coerce").fillna(1.0)
    else:
        fixtures["travel_intensity_proxy"] = 1.0

    if "stage_id" in fixtures.columns:
        fixtures["stage_id"] = pd.to_numeric(fixtures["stage_id"], errors="coerce").fillna(99).astype(int)

    if "date_et" in fixtures.columns:
        fixtures["date_et"] = pd.to_datetime(fixtures["date_et"], errors="coerce")

    if "time_et" in fixtures.columns:
        fixtures["time_et"] = fixtures["time_et"].astype(str).fillna("")

    return fixtures