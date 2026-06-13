from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd
from unidecode import unidecode

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

HOST_TEAMS = {"Mexico", "Canada", "USA"}

WEST_REGION = {
    "vancouver",
    "seattle",
    "los angeles",
    "guadalajara",
    "mexico city",
    "monterrey",
}
CENTRAL_REGION = {
    "houston",
    "dallas",
    "atlanta",
    "toronto",
    "boston",
    "san francisco",
    "san francisco bay area",
    "kansas city",
}
EAST_REGION = {
    "miami",
    "new york",
    "new jersey",
    "philadelphia",
}

def canon_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        re.sub(r"[^a-z0-9]+", "_", unidecode(str(c).lower())).strip("_")
        for c in df.columns
    ]
    return df

def clean_team(x) -> str:
    if pd.isna(x):
        return ""
    s = unidecode(str(x))
    s = re.sub(r"\s*\([A-Z]{3}\)\s*$", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def stage_order(stage: str) -> int:
    s = unidecode(str(stage)).lower().strip()
    if "group" in s:
        return 1
    if "round of 32" in s:
        return 2
    if "round of 16" in s:
        return 3
    if "quarter" in s:
        return 4
    if "semi" in s:
        return 5
    if "final" in s:
        return 6
    return 99

def infer_region(city: str) -> str:
    if pd.isna(city) or str(city).strip() == "":
        return "Unknown"
    s = unidecode(str(city)).lower().strip()

    if any(k in s for k in WEST_REGION):
        return "West"
    if any(k in s for k in CENTRAL_REGION):
        return "Central"
    if any(k in s for k in EAST_REGION):
        return "East"
    return "Unknown"

def load_csv(name: str) -> pd.DataFrame:
    path = RAW / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)

def build_fixture_features():
    fixtures = canon_cols(load_csv("fixtures_raw.csv"))
    stadiums = canon_cols(load_csv("world_cup_2026_stadiums.csv"))

    # Flexible column lookup in fixtures_raw.csv
    col_match_id = "match_id" if "match_id" in fixtures.columns else None
    col_stage = "stage" if "stage" in fixtures.columns else None
    col_group = "group" if "group" in fixtures.columns else None
    col_home = "home_team" if "home_team" in fixtures.columns else None
    col_away = "away_team" if "away_team" in fixtures.columns else None
    col_date = "date_et" if "date_et" in fixtures.columns else ("date" if "date" in fixtures.columns else None)
    col_time = "time_et" if "time_et" in fixtures.columns else ("time" if "time" in fixtures.columns else None)
    col_city = "venue_city" if "venue_city" in fixtures.columns else ("city" if "city" in fixtures.columns else None)
    col_country = "venue_country" if "venue_country" in fixtures.columns else ("country" if "country" in fixtures.columns else None)
    col_stadium = "venue_stadium" if "venue_stadium" in fixtures.columns else ("stadium" if "stadium" in fixtures.columns else None)

    required = [col_stage, col_home, col_away]
    if any(c is None for c in required):
        raise ValueError("fixtures_raw.csv must contain at least stage, home_team, away_team")

    fx = pd.DataFrame()
    if col_match_id:
        fx["match_id"] = fixtures[col_match_id]
    else:
        fx["match_id"] = np.arange(1, len(fixtures) + 1)

    fx["stage"] = fixtures[col_stage].astype(str).str.strip()
    fx["stage_id"] = fx["stage"].map(stage_order)

    fx["group"] = fixtures[col_group].astype(str).str.strip() if col_group else ""
    fx["home_team"] = fixtures[col_home].map(clean_team)
    fx["away_team"] = fixtures[col_away].map(clean_team)

    if col_date:
        fx["date_et"] = pd.to_datetime(fixtures[col_date], errors="coerce")
    else:
        fx["date_et"] = pd.NaT

    fx["time_et"] = fixtures[col_time].astype(str).str.strip() if col_time else ""

    fx["venue_city"] = fixtures[col_city].astype(str).str.strip() if col_city else ""
    fx["venue_country"] = fixtures[col_country].astype(str).str.strip() if col_country else ""
    fx["venue_stadium"] = fixtures[col_stadium].astype(str).str.strip() if col_stadium else ""

    fx["venue_region"] = fx["venue_city"].apply(infer_region)

    # Optional stadium enrichment
    stadiums = stadiums.copy()
    s_stadium = "stadium" if "stadium" in stadiums.columns else None
    s_city = "city" if "city" in stadiums.columns else None
    s_country = "country" if "country" in stadiums.columns else None
    s_lat = "latitude" if "latitude" in stadiums.columns else ("lat" if "lat" in stadiums.columns else None)
    s_lon = "longitude" if "longitude" in stadiums.columns else ("lon" if "lon" in stadiums.columns else ("lng" if "lng" in stadiums.columns else None))
    s_cap = "capacity" if "capacity" in stadiums.columns else None

    if s_city and s_country:
        stadiums["city_key"] = stadiums[s_city].astype(str).str.strip().str.lower()
        stadiums["country_key"] = stadiums[s_country].astype(str).str.strip().str.lower()

        fx["city_key"] = fx["venue_city"].astype(str).str.strip().str.lower()
        fx["country_key"] = fx["venue_country"].astype(str).str.strip().str.lower()

        merge_cols = ["city_key", "country_key"]
        keep_cols = merge_cols.copy()
        if s_stadium:
            keep_cols.append(s_stadium)
        if s_lat:
            keep_cols.append(s_lat)
        if s_lon:
            keep_cols.append(s_lon)
        if s_cap:
            keep_cols.append(s_cap)

        fx = fx.merge(
            stadiums[keep_cols].drop_duplicates(merge_cols),
            on=merge_cols,
            how="left",
            suffixes=("", "_stadium"),
        )

        if s_stadium and s_stadium in fx.columns:
            fx["venue_stadium"] = fx["venue_stadium"].replace("", np.nan).fillna(fx[s_stadium])

        fx["venue_latitude"] = pd.to_numeric(fx[s_lat], errors="coerce") if s_lat and s_lat in fx.columns else np.nan
        fx["venue_longitude"] = pd.to_numeric(fx[s_lon], errors="coerce") if s_lon and s_lon in fx.columns else np.nan
        fx["venue_capacity"] = pd.to_numeric(fx[s_cap], errors="coerce") if s_cap and s_cap in fx.columns else np.nan
    else:
        fx["venue_latitude"] = np.nan
        fx["venue_longitude"] = np.nan
        fx["venue_capacity"] = np.nan

    # Host flags
    fx["home_host_flag"] = fx["home_team"].isin(HOST_TEAMS).astype(int)
    fx["away_host_flag"] = fx["away_team"].isin(HOST_TEAMS).astype(int)
    fx["is_host_match"] = ((fx["home_host_flag"] + fx["away_host_flag"]) > 0).astype(int)

    # Travel intensity proxy by venue region
    region_score = {"West": 2.0, "Central": 1.0, "East": 0.0, "Unknown": 1.0}
    fx["travel_intensity_proxy"] = fx["venue_region"].map(region_score).fillna(1.0)

    # Schedule ordering helpers
    if fx["date_et"].notna().any():
        fx["days_from_start"] = (fx["date_et"] - fx["date_et"].min()).dt.days
    else:
        fx["days_from_start"] = np.nan

    # Clean final output
    keep = [
        "match_id",
        "stage",
        "stage_id",
        "group",
        "home_team",
        "away_team",
        "date_et",
        "time_et",
        "days_from_start",
        "venue_stadium",
        "venue_city",
        "venue_country",
        "venue_region",
        "venue_latitude",
        "venue_longitude",
        "venue_capacity",
        "home_host_flag",
        "away_host_flag",
        "is_host_match",
        "travel_intensity_proxy",
    ]
    keep = [c for c in keep if c in fx.columns]
    fx = fx[keep].sort_values(
        by=[c for c in ["date_et", "time_et", "stage_id", "match_id"] if c in fx.columns]
    ).reset_index(drop=True)

    out = PROCESSED / "fixture_features.csv"
    fx.to_csv(out, index=False)
    print(f"Saved {out} | shape={fx.shape}")
    print(fx.head(10).to_string(index=False))
    return fx

if __name__ == "__main__":
    build_fixture_features()