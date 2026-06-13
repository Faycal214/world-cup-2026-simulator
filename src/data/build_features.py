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

SNAPSHOT_DATE = pd.Timestamp("2026-06-01")
HOST_TEAMS = {"Mexico", "Canada", "USA"}

SCENARIO_BIAS = {
    "Argentina": 0.0,
    "Iran": 0.0,
    "Mexico": 0.0,
    "Canada": 0.0,
    "USA": 0.0,
}

def canon_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        re.sub(r"[^a-z0-9]+", "_", unidecode(str(c).lower())).strip("_")
        for c in df.columns
    ]
    return df

def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None

def clean_team(x) -> str:
    if pd.isna(x):
        return ""
    s = unidecode(str(x))
    s = re.sub(r"\s*\([A-Z]{3}\)\s*$", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def parse_percent(x):
    if pd.isna(x):
        return np.nan
    s = unidecode(str(x)).strip().replace("%", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return np.nan

def parse_market_value(x):
    if pd.isna(x):
        return np.nan
    s = unidecode(str(x)).lower().strip()
    s = s.replace("€", "").replace("$", "").replace(" ", "").replace(",", ".")
    mult = 1.0
    if s.endswith("bn"):
        mult = 1e9
        s = s[:-2]
    elif s.endswith("m"):
        mult = 1e6
        s = s[:-1]
    elif s.endswith("k"):
        mult = 1e3
        s = s[:-1]
    s = re.sub(r"[^0-9.]+", "", s)
    if not s:
        return np.nan
    try:
        return float(s) * mult
    except Exception:
        return np.nan

def parse_date(x):
    return pd.to_datetime(x, dayfirst=True, errors="coerce")

def load_csv(name: str) -> pd.DataFrame:
    path = RAW / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)

def build_team_features():
    official = canon_cols(load_csv("official_squads.csv"))
    team_values = canon_cols(load_csv("worldcup_team_values.csv"))
    matches = canon_cols(load_csv("matches.csv"))
    elo = canon_cols(load_csv("elo_history.csv"))

    team_col = pick_col(official, ["team", "country"])
    player_col = pick_col(official, ["player", "player_raw", "name"])
    pos_col = pick_col(official, ["position"])
    dob_col = pick_col(official, ["dob", "date_of_birth"])
    club_col = pick_col(official, ["club"])
    caps_col = pick_col(official, ["caps"])
    goals_col = pick_col(official, ["goals"])
    height_col = pick_col(official, ["height_cm", "height"])

    if team_col is None or player_col is None:
        raise ValueError("official_squads.csv must contain team and player columns")

    official = official.copy()
    official["team"] = official[team_col].map(clean_team)
    official["player"] = official[player_col].astype(str).str.strip()
    official["position"] = official[pos_col].astype(str).str.upper().str.strip() if pos_col else ""
    official["dob"] = parse_date(official[dob_col]) if dob_col else pd.NaT
    official["age"] = ((SNAPSHOT_DATE - official["dob"]).dt.days / 365.25).round(2)
    official["caps"] = pd.to_numeric(official[caps_col], errors="coerce") if caps_col else np.nan
    official["goals"] = pd.to_numeric(official[goals_col], errors="coerce") if goals_col else np.nan
    official["height_cm"] = pd.to_numeric(official[height_col], errors="coerce") if height_col else np.nan
    official["is_gk"] = (official["position"] == "GK").astype(int)
    official["is_df"] = (official["position"] == "DF").astype(int)
    official["is_mf"] = (official["position"] == "MF").astype(int)
    official["is_fw"] = (official["position"] == "FW").astype(int)

    team_features = (
        official.groupby("team", as_index=False)
        .agg(
            squad_size=("player", "count"),
            avg_age=("age", "mean"),
            median_age=("age", "median"),
            age_std=("age", "std"),
            avg_caps=("caps", "mean"),
            total_caps=("caps", "sum"),
            avg_goals=("goals", "mean"),
            total_goals=("goals", "sum"),
            avg_height_cm=("height_cm", "mean"),
            height_std_cm=("height_cm", "std"),
            gk_count=("is_gk", "sum"),
            df_count=("is_df", "sum"),
            mf_count=("is_mf", "sum"),
            fw_count=("is_fw", "sum"),
        )
        .copy()
    )

    team_features["gk_share"] = team_features["gk_count"] / team_features["squad_size"].clip(lower=1)
    team_features["df_share"] = team_features["df_count"] / team_features["squad_size"].clip(lower=1)
    team_features["mf_share"] = team_features["mf_count"] / team_features["squad_size"].clip(lower=1)
    team_features["fw_share"] = team_features["fw_count"] / team_features["squad_size"].clip(lower=1)
    team_features["attack_depth"] = team_features["fw_count"] + 0.5 * team_features["mf_count"]
    team_features["defense_depth"] = team_features["df_count"] + 1.5 * team_features["gk_count"]
    team_features["experience_index"] = 0.5 * team_features["avg_caps"].fillna(0) + 0.1 * team_features["avg_goals"].fillna(0)

    # Team value file
    tv_team_col = pick_col(team_values, ["team", "club"])
    tv_squad_col = pick_col(team_values, ["squad_size", "squad"])
    tv_avg_age_col = pick_col(team_values, ["avg_age", "o_age"])
    tv_wc_col = pick_col(team_values, ["wc_participations", "wc_particip"])
    tv_foreign_col = pick_col(team_values, ["foreigners_pct", "foreigners"])
    tv_market_col = pick_col(team_values, ["market_value", "total_market_value"])
    tv_avg_market_col = pick_col(team_values, ["avg_market_value", "average_market_value", "o_market_value"])

    if tv_team_col is None or tv_market_col is None:
        raise ValueError("worldcup_team_values.csv must contain team/club and market value columns")

    tv = team_values.copy()
    tv["team"] = tv[tv_team_col].map(clean_team)
    tv["tm_squad_size"] = pd.to_numeric(tv[tv_squad_col], errors="coerce") if tv_squad_col else np.nan
    tv["tm_avg_age"] = pd.to_numeric(tv[tv_avg_age_col], errors="coerce") if tv_avg_age_col else np.nan
    tv["wc_participations"] = pd.to_numeric(tv[tv_wc_col], errors="coerce") if tv_wc_col else np.nan
    tv["foreigners_pct"] = tv[tv_foreign_col].map(parse_percent) if tv_foreign_col else np.nan
    tv["market_value_eur"] = tv[tv_market_col].map(parse_market_value)
    tv["avg_market_value_eur"] = tv[tv_avg_market_col].map(parse_market_value) if tv_avg_market_col else np.nan

    team_features = team_features.merge(
        tv[[
            "team",
            "tm_squad_size",
            "tm_avg_age",
            "wc_participations",
            "foreigners_pct",
            "market_value_eur",
            "avg_market_value_eur",
        ]],
        on="team",
        how="left",
    )

    # Elo history: latest pre-tournament rating per team
    elo = elo.copy()
    elo_team_col = pick_col(elo, ["team", "country", "national_team"])
    elo_value_col = pick_col(elo, ["elo", "rating"])
    elo_date_col = pick_col(elo, ["date"])

    if elo_team_col and elo_value_col:
        elo["team"] = elo[elo_team_col].map(clean_team)
        elo["elo"] = pd.to_numeric(elo[elo_value_col], errors="coerce")
        if elo_date_col:
            elo["date"] = pd.to_datetime(elo[elo_date_col], errors="coerce")
            elo = elo.sort_values(["team", "date"])
            elo_latest = elo.groupby("team", as_index=False).tail(1)[["team", "elo"]]
        else:
            elo_latest = elo.groupby("team", as_index=False)["elo"].last()
        team_features = team_features.merge(elo_latest, on="team", how="left")
    else:
        team_features["elo"] = np.nan

    # Historical form from matches.csv
    # Expect columns like date, home_team, away_team, home_score, away_score
    m_date = pick_col(matches, ["date"])
    m_home = pick_col(matches, ["home_team"])
    m_away = pick_col(matches, ["away_team"])
    m_hs = pick_col(matches, ["home_score"])
    m_as = pick_col(matches, ["away_score"])
    m_tourn = pick_col(matches, ["tournament"])

    if all(c is not None for c in [m_date, m_home, m_away, m_hs, m_as]):
        mm = matches.copy()
        mm["date"] = pd.to_datetime(mm[m_date], errors="coerce")
        mm["home_team"] = mm[m_home].map(clean_team)
        mm["away_team"] = mm[m_away].map(clean_team)
        mm["home_score"] = pd.to_numeric(mm[m_hs], errors="coerce")
        mm["away_score"] = pd.to_numeric(mm[m_as], errors="coerce")
        mm["tournament"] = mm[m_tourn].astype(str) if m_tourn else "Unknown"

        # recent form: last 10 matches per team with time decay
        cutoff = pd.Timestamp("2022-01-01")
        mm = mm[mm["date"] >= cutoff].copy()
        mm["days_ago"] = (SNAPSHOT_DATE - mm["date"]).dt.days.clip(lower=0)
        mm["time_decay"] = np.exp(-mm["days_ago"] / 365.25)

        home_pts = np.where(mm["home_score"] > mm["away_score"], 3, np.where(mm["home_score"] == mm["away_score"], 1, 0))
        away_pts = np.where(mm["away_score"] > mm["home_score"], 3, np.where(mm["away_score"] == mm["home_score"], 1, 0))

        home = pd.DataFrame({
            "team": mm["home_team"],
            "gf": mm["home_score"],
            "ga": mm["away_score"],
            "pts": home_pts,
            "w": mm["time_decay"],
        })
        away = pd.DataFrame({
            "team": mm["away_team"],
            "gf": mm["away_score"],
            "ga": mm["home_score"],
            "pts": away_pts,
            "w": mm["time_decay"],
        })

        form = pd.concat([home, away], ignore_index=True)
        form["weighted_gf"] = form["gf"] * form["w"]
        form["weighted_ga"] = form["ga"] * form["w"]
        form["weighted_pts"] = form["pts"] * form["w"]

        team_form = (
            form.groupby("team", as_index=False)
            .agg(
                form_gf=("weighted_gf", "sum"),
                form_ga=("weighted_ga", "sum"),
                form_pts=("weighted_pts", "sum"),
            )
        )
        team_features = team_features.merge(team_form, on="team", how="left")
    else:
        team_features["form_gf"] = np.nan
        team_features["form_ga"] = np.nan
        team_features["form_pts"] = np.nan

    # Tournament context
    team_features["host_flag"] = team_features["team"].isin(HOST_TEAMS).astype(int)
    team_features["scenario_bias_prior"] = team_features["team"].map(SCENARIO_BIAS).fillna(0.0)

    # Strength-ready transforms
    team_features["log_market_value"] = np.log1p(team_features["market_value_eur"])
    team_features["market_value_per_player"] = team_features["market_value_eur"] / team_features["squad_size"].clip(lower=1)
    team_features["caps_per_player"] = team_features["total_caps"] / team_features["squad_size"].clip(lower=1)
    team_features["goals_per_player"] = team_features["total_goals"] / team_features["squad_size"].clip(lower=1)

    cols = [
        "team",
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
        "host_flag",
        "scenario_bias_prior",
        "log_market_value",
        "market_value_per_player",
        "caps_per_player",
        "goals_per_player",
    ]
    cols = [c for c in cols if c in team_features.columns]
    team_features = team_features[cols].sort_values("team").reset_index(drop=True)

    out = PROCESSED / "team_features.csv"
    team_features.to_csv(out, index=False)
    print(f"Saved {out} | shape={team_features.shape}")
    return team_features

if __name__ == "__main__":
    build_team_features()