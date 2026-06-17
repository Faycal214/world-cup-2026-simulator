from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from config import RAW_DIR, PROCESSED_DIR, HOST_TEAMS
from naming import normalize_team_name

START_DATE = pd.Timestamp("2023-01-01")
END_DATE = pd.Timestamp("2026-06-10")
HOLDOUT_FRACTION = 0.15
HALF_LIFE_DAYS = 540.0
RANDOM_SEED = 42

SPECIAL_SCENARIO_BIAS = {
    "Argentina": 0.10,
    "Mexico": 0.05,
    "Canada": 0.05,
    "United States": 0.05,
}

TEAM_TRAVEL_PENALTY = {
    "Iran": 0.35,
}

def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)

def _canon_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        re.sub(r"[^a-z0-9]+", "_", str(c).lower()).strip("_")
        for c in df.columns
    ]
    return df

def _get_bool_series(df: pd.DataFrame, col: str, default: bool = False) -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index)
    s = df[col]
    if s.dtype == bool:
        return s.fillna(default)
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y", "t"])
        .fillna(default)
    )

def _time_decay(days_ago: float) -> float:
    if pd.isna(days_ago):
        return 1.0
    return float(np.exp(-np.log(2.0) * days_ago / HALF_LIFE_DAYS))

def _standardize(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(0.0, index=s.index)
    s = s.fillna(s.median())
    std = s.std(ddof=0)
    if std == 0 or not np.isfinite(std):
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std

def _poisson_draw_prob(lam_home: float, lam_away: float, k_max: int = 10) -> float:
    if not np.isfinite(lam_home) or not np.isfinite(lam_away):
        return np.nan
    p = 0.0
    for k in range(k_max + 1):
        p += (
            np.exp(-lam_home) * lam_home**k / factorial(k)
            * np.exp(-lam_away) * lam_away**k / factorial(k)
        )
    return float(p)

def competition_weight_from_name(tournament: str) -> float:
    s = str(tournament).strip()

    exact = {
        "Friendly": 0.30,
        "FIFA World Cup qualification": 0.75,
        "UEFA Euro qualification": 0.75,
        "UEFA Nations League": 0.60,
        "CONCACAF Nations League": 0.60,
        "CONMEBOL Nations League": 0.60,
        "CONMEBOL Copa América": 0.95,
        "Copa América": 0.95,
        "AFC Asian Cup": 0.90,
        "AFC Asian Cup qualification": 0.75,
        "CAF Africa Cup of Nations": 0.90,
        "CAF Africa Cup of Nations qualification": 0.75,
        "OFC Nations Cup": 0.85,
        "FIFA World Cup": 1.00,
        "FIFA World Cup qualification": 0.75,
    }

    if s in exact:
        return exact[s]

    s_low = s.lower()
    fallback = [
        (r"world cup qualification", 0.75),
        (r"world cup", 1.00),
        (r"euro qualification", 0.75),
        (r"euro", 0.95),
        (r"nations league", 0.60),
        (r"copa am[eé]rica", 0.95),
        (r"africa cup of nations|afcon", 0.90),
        (r"asian cup", 0.90),
        (r"friendly", 0.30),
    ]
    for pattern, weight in fallback:
        if re.search(pattern, s_low):
            return weight

    return 0.70

def load_team_universe() -> pd.DataFrame:
    team_features = _canon_cols(_load_csv(PROCESSED_DIR / "team_features.csv"))
    if "team" not in team_features.columns:
        raise ValueError("team_features.csv must contain a 'team' column")
    team_features["team"] = team_features["team"].map(normalize_team_name)
    return team_features

def load_matches() -> pd.DataFrame:
    filtered_path = PROCESSED_DIR / "filtered_wc_matches.csv"
    raw_path = RAW_DIR / "matches.csv"

    if filtered_path.exists():
        matches = _canon_cols(_load_csv(filtered_path))
    else:
        matches = _canon_cols(_load_csv(raw_path))

    required = {"date", "home_team", "away_team", "home_score", "away_score", "tournament"}
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"matches file is missing columns: {sorted(missing)}")

    matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
    matches["home_team"] = matches["home_team"].map(normalize_team_name)
    matches["away_team"] = matches["away_team"].map(normalize_team_name)
    matches["home_score"] = pd.to_numeric(matches["home_score"], errors="coerce")
    matches["away_score"] = pd.to_numeric(matches["away_score"], errors="coerce")
    matches["neutral"] = _get_bool_series(matches, "neutral", default=False)
    matches["tournament"] = matches["tournament"].astype(str).str.strip()

    matches = matches.loc[matches["date"].notna()].copy()
    matches = matches.loc[(matches["date"] >= START_DATE) & (matches["date"] <= END_DATE)].copy()
    matches = matches.reset_index(drop=True)
    return matches

def build_competition_inventory(matches: pd.DataFrame) -> pd.DataFrame:
    competitions = (
        matches["tournament"]
        .astype(str)
        .str.strip()
        .loc[lambda s: s.ne("") & s.ne("nan")]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )
    inv = pd.DataFrame({"tournament": competitions})
    inv["weight"] = inv["tournament"].map(competition_weight_from_name)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    inv.to_csv(PROCESSED_DIR / "competition_weights.csv", index=False)
    return inv

def build_long_match_table(matches: pd.DataFrame, team_universe: pd.DataFrame) -> pd.DataFrame:
    team_set = set(team_universe["team"].tolist())

    matches = matches.loc[
        matches["home_team"].isin(team_set) & matches["away_team"].isin(team_set)
    ].copy()

    if matches.empty:
        raise ValueError("No matches remain after filtering to the 48-team universe.")

    snapshot = matches["date"].max()
    rows = []

    for idx, m in matches.reset_index(drop=True).iterrows():
        comp_w = competition_weight_from_name(m.get("tournament", ""))
        days_ago = (snapshot - m["date"]).days
        decay = _time_decay(days_ago)
        weight = comp_w * decay

        # shared id for both rows
        game_id = f'{m["date"]:%Y%m%d}_{idx}_{m["home_team"]}_{m["away_team"]}'

        rows.append(
            {
                "game_id": game_id,
                "date": m["date"],
                "team": m["home_team"],
                "opponent": m["away_team"],
                "goals": float(m["home_score"]),
                "is_home": 1,
                "neutral": int(bool(m["neutral"])),
                "home_adv": int((not bool(m["neutral"]))),
                "weight": weight,
                "tournament": m.get("tournament", "Unknown"),
                "is_host_team": int(m["home_team"] in HOST_TEAMS),
            }
        )

        rows.append(
            {
                "game_id": game_id,
                "date": m["date"],
                "team": m["away_team"],
                "opponent": m["home_team"],
                "goals": float(m["away_score"]),
                "is_home": 0,
                "neutral": int(bool(m["neutral"])),
                "home_adv": 0,
                "weight": weight,
                "tournament": m.get("tournament", "Unknown"),
                "is_host_team": int(m["away_team"] in HOST_TEAMS),
            }
        )

    long_df = pd.DataFrame(rows)
    for c in ["weight", "goals", "is_home", "home_adv", "neutral", "is_host_team"]:
        long_df[c] = pd.to_numeric(long_df[c], errors="coerce").fillna(0.0)

    return long_df

@dataclass
class StrengthModelResult:
    fit: object
    train_long: pd.DataFrame
    valid_long: pd.DataFrame
    all_teams: list[str]

def fit_strength_model() -> StrengthModelResult:
    team_universe = load_team_universe()
    matches = load_matches()

    build_competition_inventory(matches)

    team_set = set(team_universe["team"].tolist())

    # Filter to the 48-team universe first, then split by match.
    matches = matches.loc[
        matches["home_team"].isin(team_set) & matches["away_team"].isin(team_set)
    ].copy()

    if matches.empty:
        raise ValueError("No matches remain after filtering to the 48-team universe.")

    # Match-level split, not long-row split
    split_date = matches["date"].quantile(1.0 - HOLDOUT_FRACTION)
    train_matches = matches.loc[matches["date"] <= split_date].copy()
    valid_matches = matches.loc[matches["date"] > split_date].copy()

    if train_matches.empty or valid_matches.empty:
        raise ValueError("Train/validation split produced an empty side. Check date coverage.")

    train_long = build_long_match_table(train_matches, team_universe)
    valid_long = build_long_match_table(valid_matches, team_universe)

    formula = "goals ~ home_adv + neutral + C(team) + C(opponent)"

    model = smf.glm(
        formula=formula,
        data=train_long,
        family=sm.families.Poisson(),
        freq_weights=train_long["weight"],
    )
    fit = model.fit(cov_type="HC3")

    all_teams = team_universe["team"].tolist()
    return StrengthModelResult(fit=fit, train_long=train_long, valid_long=valid_long, all_teams=all_teams)

def _coef_lookup(params: pd.Series, term: str) -> float:
    return float(params.get(term, 0.0))

def build_team_strength_priors(fit_result: StrengthModelResult) -> pd.DataFrame:
    params = fit_result.fit.params
    teams = fit_result.all_teams

    attack_raw = []
    defense_raw = []

    for team in teams:
        attack_term = f"C(team)[T.{team}]"
        opp_term = f"C(opponent)[T.{team}]"
        attack_raw.append(_coef_lookup(params, attack_term))
        defense_raw.append(-_coef_lookup(params, opp_term))

    out = pd.DataFrame(
        {
            "team": teams,
            "attack_raw": attack_raw,
            "defense_raw": defense_raw,
        }
    )

    out["attack_prior"] = _standardize(out["attack_raw"])
    out["defense_prior"] = _standardize(out["defense_raw"])

    team_features = _canon_cols(_load_csv(PROCESSED_DIR / "team_features.csv"))
    team_features["team"] = team_features["team"].map(normalize_team_name)

    if "elo" not in team_features.columns:
        team_features["elo"] = np.nan
    if "market_value_eur" not in team_features.columns:
        team_features["market_value_eur"] = np.nan
    if "form_pts" not in team_features.columns:
        team_features["form_pts"] = np.nan
    if "experience_index" not in team_features.columns:
        team_features["experience_index"] = np.nan
    if "host_flag" not in team_features.columns:
        team_features["host_flag"] = 0

    team_features["elo_prior"] = _standardize(team_features["elo"])
    team_features["market_prior"] = _standardize(np.log1p(pd.to_numeric(team_features["market_value_eur"], errors="coerce").fillna(0.0)))
    team_features["form_prior"] = _standardize(team_features["form_pts"])
    team_features["experience_prior"] = _standardize(team_features["experience_index"])

    priors = (
        team_features[
            [
                "team",
                "elo_prior",
                "market_prior",
                "form_prior",
                "experience_prior",
                "host_flag",
            ]
        ]
        .merge(out[["team", "attack_prior", "defense_prior"]], on="team", how="left")
    )

    priors["host_prior"] = pd.to_numeric(priors["host_flag"], errors="coerce").fillna(0.0)
    priors["scenario_bias_prior"] = priors["team"].map(SPECIAL_SCENARIO_BIAS).fillna(0.0)
    priors["travel_prior"] = priors["team"].map(TEAM_TRAVEL_PENALTY).fillna(0.0)

    for c in [
        "attack_prior", "defense_prior", "elo_prior", "market_prior",
        "form_prior", "experience_prior", "host_prior",
        "scenario_bias_prior", "travel_prior",
    ]:
        priors[c] = pd.to_numeric(priors[c], errors="coerce").fillna(0.0)

    priors["attack_prior"] = priors["attack_prior"].clip(-2.5, 2.5)
    priors["defense_prior"] = priors["defense_prior"].clip(-2.5, 2.5)

    priors = priors[
        [
            "team",
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
    ].sort_values("team").reset_index(drop=True)

    return priors

def validate_model(fit_result: StrengthModelResult) -> dict:
    fit = fit_result.fit
    valid_long = fit_result.valid_long.copy()

    if valid_long.empty:
        return {"warning": "No validation rows available after split."}

    valid_long["mu"] = fit.predict(valid_long)

    home_rows = valid_long.loc[
        valid_long["is_home"] == 1,
        ["game_id", "date", "goals", "mu"]
    ].rename(columns={"goals": "home_goals", "mu": "home_mu"})

    away_rows = valid_long.loc[
        valid_long["is_home"] == 0,
        ["game_id", "date", "goals", "mu"]
    ].rename(columns={"goals": "away_goals", "mu": "away_mu"})

    m = home_rows.merge(away_rows, on="game_id", how="inner")

    if m.empty:
        return {
            "warning": "No paired validation matches available.",
            "validation_rows": int(len(valid_long)),
            "home_rows": int(len(home_rows)),
            "away_rows": int(len(away_rows)),
        }

    actual_draw_rate = float((m["home_goals"] == m["away_goals"]).mean())
    pred_draw_rate = float(np.nanmean([
        _poisson_draw_prob(h, a) for h, a in zip(m["home_mu"], m["away_mu"])
    ]))

    actual_winner = np.where(
        m["home_goals"] > m["away_goals"], "home",
        np.where(m["away_goals"] > m["home_goals"], "away", "draw"),
    )
    pred_fav = np.where(
        m["home_mu"] > m["away_mu"], "home",
        np.where(m["away_mu"] > m["home_mu"], "away", "draw"),
    )

    upset_mask = (
        (pred_fav == "home") & (actual_winner == "away")
    ) | (
        (pred_fav == "away") & (actual_winner == "home")
    )

    avg_actual_goals = float((m["home_goals"] + m["away_goals"]).mean())
    avg_pred_goals = float((m["home_mu"] + m["away_mu"]).mean())
    
    eps = 1e-10

    y = (m["home_goals"] + m["away_goals"]).values
    mu = np.maximum((m["home_mu"] + m["away_mu"]).values, eps)

    poisson_dev = 2 * np.sum(
        np.where(
            y == 0,
            -mu,
            y * np.log(y / mu) - (y - mu)
        )
    )

    return {
        "validation_matches": int(len(m)),
        "avg_actual_goals_per_match": avg_actual_goals,
        "avg_predicted_goals_per_match": avg_pred_goals,
        "actual_draw_rate": actual_draw_rate,
        "predicted_draw_rate": pred_draw_rate,
        "upset_rate": float(upset_mask.mean()),
        "home_goal_mae": float(np.abs(m["home_goals"] - m["home_mu"]).mean()),
        "away_goal_mae": float(np.abs(m["away_goals"] - m["away_mu"]).mean()),
        "poisson_deviance": float(poisson_dev),
    }

def sample_asymptotic_posterior(fit_result: StrengthModelResult, n_draws: int = 1000, seed: int = RANDOM_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    params = fit_result.fit.params.values
    cov = fit_result.fit.cov_params().values
    return rng.multivariate_normal(params, cov, size=n_draws)

def run_calibration() -> tuple[pd.DataFrame, dict]:
    fit_result = fit_strength_model()
    priors = build_team_strength_priors(fit_result)
    metrics = validate_model(fit_result)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    priors.to_csv(PROCESSED_DIR / "team_strength_priors.csv", index=False)

    with open(PROCESSED_DIR / "calibration_report.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    return priors, metrics

if __name__ == "__main__":
    priors, metrics = run_calibration()
    print(priors.head(10).to_string(index=False))
    print(metrics)