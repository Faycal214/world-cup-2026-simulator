from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import pytensor.tensor as pt

from config import RAW_DIR, PROCESSED_DIR, HOST_TEAMS
from naming import normalize_team_name

START_DATE = pd.Timestamp("2023-01-01")
END_DATE = pd.Timestamp("2026-06-10")
HOLDOUT_FRACTION = 0.15
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
    return float(np.exp(-np.log(2.0) * days_ago / 540.0))


def _competition_weight(tournament: str) -> float:
    s = str(tournament).strip()
    if not s:
        return 0.70

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


def _standardize(arr: pd.Series | np.ndarray) -> np.ndarray:
    s = pd.Series(arr).astype(float)
    if s.notna().sum() == 0:
        return np.zeros(len(s), dtype=float)
    s = s.fillna(s.median())
    std = s.std(ddof=0)
    if std == 0 or not np.isfinite(std):
        return np.zeros(len(s), dtype=float)
    return ((s - s.mean()) / std).to_numpy()


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


def load_team_universe() -> pd.DataFrame:
    team_features = _canon_cols(_load_csv(PROCESSED_DIR / "team_features.csv"))
    if "team" not in team_features.columns:
        raise ValueError("team_features.csv must contain a 'team' column")
    team_features["team"] = team_features["team"].map(normalize_team_name)
    team_features = team_features.sort_values("team").reset_index(drop=True)
    return team_features


def load_matches() -> pd.DataFrame:
    filtered_path = PROCESSED_DIR / "filtered_wc_matches.csv"
    raw_path = RAW_DIR / "matches.csv"

    matches = _canon_cols(_load_csv(filtered_path if filtered_path.exists() else raw_path))

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
    inv["weight"] = inv["tournament"].map(_competition_weight)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    inv.to_csv(PROCESSED_DIR / "competition_weights.csv", index=False)
    pd.DataFrame({"tournament": competitions}).to_csv(
        PROCESSED_DIR / "competition_names.csv",
        index=False,
    )
    return inv


def build_long_match_table(
    matches: pd.DataFrame,
    team_universe: pd.DataFrame,
    competition_inventory: pd.DataFrame,
) -> pd.DataFrame:
    team_set = set(team_universe["team"].tolist())
    matches = matches.loc[
        matches["home_team"].isin(team_set) & matches["away_team"].isin(team_set)
    ].copy()

    if matches.empty:
        raise ValueError("No matches remain after filtering to the 48-team universe.")

    snapshot = matches["date"].max()
    team_index = {team: i for i, team in enumerate(team_universe["team"].tolist())}
    comp_index = {c: i for i, c in enumerate(competition_inventory["tournament"].tolist())}

    rows = []
    for idx, m in matches.reset_index(drop=True).iterrows():
        comp_name = str(m.get("tournament", "Unknown")).strip()
        comp_w = _competition_weight(comp_name)
        days_ago = (snapshot - m["date"]).days
        decay = _time_decay(days_ago)
        weight = comp_w * decay

        game_id = f'{m["date"]:%Y%m%d}_{idx}_{m["home_team"]}_{m["away_team"]}'
        comp_idx = comp_index.get(comp_name, 0)

        rows.append(
            {
                "game_id": game_id,
                "date": m["date"],
                "team": m["home_team"],
                "opponent": m["away_team"],
                "team_idx": team_index[m["home_team"]],
                "opp_idx": team_index[m["away_team"]],
                "comp_idx": comp_idx,
                "goals": float(m["home_score"]),
                "is_home": 1,
                "neutral": int(bool(m["neutral"])),
                "home_adv": int(not bool(m["neutral"])),
                "weight": weight,
                "tournament": comp_name,
                "is_host_team": int(m["home_team"] in HOST_TEAMS),
            }
        )
        rows.append(
            {
                "game_id": game_id,
                "date": m["date"],
                "team": m["away_team"],
                "opponent": m["home_team"],
                "team_idx": team_index[m["away_team"]],
                "opp_idx": team_index[m["home_team"]],
                "comp_idx": comp_idx,
                "goals": float(m["away_score"]),
                "is_home": 0,
                "neutral": int(bool(m["neutral"])),
                "home_adv": 0,
                "weight": weight,
                "tournament": comp_name,
                "is_host_team": int(m["away_team"] in HOST_TEAMS),
            }
        )

    long_df = pd.DataFrame(rows)
    for c in ["weight", "goals", "is_home", "home_adv", "neutral", "is_host_team", "team_idx", "opp_idx", "comp_idx"]:
        long_df[c] = pd.to_numeric(long_df[c], errors="coerce").fillna(0).astype(int if c in ["is_home", "home_adv", "neutral", "is_host_team", "team_idx", "opp_idx", "comp_idx"] else float)

    long_df["weight"] = pd.to_numeric(long_df["weight"], errors="coerce").fillna(1.0)
    long_df["goals"] = pd.to_numeric(long_df["goals"], errors="coerce").fillna(0.0)
    return long_df


@dataclass
class BayesianFitResult:
    idata: az.InferenceData
    team_order: list[str]
    competition_order: list[str]
    train_long: pd.DataFrame
    valid_long: pd.DataFrame
    metrics: dict


def fit_bayesian_strength_model(
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 2,
    cores: int = 2,
    target_accept: float = 0.92,
    seed: int = RANDOM_SEED,
) -> BayesianFitResult:
    team_universe = load_team_universe()
    matches = load_matches()
    competition_inventory = build_competition_inventory(matches)

    team_set = set(team_universe["team"].tolist())
    matches = matches.loc[
        matches["home_team"].isin(team_set) & matches["away_team"].isin(team_set)
    ].copy()

    if matches.empty:
        raise ValueError("No matches remain after filtering to the 48-team universe.")

    split_date = matches["date"].quantile(1.0 - HOLDOUT_FRACTION)
    train_matches = matches.loc[matches["date"] <= split_date].copy()
    valid_matches = matches.loc[matches["date"] > split_date].copy()

    if train_matches.empty or valid_matches.empty:
        raise ValueError("Train/validation split produced an empty side. Check date coverage.")

    train_long = build_long_match_table(train_matches, team_universe, competition_inventory)
    valid_long = build_long_match_table(valid_matches, team_universe, competition_inventory)

    team_order = team_universe["team"].tolist()
    competition_order = competition_inventory["tournament"].tolist()

    coords = {
        "team": team_order,
        "competition": competition_order,
        "obs_id": np.arange(len(train_long)),
    }

    with pm.Model(coords=coords) as model:
        team_idx = pm.Data("team_idx", train_long["team_idx"].values, dims="obs_id")
        opp_idx = pm.Data("opp_idx", train_long["opp_idx"].values, dims="obs_id")
        comp_idx = pm.Data("comp_idx", train_long["comp_idx"].values, dims="obs_id")
        home_adv = pm.Data("home_adv", train_long["home_adv"].values, dims="obs_id")
        weight = pm.Data("weight", train_long["weight"].values, dims="obs_id")
        goals = pm.Data("goals", train_long["goals"].values, dims="obs_id")

        sigma_attack = pm.HalfNormal("sigma_attack", sigma=1.0)
        sigma_defense = pm.HalfNormal("sigma_defense", sigma=1.0)
        sigma_comp = pm.HalfNormal("sigma_comp", sigma=0.4)

        attack_raw = pm.Normal("attack_raw", mu=0.0, sigma=sigma_attack, dims="team")
        defense_raw = pm.Normal("defense_raw", mu=0.0, sigma=sigma_defense, dims="team")
        comp_raw = pm.Normal("competition_raw", mu=0.0, sigma=sigma_comp, dims="competition")

        attack = pm.Deterministic("attack", attack_raw - pt.mean(attack_raw), dims="team")
        defense = pm.Deterministic("defense", defense_raw - pt.mean(defense_raw), dims="team")
        comp_effect = pm.Deterministic("competition_effect", comp_raw - pt.mean(comp_raw), dims="competition")

        intercept = pm.Normal("intercept", mu=np.log(1.28), sigma=0.45)
        beta_home = pm.Normal("beta_home", mu=0.12, sigma=0.18)

        linear = (
            intercept
            + beta_home * home_adv
            + attack[team_idx]
            + defense[opp_idx]
            + comp_effect[comp_idx]
        )
        mu = pm.Deterministic("mu", pt.exp(linear), dims="obs_id")

        logp = pm.logp(pm.Poisson.dist(mu=mu), goals)
        pm.Potential("weighted_loglike", weight * logp)

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            random_seed=seed,
            return_inferencedata=True,
            progressbar=True,
            nuts={"target_accept": target_accept},
        )

    metrics = validate_bayesian_model(idata, valid_long, team_order, competition_order)

    return BayesianFitResult(
        idata=idata,
        team_order=team_order,
        competition_order=competition_order,
        train_long=train_long,
        valid_long=valid_long,
        metrics=metrics,
    )


def validate_bayesian_model(
    idata: az.InferenceData,
    valid_long: pd.DataFrame,
    team_order: list[str],
    competition_order: list[str],
) -> dict:
    if valid_long.empty:
        return {"warning": "No validation rows available after split."}

    posterior = idata.posterior.mean(dim=("chain", "draw"))

    attack_mean = np.asarray(posterior["attack"].values)
    defense_mean = np.asarray(posterior["defense"].values)
    comp_mean = np.asarray(posterior["competition_effect"].values)
    intercept = float(posterior["intercept"].values)
    beta_home = float(posterior["beta_home"].values)

    team_to_idx = {t: i for i, t in enumerate(team_order)}

    valid = valid_long.copy()
    valid["mu"] = np.exp(
        intercept
        + beta_home * valid["home_adv"].astype(float).values
        + attack_mean[valid["team_idx"].astype(int).values]
        + defense_mean[valid["opp_idx"].astype(int).values]
        + comp_mean[valid["comp_idx"].astype(int).values]
    )

    home_rows = valid.loc[valid["is_home"] == 1, ["game_id", "date", "goals", "mu"]].rename(
        columns={"goals": "home_goals", "mu": "home_mu"}
    )
    away_rows = valid.loc[valid["is_home"] == 0, ["game_id", "date", "goals", "mu"]].rename(
        columns={"goals": "away_goals", "mu": "away_mu"}
    )

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

    return {
        "validation_matches": int(len(m)),
        "avg_actual_goals_per_match": avg_actual_goals,
        "avg_predicted_goals_per_match": avg_pred_goals,
        "actual_draw_rate": actual_draw_rate,
        "predicted_draw_rate": pred_draw_rate,
        "upset_rate": float(upset_mask.mean()),
        "home_goal_mae": float(np.abs(m["home_goals"] - m["home_mu"]).mean()),
        "away_goal_mae": float(np.abs(m["away_goals"] - m["away_mu"]).mean()),
    }


def posterior_summary_table(
    idata: az.InferenceData,
    team_universe: pd.DataFrame,
    team_order: list[str],
) -> pd.DataFrame:
    post = idata.posterior
    attack_mean = post["attack"].mean(dim=("chain", "draw")).values
    attack_sd = post["attack"].std(dim=("chain", "draw")).values
    defense_mean = post["defense"].mean(dim=("chain", "draw")).values
    defense_sd = post["defense"].std(dim=("chain", "draw")).values

    base = team_universe.copy()
    base["team"] = base["team"].map(normalize_team_name)
    base = base.set_index("team").loc[team_order].reset_index()

    base["attack_posterior_mean"] = attack_mean
    base["attack_posterior_sd"] = attack_sd
    base["defense_posterior_mean"] = defense_mean
    base["defense_posterior_sd"] = defense_sd

    base["attack_prior"] = _standardize(base["attack_posterior_mean"])
    base["defense_prior"] = _standardize(base["defense_posterior_mean"])

    if "elo" not in base.columns:
        base["elo"] = np.nan
    if "market_value_eur" not in base.columns:
        base["market_value_eur"] = np.nan
    if "form_pts" not in base.columns:
        base["form_pts"] = np.nan
    if "experience_index" not in base.columns:
        base["experience_index"] = np.nan
    if "host_flag" not in base.columns:
        base["host_flag"] = 0
    if "scenario_bias_prior" not in base.columns:
        base["scenario_bias_prior"] = 0.0
    if "travel_prior" not in base.columns:
        base["travel_prior"] = 0.0

    base["elo_prior"] = _standardize(base["elo"])
    base["market_prior"] = _standardize(np.log1p(pd.to_numeric(base["market_value_eur"], errors="coerce").fillna(0.0)))
    base["form_prior"] = _standardize(base["form_pts"])
    base["experience_prior"] = _standardize(base["experience_index"])

    base["host_prior"] = pd.to_numeric(base["host_flag"], errors="coerce").fillna(0.0)

    cols = [
        "team",
        "attack_prior",
        "defense_prior",
        "attack_posterior_mean",
        "attack_posterior_sd",
        "defense_posterior_mean",
        "defense_posterior_sd",
        "elo_prior",
        "market_prior",
        "form_prior",
        "experience_prior",
        "host_prior",
        "travel_prior",
        "scenario_bias_prior",
    ]
    return base[cols].sort_values("team").reset_index(drop=True)


def sample_team_table_from_posterior(
    idata: az.InferenceData,
    team_universe: pd.DataFrame,
    team_order: list[str],
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng() if rng is None else rng
    post = idata.posterior.stack(sample=("chain", "draw"))
    idx = int(rng.integers(post.sizes["sample"]))

    attack = np.asarray(post["attack"].isel(sample=idx).values)
    defense = np.asarray(post["defense"].isel(sample=idx).values)

    base = team_universe.copy()
    base["team"] = base["team"].map(normalize_team_name)
    base = base.set_index("team").loc[team_order].reset_index()

    base["attack_posterior_mean"] = attack
    base["defense_posterior_mean"] = defense
    base["attack_prior"] = _standardize(attack)
    base["defense_prior"] = _standardize(defense)

    if "elo_prior" not in base.columns:
        base["elo_prior"] = 0.0
    if "market_prior" not in base.columns:
        base["market_prior"] = 0.0
    if "form_prior" not in base.columns:
        base["form_prior"] = 0.0
    if "experience_prior" not in base.columns:
        base["experience_prior"] = 0.0
    if "host_prior" not in base.columns:
        base["host_prior"] = 0.0
    if "travel_prior" not in base.columns:
        base["travel_prior"] = 0.0
    if "scenario_bias_prior" not in base.columns:
        base["scenario_bias_prior"] = 0.0

    return base


def run_bayesian_calibration(
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 2,
    cores: int = 2,
    target_accept: float = 0.92,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, dict, az.InferenceData]:
    fit_result = fit_bayesian_strength_model(
        draws=draws,
        tune=tune,
        chains=chains,
        cores=cores,
        target_accept=target_accept,
        seed=seed,
    )

    team_universe = load_team_universe()
    priors = posterior_summary_table(fit_result.idata, team_universe, fit_result.team_order)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    priors.to_csv(PROCESSED_DIR / "team_strength_priors.csv", index=False)
    
    def _flatten_samples(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr)
        if arr.ndim >= 2:
            return arr.reshape(arr.shape[0] * arr.shape[1], *arr.shape[2:])
        return arr

    def save_posterior_draws(idata: az.InferenceData, team_order: list[str], competition_order: list[str]) -> Path:
        post = idata.posterior
        out_path = PROCESSED_DIR / "bayesian_posterior_draws.npz"

        attack = _flatten_samples(post["attack"].values)
        defense = _flatten_samples(post["defense"].values)
        competition_effect = _flatten_samples(post["competition_effect"].values)

        intercept = _flatten_samples(post["intercept"].values).reshape(-1)
        beta_home = _flatten_samples(post["beta_home"].values).reshape(-1)
        sigma_attack = _flatten_samples(post["sigma_attack"].values).reshape(-1)
        sigma_defense = _flatten_samples(post["sigma_defense"].values).reshape(-1)
        sigma_comp = _flatten_samples(post["sigma_comp"].values).reshape(-1)

        np.savez_compressed(
            out_path,
            attack=attack,
            defense=defense,
            competition_effect=competition_effect,
            intercept=intercept,
            beta_home=beta_home,
            sigma_attack=sigma_attack,
            sigma_defense=sigma_defense,
            sigma_comp=sigma_comp,
            team_order=np.array(team_order, dtype=object),
            competition_order=np.array(competition_order, dtype=object),
        )

        return out_path
    
    posterior_path = save_posterior_draws(fit_result.idata, fit_result.team_order, fit_result.competition_order)

    meta = {
        "team_order": fit_result.team_order,
        "competition_order": fit_result.competition_order,
        "posterior_path": str(posterior_path.name),
    }
    with open(PROCESSED_DIR / "bayesian_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    report = dict(fit_result.metrics)
    report["draws"] = draws
    report["tune"] = tune
    report["chains"] = chains
    report["cores"] = cores
    report["target_accept"] = target_accept
    with open(PROCESSED_DIR / "bayesian_calibration_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return priors, report, fit_result.idata


if __name__ == "__main__":
    priors, report, _ = run_bayesian_calibration()
    print(priors.head(10).to_string(index=False))
    print(report)