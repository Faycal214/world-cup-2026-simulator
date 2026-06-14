from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

team = pd.read_csv(PROCESSED / "team_features.csv").copy()

def safe_zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")

    if s.notna().sum() == 0:
        return pd.Series(0.0, index=s.index)

    s = s.fillna(s.median())
    std = s.std(ddof=0)

    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=s.index)

    return (s - s.mean()) / std

# Make sure the raw inputs are numeric and finite
team["elo"] = pd.to_numeric(team.get("elo", np.nan), errors="coerce")
team["market_value_eur"] = pd.to_numeric(team.get("market_value_eur", np.nan), errors="coerce")
team["form_pts"] = pd.to_numeric(team.get("form_pts", np.nan), errors="coerce")
team["experience_index"] = pd.to_numeric(team.get("experience_index", np.nan), errors="coerce")

# Fill missing values before transforms
if team["market_value_eur"].notna().any():
    mv_fill = team["market_value_eur"].median()
else:
    mv_fill = 0.0

team["market_value_eur"] = team["market_value_eur"].fillna(mv_fill)
team["elo"] = team["elo"].fillna(team["elo"].median() if team["elo"].notna().any() else 1500.0)
team["form_pts"] = team["form_pts"].fillna(0.0)
team["experience_index"] = team["experience_index"].fillna(team["experience_index"].median() if team["experience_index"].notna().any() else 0.0)

# Stable transformed features
team["log_market_value"] = np.log1p(team["market_value_eur"].clip(lower=0))
team["elo_prior"] = safe_zscore(team["elo"])
team["market_prior"] = safe_zscore(team["log_market_value"])
team["form_prior"] = safe_zscore(team["form_pts"])
team["experience_prior"] = safe_zscore(team["experience_index"])

# Priors
team["attack_prior"] = (
      0.40 * team["market_prior"]
    + 0.25 * team["elo_prior"]
    + 0.20 * team["form_prior"]
    + 0.15 * team["experience_prior"]
)

team["defense_prior"] = (
      0.35 * team["elo_prior"]
    + 0.25 * team["experience_prior"]
    + 0.25 * team["form_prior"]
    + 0.15 * team["market_prior"]
)

# Final safety cleanup
for col in ["attack_prior", "defense_prior", "elo_prior", "market_prior", "form_prior", "experience_prior"]:
    team[col] = pd.to_numeric(team[col], errors="coerce").fillna(0.0)

team["attack_prior"] = team["attack_prior"].clip(-2.5, 2.5)
team["defense_prior"] = team["defense_prior"].clip(-2.5, 2.5)

team["host_prior"] = pd.to_numeric(team.get("host_flag", 0), errors="coerce").fillna(0.0)
team["travel_prior"] = 0.0
team["scenario_bias_prior"] = pd.to_numeric(team.get("scenario_bias_prior", 0), errors="coerce").fillna(0.0)

priors = team[
    [
        "team",
        "attack_prior",
        "defense_prior",
        "elo_prior",
        "market_prior",
        "form_prior",
        "host_prior",
        "travel_prior",
        "scenario_bias_prior",
    ]
].copy()

priors.to_csv(PROCESSED / "team_strength_priors.csv", index=False)
print(priors.head(10).to_string(index=False))
print("saved:", PROCESSED / "team_strength_priors.csv", priors.shape)