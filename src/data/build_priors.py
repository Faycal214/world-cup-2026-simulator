from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

team = pd.read_csv(
    ROOT / "data" / "processed" / "team_features.csv"
)

def zscore(x):
    return (x - x.mean()) / x.std()

team["elo_prior"] = zscore(team["elo"].fillna(team["elo"].mean()))

team["market_prior"] = zscore(
    team["log_market_value"]
)

team["form_prior"] = zscore(
    team["form_pts"].fillna(0)
)

team["experience_prior"] = zscore(
    team["experience_index"].fillna(0)
)

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

team["host_prior"] = team["host_flag"]

team["travel_prior"] = 0.0

team["scenario_bias_prior"] = team["scenario_bias_prior"]

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
        "scenario_bias_prior"
    ]
]

priors.to_csv(
    ROOT / "data" / "processed" / "team_strength_priors.csv",
    index=False
)

print(priors.head())
print(priors.shape)