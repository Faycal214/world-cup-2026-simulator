from __future__ import annotations

from pathlib import Path
import numpy as np

from config import PROCESSED_DIR

POSTERIOR_PATH = PROCESSED_DIR / "bayesian_posterior_draws.npz"


def load_bayesian_posterior(path: Path | None = None) -> dict:
    path = path or POSTERIOR_PATH
    if not path.exists():
        raise FileNotFoundError(f"Missing posterior draws file: {path}")

    data = np.load(path, allow_pickle=True)

    return {
        "attack": data["attack"],
        "defense": data["defense"],
        "competition_effect": data["competition_effect"],
        "intercept": data["intercept"],
        "beta_home": data["beta_home"],
        "sigma_attack": data["sigma_attack"],
        "sigma_defense": data["sigma_defense"],
        "sigma_comp": data["sigma_comp"],
        "team_order": data["team_order"].tolist(),
        "competition_order": data["competition_order"].tolist(),
    }


def sample_strength_table(base_profiles, posterior: dict, rng: np.random.Generator):
    team_order = posterior["team_order"]
    attack_draws = posterior["attack"]
    defense_draws = posterior["defense"]

    idx = int(rng.integers(0, attack_draws.shape[0]))

    attack = attack_draws[idx]
    defense = defense_draws[idx]

    # Compression factor: lower = more balanced tournament
    temperature = 0.80

    attack_mean = attack_draws.mean(axis=0)
    defense_mean = defense_draws.mean(axis=0)

    attack = attack_mean + temperature * (attack - attack_mean)
    defense = defense_mean + temperature * (defense - defense_mean)

    sampled = base_profiles.copy().set_index("team", drop=False)

    for i, team in enumerate(team_order):
        if team in sampled.index:
            sampled.loc[team, "attack_prior"] = float(attack[i])
            sampled.loc[team, "defense_prior"] = float(defense[i])

    return sampled.reset_index(drop=True)