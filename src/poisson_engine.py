from __future__ import annotations

import numpy as np

from src.climate_engine import get_climate_factor
from src.stakes_engine import get_stakes_modifier
from src.team_mentality import get_mentality_modifier
from src.config import HOST_TEAMS

KNOCKOUT_STAGES = {"round_of_32", "round_of_16", "quarter_finals", "semi_finals", "final"}

class MatchSimulator:
    def __init__(self, random_state: int | None = None, elo_scale: float = 0.0012):
        self.rng = np.random.default_rng(random_state)
        self.elo_scale = elo_scale

    @staticmethod
    def _safe_float(value, default=0.0) -> float:
        try:
            if value is None:
                return default
            if isinstance(value, str) and not value.strip():
                return default
            return float(value)
        except Exception:
            return default

    def calculate_lambda(
        self,
        team_profile: dict,
        opponent_profile: dict,
        context: dict,
        team_class: str = None,
        opponent_class: str = None,
        stakes: str = None,
    ) -> float:
        venue_name = context.get("venue_name", "")
        climate = get_climate_factor(team_profile, venue_name)["factor"]

        mentality = 1.0
        if team_class and opponent_class:
            mentality = get_mentality_modifier(team_class, opponent_class)

        stakes_modifier = get_stakes_modifier(stakes) if stakes else 1.0

        elo_a = self._safe_float(team_profile.get("current_elo", team_profile.get("elo", 1500.0)), 1500.0)
        elo_b = self._safe_float(opponent_profile.get("current_elo", opponent_profile.get("elo", 1500.0)), 1500.0)

        attack = self._safe_float(team_profile.get("attack_prior"), 0.0)
        defense_opp = self._safe_float(opponent_profile.get("defense_prior"), 0.0)

        elo_term = self._safe_float(team_profile.get("elo_prior"), 0.0) - self._safe_float(opponent_profile.get("elo_prior"), 0.0)
        market_term = self._safe_float(team_profile.get("market_prior"), 0.0) - self._safe_float(opponent_profile.get("market_prior"), 0.0)
        form_term = self._safe_float(team_profile.get("form_prior"), 0.0) - self._safe_float(opponent_profile.get("form_prior"), 0.0)

        host_term = self._safe_float(team_profile.get("host_prior"), 0.0) - self._safe_float(opponent_profile.get("host_prior"), 0.0)
        travel_term = -self._safe_float(team_profile.get("travel_prior"), 0.0)
        scenario_term = self._safe_float(team_profile.get("scenario_bias_prior"), 0.0)

        # A compact but expressive intensity model.
        linear = (
            0.12
            + 0.55 * attack
            - 0.42 * defense_opp
            + 0.18 * elo_term
            + 0.10 * market_term
            + 0.12 * form_term
            + 0.24 * host_term
            + 0.08 * travel_term
            + 0.08 * scenario_term
        )

        # Host nations can get a mild structural lift.
        if team_profile.get("team") in HOST_TEAMS:
            linear += 0.06

        # Slight dampening in highly tense matches if stakes are set.
        if stakes == "Knockout_Tension":
            linear -= 0.08

        lambda_base = np.exp(linear)
        lambda_final = lambda_base * climate * mentality * stakes_modifier

        return float(np.clip(lambda_final, 0.15, 4.80))

    def _shared_intensity(self, lambda_a: float, lambda_b: float) -> float:
        # Shared latent state for a bivariate Poisson: small positive covariance.
        shared = 0.10 + 0.18 * np.sqrt(max(lambda_a, 0.0) * max(lambda_b, 0.0))
        return float(np.clip(shared, 0.03, 0.60))

    def _simulate_bivariate_goals(self, lambda_a: float, lambda_b: float) -> tuple[int, int, float]:
        shared = self._shared_intensity(lambda_a, lambda_b)
        private_a = max(lambda_a - shared, 0.05)
        private_b = max(lambda_b - shared, 0.05)

        x1 = self.rng.poisson(private_a)
        x2 = self.rng.poisson(private_b)
        x3 = self.rng.poisson(shared)

        return int(x1 + x3), int(x2 + x3), shared

    def penalty_shootout(self, team_a: dict, team_b: dict) -> dict:
        elo_a = self._safe_float(team_a.get("current_elo", team_a.get("elo", 1500.0)), 1500.0)
        elo_b = self._safe_float(team_b.get("current_elo", team_b.get("elo", 1500.0)), 1500.0)
        diff = elo_a - elo_b

        p_a = float(np.clip(0.75 + 0.00008 * diff, 0.58, 0.88))
        p_b = float(np.clip(0.75 - 0.00008 * diff, 0.58, 0.88))

        score_a = 0
        score_b = 0

        for _ in range(5):
            score_a += int(self.rng.binomial(1, p_a))
            score_b += int(self.rng.binomial(1, p_b))

        if score_a != score_b:
            return {
                "winner": "A" if score_a > score_b else "B",
                "score_a": score_a,
                "score_b": score_b,
            }

        # Sudden death
        while score_a == score_b:
            score_a += int(self.rng.binomial(1, p_a))
            score_b += int(self.rng.binomial(1, p_b))

        return {
            "winner": "A" if score_a > score_b else "B",
            "score_a": score_a,
            "score_b": score_b,
        }

    def simulate_match(
        self,
        team_a: dict,
        team_b: dict,
        context: dict | None = None,
        team_class_a: str | None = None,
        team_class_b: str | None = None,
        stakes_a: str | None = None,
        stakes_b: str | None = None,
    ) -> dict:
        context = context or {"stage": "group", "venue_name": "Neutral"}

        lambda_a = self.calculate_lambda(
            team_a,
            team_b,
            context,
            team_class=team_class_a,
            opponent_class=team_class_b,
            stakes=stakes_a,
        )
        lambda_b = self.calculate_lambda(
            team_b,
            team_a,
            context,
            team_class=team_class_b,
            opponent_class=team_class_a,
            stakes=stakes_b,
        )

        score_a, score_b, shared = self._simulate_bivariate_goals(lambda_a, lambda_b)

        extra_time = False
        penalty_winner = None
        penalty_score = None
        stage = context.get("stage", "group")

        if stage in KNOCKOUT_STAGES and score_a == score_b:
            extra_time = True
            et_a, et_b, _ = self._simulate_bivariate_goals(lambda_a / 3.0, lambda_b / 3.0)
            score_a += et_a
            score_b += et_b

            if score_a == score_b:
                shoot = self.penalty_shootout(team_a, team_b)
                penalty_winner = shoot["winner"]
                penalty_score = (shoot["score_a"], shoot["score_b"])

        winner = None
        method = "90min"

        if score_a > score_b:
            winner = "A"
        elif score_b > score_a:
            winner = "B"
        else:
            if penalty_winner is not None:
                winner = penalty_winner
                method = "PKs"
            else:
                method = "Draw"

        if extra_time:
            method = "AET" if penalty_winner is None else "AET+PKs"

        return {
            "score_a": score_a,
            "score_b": score_b,
            "lambda_a": lambda_a,
            "lambda_b": lambda_b,
            "shared_component": shared,
            "extra_time": extra_time,
            "penalty_winner": penalty_winner,
            "penalty_score": penalty_score,
            "winner": winner,
            "method": method,
        }