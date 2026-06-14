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
    def _safe_float(value, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            if isinstance(value, str) and not value.strip():
                return default
            out = float(value)
            if not np.isfinite(out):
                return default
            return out
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

        attack = self._safe_float(team_profile.get("attack_prior"), 0.0)
        defense_opp = self._safe_float(opponent_profile.get("defense_prior"), 0.0)

        elo_term = self._safe_float(team_profile.get("elo_prior"), 0.0) - self._safe_float(opponent_profile.get("elo_prior"), 0.0)
        market_term = self._safe_float(team_profile.get("market_prior"), 0.0) - self._safe_float(opponent_profile.get("market_prior"), 0.0)
        form_term = self._safe_float(team_profile.get("form_prior"), 0.0) - self._safe_float(opponent_profile.get("form_prior"), 0.0)

        host_term = self._safe_float(team_profile.get("host_prior"), 0.0) - self._safe_float(opponent_profile.get("host_prior"), 0.0)
        travel_term = -self._safe_float(team_profile.get("travel_prior"), 0.0)
        scenario_term = self._safe_float(team_profile.get("scenario_bias_prior"), 0.0)

        # Smaller coefficients = more realistic goal rates
        linear = (
            0.05
            + 0.22 * attack
            - 0.30 * defense_opp
            + 0.07 * elo_term
            + 0.05 * market_term
            + 0.06 * form_term
            + 0.10 * host_term
            + 0.08 * travel_term
            + 0.08 * scenario_term
        )

        if stakes == "Knockout_Tension":
            linear -= 0.06

        linear = float(np.clip(linear, -1.10, 0.80))
        lambda_final = float(np.exp(linear) * climate * mentality * stakes_modifier)

        return float(np.clip(lambda_final, 0.25, 2.40))

    def _shared_intensity(self, lambda_a: float, lambda_b: float) -> float:
        shared = 0.4 + 0.08 * np.sqrt(max(lambda_a, 0.0) * max(lambda_b, 0.0))
        if not np.isfinite(shared):
            shared = 0.10
        return float(np.clip(shared, 0.02, 0.20))

    def _simulate_bivariate_goals(self, lambda_a: float, lambda_b: float) -> tuple[int, int, float]:
        lambda_a = self._safe_float(lambda_a, 0.6)
        lambda_b = self._safe_float(lambda_b, 0.6)

        lambda_a = max(lambda_a, 0.15)
        lambda_b = max(lambda_b, 0.15)

        shared = self._shared_intensity(lambda_a, lambda_b)

        private_a = max(lambda_a - shared, 0.05)
        private_b = max(lambda_b - shared, 0.05)

        private_a = float(np.clip(private_a, 0.05, 4.0))
        private_b = float(np.clip(private_b, 0.05, 4.0))
        shared = float(np.clip(shared, 0.03, 0.60))

        x1 = int(self.rng.poisson(private_a))
        x2 = int(self.rng.poisson(private_b))
        x3 = int(self.rng.poisson(shared))

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