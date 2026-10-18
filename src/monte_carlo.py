from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from config import PROCESSED_DIR, KNOCKOUT_ROTATION_VENUES
from data_loader import load_team_profiles, load_fixture_features
from knockout_engine import KnockoutEngine
from naming import normalize_team_name
from poisson_engine import MatchSimulator
from stakes_engine import calculate_stakes
from tournament_engine import TournamentEngine
from bayesian_posterior import load_bayesian_posterior, sample_strength_table

DEFAULT_N_SIMS = 10_000
DEFAULT_SEED = 42
OUTPUT_FILE = PROCESSED_DIR / "simulation_summary.csv"


def _knockout_venue(match_number: int) -> str:
    idx = (match_number - 73) % len(KNOCKOUT_ROTATION_VENUES)
    return KNOCKOUT_ROTATION_VENUES[idx]


def _winner_from_result(home: str, away: str, result: dict) -> str:
    if result.get("penalty_winner") == "A":
        return home
    if result.get("penalty_winner") == "B":
        return away

    if result["score_a"] > result["score_b"]:
        return home
    if result["score_b"] > result["score_a"]:
        return away

    return home if result.get("winner") == "A" else away


@dataclass
class MonteCarloCounts:
    teams: list[str]
    team_to_idx: dict[str, int]
    r16: np.ndarray
    qf: np.ndarray
    sf: np.ndarray
    final: np.ndarray
    title: np.ndarray

    @classmethod
    def create(cls, teams: list[str]) -> "MonteCarloCounts":
        team_to_idx = {team: i for i, team in enumerate(teams)}
        n = len(teams)
        zeros = np.zeros(n, dtype=np.int64)
        return cls(
            teams=teams,
            team_to_idx=team_to_idx,
            r16=zeros.copy(),
            qf=zeros.copy(),
            sf=zeros.copy(),
            final=zeros.copy(),
            title=zeros.copy(),
        )

    def increment(self, stage: str, team_names: list[str]) -> None:
        arr = getattr(self, stage)
        for team in team_names:
            if team in self.team_to_idx:
                arr[self.team_to_idx[team]] += 1

    def to_frame(self, n_sims: int) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "team": self.teams,
                "r16_probability": self.r16 / n_sims,
                "qf_probability": self.qf / n_sims,
                "sf_probability": self.sf / n_sims,
                "final_probability": self.final / n_sims,
                "title_probability": self.title / n_sims,
            }
        )
        return df.sort_values("title_probability", ascending=False).reset_index(drop=True)


class MonteCarloWorldCup:
    def __init__(
        self,
        team_profiles: pd.DataFrame,
        fixture_features: pd.DataFrame,
        seed: int = DEFAULT_SEED,
    ):
        team_profiles = team_profiles.copy()
        team_profiles["team"] = team_profiles["team"].map(normalize_team_name)
        team_profiles = team_profiles.dropna(subset=["team"]).copy()
        team_profiles = team_profiles.drop_duplicates(subset=["team"], keep="first")
        team_profiles = team_profiles.set_index("team", drop=False).sort_index()

        self.base_team_profiles = team_profiles.copy()
        self.team_profiles = team_profiles.copy()
        self.fixture_features = fixture_features.copy() if fixture_features is not None else pd.DataFrame()
        self.rng = np.random.default_rng(seed)

        self.teams = list(self.team_profiles.index)
        self.counts = MonteCarloCounts.create(self.teams)

        self.posterior = load_bayesian_posterior()

    def _profile(self, team: str) -> dict:
        key = normalize_team_name(team)
        if key not in self.team_profiles.index:
            raise KeyError(f"Team not found in team_profiles: {team}")
        return self.team_profiles.loc[key].to_dict()

    def _play_group_match(
        self,
        match_simulator: MatchSimulator,
        tournament: TournamentEngine,
        group: str,
        matchday: int,
        home: str,
        away: str,
    ) -> tuple[int, int]:
        fixture_ctx = tournament.get_fixture_context(group, matchday, home, away)

        profile_home = tournament.build_simulation_profile(
            home,
            fixture_context=fixture_ctx,
            matchday=matchday,
            is_home=True,
        )
        profile_away = tournament.build_simulation_profile(
            away,
            fixture_context=fixture_ctx,
            matchday=matchday,
            is_home=False,
        )

        home_pts = int(tournament.group_tables[group].loc[home, "Pts"])
        away_pts = int(tournament.group_tables[group].loc[away, "Pts"])
        home_mp = int(tournament.group_tables[group].loc[home, "MP"])
        away_mp = int(tournament.group_tables[group].loc[away, "MP"])

        stakes_home = calculate_stakes(
            home,
            home_pts,
            home_mp,
            "group_stage_final" if matchday == 3 else "group",
            profile_home.get("team_class"),
        )
        stakes_away = calculate_stakes(
            away,
            away_pts,
            away_mp,
            "group_stage_final" if matchday == 3 else "group",
            profile_away.get("team_class"),
        )

        sim_ctx = {
            **fixture_ctx,
            "stage": "group_stage_final" if matchday == 3 else "group",
            "group": group,
        }

        result = match_simulator.simulate_match(
            profile_home,
            profile_away,
            sim_ctx,
            team_class_a=profile_home.get("team_class"),
            team_class_b=profile_away.get("team_class"),
            stakes_a=stakes_home,
            stakes_b=stakes_away,
        )

        tournament.update_table(group, home, away, result["score_a"], result["score_b"])
        return int(result["score_a"]), int(result["score_b"])

    def _play_knockout_match(
        self,
        match_simulator: MatchSimulator,
        tournament: TournamentEngine,
        home: str,
        away: str,
        stage_context: str,
        match_number: int,
    ) -> str:
        venue_name = _knockout_venue(match_number)
        fixture_ctx = tournament.build_knockout_context(match_number, stage_context, venue_name)

        profile_home = tournament.build_simulation_profile(
            home,
            fixture_context=fixture_ctx,
            matchday=None,
            is_home=True,
        )
        profile_away = tournament.build_simulation_profile(
            away,
            fixture_context=fixture_ctx,
            matchday=None,
            is_home=False,
        )

        result = match_simulator.simulate_match(
            profile_home,
            profile_away,
            fixture_ctx,
            team_class_a=profile_home.get("team_class"),
            team_class_b=profile_away.get("team_class"),
            stakes_a="Knockout_Tension",
            stakes_b="Knockout_Tension",
        )

        return _winner_from_result(home, away, result)

    def simulate_one_tournament(self) -> str:
        sampled_profiles = sample_strength_table(
            self.base_team_profiles,
            self.posterior,
            self.rng,
        )

        sampled_profiles = sampled_profiles.copy()
        sampled_profiles["team"] = sampled_profiles["team"].map(normalize_team_name)
        sampled_profiles = sampled_profiles.dropna(subset=["team"]).copy()
        sampled_profiles = sampled_profiles.drop_duplicates(subset=["team"], keep="first")
        sampled_profiles = sampled_profiles.set_index("team", drop=False).sort_index()

        tournament = TournamentEngine(sampled_profiles, self.fixture_features)
        match_simulator = MatchSimulator(
            random_state=int(self.rng.integers(0, 2**32 - 1))
        )

        self.team_profiles = sampled_profiles.copy()

        # -------------------------
        # Group stage
        # -------------------------
        for matchday in [1, 2, 3]:
            fixtures = tournament.generate_fixtures(matchday)
            for group, home, away in fixtures:
                self._play_group_match(
                    match_simulator=match_simulator,
                    tournament=tournament,
                    group=group,
                    matchday=matchday,
                    home=home,
                    away=away,
                )

        # -------------------------
        # Round of 32
        # -------------------------
        top2_qualifiers = tournament.get_top_two_qualifiers()
        advancing_3rd, groups_3rd_set = KnockoutEngine.extract_and_rank_third_places(
            tournament.group_tables
        )

        r32_pairings = KnockoutEngine.get_round_of_32_pairings(
            top2_qualifiers,
            advancing_3rd,
            groups_3rd_set,
        )

        r32_winners_by_match = {}
        r32_winners = []
        for i, (home, away) in enumerate(r32_pairings):
            match_number = 73 + i
            winner = self._play_knockout_match(
                match_simulator=match_simulator,
                tournament=tournament,
                home=home,
                away=away,
                stage_context="round_of_32",
                match_number=match_number,
            )
            r32_winners_by_match[match_number] = winner
            r32_winners.append(winner)

        self.counts.increment("r16", r32_winners)

        # -------------------------
        # Round of 16
        # -------------------------
        r16_pairings = KnockoutEngine.get_round_of_16_pairings(r32_winners_by_match)

        r16_winners_by_match = {}
        r16_winners = []
        for match_number, home, away in r16_pairings:
            winner = self._play_knockout_match(
                match_simulator=match_simulator,
                tournament=tournament,
                home=home,
                away=away,
                stage_context="round_of_16",
                match_number=match_number,
            )
            r16_winners_by_match[match_number] = winner
            r16_winners.append(winner)

        self.counts.increment("qf", r16_winners)

        # -------------------------
        # Quarter-finals
        # -------------------------
        qf_pairings = KnockoutEngine.get_quarter_final_pairings(r16_winners_by_match)

        qf_winners_by_match = {}
        qf_winners = []
        for match_number, home, away in qf_pairings:
            winner = self._play_knockout_match(
                match_simulator=match_simulator,
                tournament=tournament,
                home=home,
                away=away,
                stage_context="quarter_finals",
                match_number=match_number,
            )
            qf_winners_by_match[match_number] = winner
            qf_winners.append(winner)

        self.counts.increment("sf", qf_winners)

        # -------------------------
        # Semi-finals
        # -------------------------
        sf_pairings = KnockoutEngine.get_semi_final_pairings(qf_winners_by_match)

        sf_winners_by_match = {}
        finalists = []
        for match_number, home, away in sf_pairings:
            winner = self._play_knockout_match(
                match_simulator=match_simulator,
                tournament=tournament,
                home=home,
                away=away,
                stage_context="semi_finals",
                match_number=match_number,
            )
            sf_winners_by_match[match_number] = winner
            finalists.append(winner)

        self.counts.increment("final", finalists)

        # -------------------------
        # Final
        # -------------------------
        final_match_number, final_home, final_away = KnockoutEngine.get_final_pairing(sf_winners_by_match)
        champion = self._play_knockout_match(
            match_simulator=match_simulator,
            tournament=tournament,
            home=final_home,
            away=final_away,
            stage_context="final",
            match_number=final_match_number,
        )
        self.counts.increment("title", [champion])

        return champion

    def run(self, n_sims: int = DEFAULT_N_SIMS) -> pd.DataFrame:
        n_sims = int(n_sims)
        if n_sims <= 0:
            raise ValueError("n_sims must be positive")

        checkpoint_every = 100
        save_every = 500

        for i in tqdm(range(n_sims), desc="Monte Carlo simulations"):
            self.simulate_one_tournament()

            if (i + 1) % checkpoint_every == 0:
                print(f"Completed {i + 1}/{n_sims} simulations", flush=True)

            if (i + 1) % save_every == 0:
                summary = self.counts.to_frame(i + 1)
                PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
                summary.to_csv(OUTPUT_FILE, index=False)
                print(f"Checkpoint saved at {i + 1} simulations", flush=True)

        summary = self.counts.to_frame(n_sims)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        summary.to_csv(OUTPUT_FILE, index=False)
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Monte Carlo World Cup simulator")
    parser.add_argument("--n-sims", type=int, default=DEFAULT_N_SIMS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    team_profiles = load_team_profiles()
    fixture_features = load_fixture_features()

    engine = MonteCarloWorldCup(
        team_profiles=team_profiles,
        fixture_features=fixture_features,
        seed=args.seed,
    )
    summary = engine.run(n_sims=args.n_sims)

    print(summary.head(20).to_string(index=False))
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()