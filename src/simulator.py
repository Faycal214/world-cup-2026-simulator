from __future__ import annotations

from typing import List, Tuple

import pandas as pd

from config import KNOCKOUT_ROTATION_VENUES
from knockout_engine import KnockoutEngine
from naming import normalize_team_name
from poisson_engine import MatchSimulator
from stakes_engine import calculate_stakes, format_stakes_display
from tournament_engine import TournamentEngine
from venue_data import get_knockout_venue


def print_tree(stage_name: str, matches: list[dict]) -> None:
    print(f"\n==================== {stage_name.upper()} RESULTS ====================")
    for m in matches:
        meta = f" ({m['meta']})" if m.get("meta") else ""
        print(f"{m['home']} {m['score_home']} - {m['score_away']} {m['away']}{meta}")


class WorldCupSimulator:
    def __init__(
        self,
        team_profiles: pd.DataFrame,
        fixture_features: pd.DataFrame,
        random_state: int | None = None,
    ):
        self.team_profiles = team_profiles.copy()
        self.fixture_features = fixture_features.copy() if fixture_features is not None else pd.DataFrame()
        self.match_simulator = MatchSimulator(random_state=random_state)
        self.tournament = TournamentEngine(self.team_profiles, self.fixture_features)

        self.team_profiles["team"] = self.team_profiles["team"].apply(normalize_team_name)
        self.team_profiles = self.team_profiles.set_index("team", drop=False)

    def _profile(self, team: str) -> dict:
        key = normalize_team_name(team)
        if key not in self.team_profiles.index:
            raise KeyError(f"Team not found in team_profiles: {team}")
        return self.team_profiles.loc[key].to_dict()

    def run_group_stage(self) -> None:
        for matchday in [1, 2, 3]:
            print(f"\n⚡ SIMULATING MATCHDAY {matchday}")
            fixtures = self.tournament.generate_fixtures(matchday)

            for group, home, away in fixtures:
                profile_home = self._profile(home)
                profile_away = self._profile(away)

                home_pts = int(self.tournament.group_tables[group].loc[home, "Pts"])
                away_pts = int(self.tournament.group_tables[group].loc[away, "Pts"])
                home_mp = int(self.tournament.group_tables[group].loc[home, "MP"])
                away_mp = int(self.tournament.group_tables[group].loc[away, "MP"])

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

                fixture_ctx = self.tournament.get_fixture_context(group, matchday, home, away)

                report_ctx = {
                    "stage": f"Group {group} - Matchday {matchday}",
                    "venue_name": fixture_ctx.get("venue_name", ""),
                    "stakes_home": format_stakes_display(stakes_home),
                    "stakes_away": format_stakes_display(stakes_away),
                }
                TournamentEngine.pre_match_report(home, away, report_ctx)

                sim_ctx = {
                    "stage": "group_stage_final" if matchday == 3 else "group",
                    "group": group,
                    "venue_name": fixture_ctx.get("venue_name", ""),
                    "travel_intensity_proxy": fixture_ctx.get("travel_intensity_proxy", 1.0),
                }

                res = self.match_simulator.simulate_match(
                    profile_home,
                    profile_away,
                    sim_ctx,
                    team_class_a=profile_home.get("team_class"),
                    team_class_b=profile_away.get("team_class"),
                    stakes_a=stakes_home,
                    stakes_b=stakes_away,
                )

                self.tournament.update_table(group, home, away, res["score_a"], res["score_b"])

                result_str = f"RESULT: {res['score_a']} - {res['score_b']}"
                if res["extra_time"]:
                    result_str += " (AET)"
                if res["penalty_winner"]:
                    winner = home if res["penalty_winner"] == "A" else away
                    result_str += f" ({winner} on PKs)"
                print(result_str + "\n")

            self.tournament.display_tables()

    def run_knockout_stage(self) -> None:
        top2_qualifiers = self.tournament.get_top_two_qualifiers()

        advancing_3rd, groups_3rd_set = KnockoutEngine.extract_and_rank_third_places(
            self.tournament.group_tables
        )

        print("\n⭐ ADVANCING THIRD-PLACED TEAMS (Ranked):")
        for i, team in enumerate(advancing_3rd, 1):
            print(f"  {i}. {team}")

        assert len(advancing_3rd) == 8, "Expected 8 third-place qualifiers"

        pairings = KnockoutEngine.get_round_of_32_pairings(
            top2_qualifiers,
            advancing_3rd,
            groups_3rd_set,
        )

        def run_knockout_stage(
            pairings: list[tuple[str, str]],
            stage_name: str,
            stage_context: str,
        ) -> tuple[list[str], list[dict]]:
            next_stage_teams = []
            logs = []

            for home, away in pairings:
                p_home = self._profile(home)
                p_away = self._profile(away)

                venue_name = get_knockout_venue(home, away)

                TournamentEngine.pre_match_report(
                    home,
                    away,
                    {
                        "stage": stage_name,
                        "venue_name": venue_name,
                        "stakes_home": "Knockout",
                        "stakes_away": "Knockout",
                    },
                )

                res = self.match_simulator.simulate_match(
                    p_home,
                    p_away,
                    {"stage": stage_context, "venue_name": venue_name},
                    team_class_a=p_home.get("team_class"),
                    team_class_b=p_away.get("team_class"),
                    stakes_a="Knockout_Tension",
                    stakes_b="Knockout_Tension",
                )

                if res["penalty_winner"]:
                    winner = home if res["penalty_winner"] == "A" else away
                else:
                    winner = home if res["score_a"] > res["score_b"] else away

                next_stage_teams.append(winner)

                meta = ""
                if res["extra_time"]:
                    meta = "AET"
                if res["penalty_winner"]:
                    meta = f"{meta} ({winner} on PKs)".strip()

                logs.append(
                    {
                        "home": home,
                        "away": away,
                        "score_home": res["score_a"],
                        "score_away": res["score_b"],
                        "meta": meta,
                    }
                )

                result_str = f"RESULT: {res['score_a']} - {res['score_b']}"
                if res["extra_time"]:
                    result_str += " (AET)"
                if res["penalty_winner"]:
                    result_str += f" ({winner} on PKs)"
                print(result_str + "\n")

            return next_stage_teams, logs

        r32_winners, logs = run_knockout_stage(pairings, "Round of 32", "round_of_32")
        print_tree("Round of 32", logs)

        r16_pairings = list(zip(r32_winners[::2], r32_winners[1::2]))
        r16_winners, logs = run_knockout_stage(r16_pairings, "Round of 16", "round_of_16")
        print_tree("Round of 16", logs)

        qf_pairings = list(zip(r16_winners[::2], r16_winners[1::2]))
        qf_winners, logs = run_knockout_stage(qf_pairings, "Quarter-Finals", "quarter_finals")
        print_tree("Quarter-Finals", logs)

        sf_pairings = list(zip(qf_winners[::2], qf_winners[1::2]))
        sf_winners, logs = run_knockout_stage(sf_pairings, "Semi-Finals", "semi_finals")
        print_tree("Semi-Finals", logs)

        final_pairing = [(sf_winners[0], sf_winners[1])]
        champion, logs = run_knockout_stage(final_pairing, "World Cup Final", "final")
        print_tree("World Cup Final", logs)

        print(f"\n🏆 THE 2026 WORLD CUP CHAMPION IS: {champion[0].upper()}")

    def run(self) -> None:
        self.run_group_stage()
        self.run_knockout_stage()