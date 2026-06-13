from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

try:
    from tabulate import tabulate
except Exception:
    tabulate = None

from src.config import GROUPS
from src.naming import normalize_team_name
from src.venue_data import get_venue_by_group

GROUP_COLUMNS = ["MP", "W", "D", "L", "GF", "GA", "GD", "Pts"]

def _empty_table(teams: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Team": teams,
            "MP": 0,
            "W": 0,
            "D": 0,
            "L": 0,
            "GF": 0,
            "GA": 0,
            "GD": 0,
            "Pts": 0,
        }
    ).set_index("Team")

class TournamentEngine:
    def __init__(self, teams_df: pd.DataFrame, fixture_features: pd.DataFrame | None = None):
        self.teams_df = teams_df.copy()
        self.fixture_features = fixture_features.copy() if fixture_features is not None else None
        self.group_tables: dict[str, pd.DataFrame] = {}
        self.group_assignments: dict[str, list[str]] = {}
        self.flag_map: dict[str, str] = {}
        self.initialize_groups()

    def initialize_groups(self) -> None:
        for g_letter, team_list in GROUPS.items():
            teams = [normalize_team_name(t) for t in team_list]
            self.group_assignments[g_letter] = teams
            for t in teams:
                self.flag_map[t] = t
            self.group_tables[g_letter] = _empty_table(teams)

    def generate_fixtures(self, matchday: int) -> list[tuple[str, str, str]]:
        fixtures = []
        for g_letter, teams in self.group_assignments.items():
            if matchday == 1:
                fixtures.append((g_letter, teams[0], teams[1]))
                fixtures.append((g_letter, teams[2], teams[3]))
            elif matchday == 2:
                fixtures.append((g_letter, teams[0], teams[2]))
                fixtures.append((g_letter, teams[1], teams[3]))
            elif matchday == 3:
                fixtures.append((g_letter, teams[0], teams[3]))
                fixtures.append((g_letter, teams[1], teams[2]))
        return fixtures

    def get_fixture_context(self, group: str, matchday: int, home: str, away: str) -> dict:
        venue_name = get_venue_by_group(group, matchday)
        venue_city = venue_name
        venue_country = ""
        venue_stadium = ""
        date_et = None
        time_et = ""
        travel_intensity_proxy = 1.0

        if self.fixture_features is not None and not self.fixture_features.empty:
            fx = self.fixture_features.copy()
            fx["home_team_norm"] = fx.get("home_team", "").astype(str).map(normalize_team_name)
            fx["away_team_norm"] = fx.get("away_team", "").astype(str).map(normalize_team_name)
            fx["group_norm"] = fx.get("group", "").astype(str).str.strip()

            mask = (
                (fx["group_norm"] == group)
                & (fx["home_team_norm"] == home)
                & (fx["away_team_norm"] == away)
            )
            match = fx.loc[mask]
            if match.empty:
                match = fx.loc[fx["group_norm"] == group].head(1)

            if not match.empty:
                row = match.iloc[0]
                if "venue_city" in row and pd.notna(row.get("venue_city")) and str(row.get("venue_city")).strip():
                    venue_city = str(row.get("venue_city")).strip()
                    venue_name = venue_city
                if "venue_country" in row and pd.notna(row.get("venue_country")):
                    venue_country = str(row.get("venue_country")).strip()
                if "venue_stadium" in row and pd.notna(row.get("venue_stadium")):
                    venue_stadium = str(row.get("venue_stadium")).strip()
                if "date_et" in row and pd.notna(row.get("date_et")):
                    date_et = row.get("date_et")
                if "time_et" in row and pd.notna(row.get("time_et")):
                    time_et = str(row.get("time_et"))
                if "travel_intensity_proxy" in row and pd.notna(row.get("travel_intensity_proxy")):
                    travel_intensity_proxy = float(row.get("travel_intensity_proxy"))

        return {
            "stage": "group",
            "group": group,
            "matchday": matchday,
            "venue_name": venue_name,
            "venue_city": venue_city,
            "venue_country": venue_country,
            "venue_stadium": venue_stadium,
            "date_et": date_et,
            "time_et": time_et,
            "travel_intensity_proxy": travel_intensity_proxy,
        }

    def update_table(self, group: str, team_a: str, team_b: str, sa: int, sb: int) -> None:
        table = self.group_tables[group]
        table.loc[team_a, "MP"] += 1
        table.loc[team_b, "MP"] += 1
        table.loc[team_a, "GF"] += sa
        table.loc[team_a, "GA"] += sb
        table.loc[team_b, "GF"] += sb
        table.loc[team_b, "GA"] += sa
        table.loc[team_a, "GD"] = table.loc[team_a, "GF"] - table.loc[team_a, "GA"]
        table.loc[team_b, "GD"] = table.loc[team_b, "GF"] - table.loc[team_b, "GA"]

        if sa > sb:
            table.loc[team_a, "Pts"] += 3
            table.loc[team_a, "W"] += 1
            table.loc[team_b, "L"] += 1
        elif sb > sa:
            table.loc[team_b, "Pts"] += 3
            table.loc[team_b, "W"] += 1
            table.loc[team_a, "L"] += 1
        else:
            table.loc[team_a, "Pts"] += 1
            table.loc[team_b, "Pts"] += 1
            table.loc[team_a, "D"] += 1
            table.loc[team_b, "D"] += 1

    def sorted_group_table(self, group: str) -> pd.DataFrame:
        table = self.group_tables[group].copy()
        table = table.sort_values(by=["Pts", "GD", "GF", "GA"], ascending=[False, False, False, True])
        return table

    def get_top_two_qualifiers(self) -> dict[str, list[str]]:
        output = {}
        for g_letter in self.group_tables:
            ranked = self.sorted_group_table(g_letter)
            output[g_letter] = ranked.index[:2].tolist()
        return output

    def display_tables(self) -> None:
        for g_letter in self.group_tables:
            sorted_table = self.sorted_group_table(g_letter)
            print(f"\n--- Group {g_letter} Standings ---")
            if tabulate is not None:
                print(tabulate(sorted_table.reset_index(), headers="keys", tablefmt="psql", showindex=False))
            else:
                print(sorted_table.reset_index().to_string(index=False))

    @staticmethod
    def pre_match_report(home: str, away: str, context: dict | None = None) -> None:
        if not context:
            return
        stage = str(context.get("stage", "Match")).replace("_", " ").title()
        venue = context.get("venue_name", "TBD")
        print(f"[{stage}] | {venue} | {home} vs {away}")