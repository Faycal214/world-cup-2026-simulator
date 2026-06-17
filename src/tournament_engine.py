from __future__ import annotations

import pandas as pd
from tabulate import tabulate

from naming import normalize_team_name


class TournamentEngine:
    def __init__(self, teams_df: pd.DataFrame, fixture_features: pd.DataFrame | None = None):
        self.teams_df = teams_df.copy()
        self.fixture_features = fixture_features.copy() if fixture_features is not None else pd.DataFrame()

        self.group_tables = {}
        self.group_assignments = {}
        self.flag_map = {}

        self.initialize_groups()

    def initialize_groups(self):
        group_mapping = {
            "A": ["Mexico", "South Africa", "South Korea", "Czechia"],
            "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
            "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
            "D": ["United States", "Paraguay", "Australia", "Turkey"],
            "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
            "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
            "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
            "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
            "I": ["France", "Senegal", "Iraq", "Norway"],
            "J": ["Argentina", "Algeria", "Austria", "Jordan"],
            "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
            "L": ["England", "Croatia", "Ghana", "Panama"],
        }

        for g_letter, team_list in group_mapping.items():
            valid_teams = []
            for t in team_list:
                self.flag_map[t] = t
                valid_teams.append(t)

            self.group_assignments[g_letter] = valid_teams
            self.group_tables[g_letter] = pd.DataFrame(
                {
                    "Team": valid_teams,
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

    def _fixture_rows(self) -> pd.DataFrame:
        if self.fixture_features is None or self.fixture_features.empty:
            return pd.DataFrame()

        fx = self.fixture_features.copy()

        for col in ["match_id", "matchday", "stage", "group", "home_team", "away_team", "venue_city", "venue_country", "venue_stadium", "time_et", "date_et"]:
            if col not in fx.columns:
                fx[col] = pd.NA

        fx["stage"] = fx["stage"].astype(str).str.lower()
        fx["group"] = fx["group"].astype(str).str.strip()
        fx["home_team"] = fx["home_team"].astype(str).map(normalize_team_name)
        fx["away_team"] = fx["away_team"].astype(str).map(normalize_team_name)

        group_mask = fx["stage"].str.contains("group", na=False)
        fx = fx.loc[group_mask].copy()

        if "match_id" in fx.columns:
            fx["match_id"] = pd.to_numeric(fx["match_id"], errors="coerce")
            fx = fx.sort_values(["match_id", "group"]).reset_index(drop=True)

        return fx

    def generate_fixtures(self, matchday: int):
        """
        Return fixtures in official schedule order, not group-assignment order.
        Requires fixture_features.csv to contain matchday and match_id if you want exact day-by-day replication.
        """
        fx = self._fixture_rows()

        if fx.empty:
            # Fallback only if the processed fixture file is missing.
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

        # If matchday exists, use it. Otherwise infer by evenly splitting the 72 group matches.
        if fx["matchday"].notna().any():
            fx["matchday"] = pd.to_numeric(fx["matchday"], errors="coerce")
            fx = fx.loc[fx["matchday"] == matchday].copy()
        else:
            group_fixtures = fx.reset_index(drop=True)
            per_day = len(group_fixtures) // 3
            start = (matchday - 1) * per_day
            end = matchday * per_day if matchday < 3 else len(group_fixtures)
            fx = group_fixtures.iloc[start:end].copy()

        fx = fx.sort_values(["match_id", "group"]).reset_index(drop=True)

        fixtures = []
        for _, row in fx.iterrows():
            fixtures.append((row["group"], row["home_team"], row["away_team"]))

        return fixtures

    def get_fixture_context(self, group: str, matchday: int, home: str, away: str) -> dict:
        if self.fixture_features is None or self.fixture_features.empty:
            return {
                "stage": "group",
                "group": group,
                "matchday": matchday,
                "venue_name": "",
                "venue_city": "",
                "venue_country": "",
                "venue_stadium": "",
                "date_et": None,
                "time_et": "",
                "travel_intensity_proxy": 1.0,
            }

        fx = self._fixture_rows()
        home_n = normalize_team_name(home)
        away_n = normalize_team_name(away)

        candidates = fx.loc[
            (fx["group"] == group)
            & (
                ((fx["home_team"] == home_n) & (fx["away_team"] == away_n))
                | ((fx["home_team"] == away_n) & (fx["away_team"] == home_n))
            )
        ].copy()

        if candidates.empty:
            candidates = fx.loc[fx["group"] == group].copy()

        if candidates.empty:
            return {
                "stage": "group",
                "group": group,
                "matchday": matchday,
                "venue_name": "",
                "venue_city": "",
                "venue_country": "",
                "venue_stadium": "",
                "date_et": None,
                "time_et": "",
                "travel_intensity_proxy": 1.0,
            }

        row = candidates.sort_values(["match_id"]).iloc[0]

        venue_city = str(row.get("venue_city", "") or "").strip()
        venue_stadium = str(row.get("venue_stadium", "") or "").strip()
        venue_country = str(row.get("venue_country", "") or "").strip()

        venue_name = venue_stadium or venue_city or ""

        return {
            "stage": "group",
            "group": group,
            "matchday": matchday,
            "match_id": int(row["match_id"]) if pd.notna(row.get("match_id")) else None,
            "venue_name": venue_name,
            "venue_city": venue_city,
            "venue_country": venue_country,
            "venue_stadium": venue_stadium,
            "date_et": row.get("date_et"),
            "time_et": str(row.get("time_et", "") or "").strip(),
            "travel_intensity_proxy": float(row.get("travel_intensity_proxy", 1.0) or 1.0),
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
        return table.sort_values(by=["Pts", "GD", "GF", "GA"], ascending=[False, False, False, True])

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
            print(tabulate(sorted_table.reset_index(), headers="keys", tablefmt="psql", showindex=False))

    @staticmethod
    def pre_match_report(home: str, away: str, context: dict | None = None) -> None:
        if not context:
            return
        stage = str(context.get("stage", "Match")).replace("_", " ").title()
        venue = context.get("venue_name", "").strip()
        print(f"[{stage}] | {venue} | {home} vs {away}")