from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd
from tabulate import tabulate

from naming import normalize_team_name
from config import HOST_TEAMS

def _default_if_missing(value, default):
    """Replace missing or empty scalar values without treating zero as missing."""
    if value is None or pd.isna(value):
        return default

    if isinstance(value, str) and not value.strip():
        return default

    return value

class TournamentEngine:
    def __init__(self, teams_df: pd.DataFrame, fixture_features: pd.DataFrame | None = None):
        self.teams_df = self._prepare_teams_df(teams_df)
        self.fixture_features = fixture_features.copy() if fixture_features is not None else pd.DataFrame()

        self.group_tables: dict[str, pd.DataFrame] = {}
        self.group_assignments: dict[str, list[str]] = {}
        self.flag_map: dict[str, str] = {}

        self.initialize_groups()

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

    @staticmethod
    def _safe_str(value, default: str = "") -> str:
        try:
            if value is None:
                return default
            s = str(value).strip()
            return s if s else default
        except Exception:
            return default

    def _prepare_teams_df(self, teams_df: pd.DataFrame) -> pd.DataFrame:
        df = teams_df.copy()

        if "team" not in df.columns:
            if df.index.name and str(df.index.name).lower() in {"team", "teams"}:
                df = df.reset_index()
                if "team" not in df.columns:
                    df = df.rename(columns={df.columns[0]: "team"})
            else:
                df = df.reset_index()
                if "team" not in df.columns:
                    df = df.rename(columns={df.columns[0]: "team"})

        df["team"] = df["team"].map(normalize_team_name)
        df = df.loc[df["team"].notna()].copy()
        df = df.drop_duplicates(subset=["team"], keep="first").reset_index(drop=True)
        df = df.set_index("team", drop=False).sort_index()
        return df

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

    def get_team_profile(self, team: str) -> dict:
        team_n = normalize_team_name(team)
        if team_n in self.teams_df.index:
            row = self.teams_df.loc[team_n]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            profile = row.to_dict()
        else:
            profile = {"team": team_n}

        profile["team"] = team_n
        return profile

    def build_simulation_profile(
        self,
        team: str,
        fixture_context: dict | None = None,
        matchday: int | None = None,
        is_home: bool | None = None,
    ) -> dict:
        fixture_context = fixture_context or {}
        base = self.get_team_profile(team)
        profile = dict(base)

        team_n = normalize_team_name(team)
        profile["team"] = team_n

        # Core model fields used by poisson_engine.py
        profile["attack_prior"] = self._safe_float(
            profile.get("attack_prior", profile.get("attack", profile.get("attack_posterior_mean", 0.0))),
            0.0,
        )
        profile["defense_prior"] = self._safe_float(
            profile.get("defense_prior", profile.get("defense", profile.get("defense_posterior_mean", 0.0))),
            0.0,
        )
        profile["elo_prior"] = self._safe_float(
            profile.get("elo_prior", profile.get("current_elo", profile.get("elo", 0.0))),
            0.0,
        )
        profile["market_prior"] = self._safe_float(
            profile.get("market_prior", profile.get("market_value_in_eur", profile.get("total_market_value", 0.0))),
            0.0,
        )
        profile["form_prior"] = self._safe_float(profile.get("form_prior", profile.get("form", 0.0)), 0.0)
        profile["experience_prior"] = self._safe_float(
            profile.get("experience_prior", profile.get("experience", profile.get("international_caps", 0.0))),
            0.0,
        )
        profile["host_prior"] = self._safe_float(profile.get("host_prior", 1.0 if team_n in HOST_TEAMS else 0.0), 0.0)
        profile["travel_prior"] = self._safe_float(profile.get("travel_prior", 0.0), 0.0)
        profile["scenario_bias_prior"] = self._safe_float(profile.get("scenario_bias_prior", 0.0), 0.0)

        # Optional layers
        profile["average_age"] = self._safe_float(profile.get("average_age", profile.get("age", 27.0)), 27.0)
        profile["squad_size"] = self._safe_float(profile.get("squad_size", profile.get("national_team_players", 23.0)), 23.0)
        profile["squad_depth"] = self._safe_float(profile.get("squad_depth", profile["squad_size"]), profile["squad_size"])
        profile["cumulative_minutes"] = self._safe_float(
            profile.get("cumulative_minutes", profile.get("minutes_played", profile.get("minutes", 0.0))),
            0.0,
        )
        profile["days_rest"] = self._safe_float(
            fixture_context.get(
                "days_rest",
                {1: 5.0, 2: 4.0, 3: 3.0}.get(matchday, 4.0),
            ),
            4.0,
        )
        profile["climate_tolerance"] = self._safe_float(profile.get("climate_tolerance", 0.0), 0.0)
        profile["discipline"] = self._safe_float(profile.get("discipline", 0.0), 0.0)

        # If no travel prior exists, create a small proxy from the fixture context.
        travel_proxy = self._safe_float(fixture_context.get("travel_intensity_proxy", 1.0), 1.0)
        if profile["travel_prior"] == 0.0 and not profile["host_prior"]:
            profile["travel_prior"] = max(0.0, 0.05 * travel_proxy)

        # Small venue/weather-aware adjustment hooks
        venue_climate = self._safe_str(fixture_context.get("venue_climate", "Temperate"), "Temperate")
        if venue_climate.lower() in {"hot", "humid", "tropical"} and profile["climate_tolerance"] == 0.0:
            profile["climate_tolerance"] = -0.05 if team_n not in HOST_TEAMS else 0.03

        # Keep a few aliases for other modules
        profile["current_elo"] = profile["elo_prior"]
        profile["elo"] = profile["elo_prior"]
        profile["market_value_in_eur"] = profile.get("market_value_in_eur", profile["market_prior"])
        profile["team_class"] = self._safe_str(profile.get("team_class", profile.get("tier", "")), "")

        return profile

    def _fixture_rows(self) -> pd.DataFrame:
        if self.fixture_features is None or self.fixture_features.empty:
            return pd.DataFrame()

        fx = self.fixture_features.copy()

        for col in [
            "match_id",
            "matchday",
            "stage",
            "group",
            "home_team",
            "away_team",
            "venue_city",
            "venue_country",
            "venue_stadium",
            "time_et",
            "date_et",
            "travel_intensity_proxy",
            "temperature_c",
            "humidity_pct",
            "altitude_m",
            "wind_kph",
            "venue_climate",
            "days_rest",
        ]:
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

    @staticmethod
    def _stable_seed(*parts) -> int:
        key = "|".join("" if p is None else str(p) for p in parts)
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        return int(digest, 16) % (2**32)

    @classmethod
    def _referee_profile(cls, row: pd.Series) -> dict:
        seed = cls._stable_seed(
            row.get("match_id"),
            row.get("group"),
            row.get("venue_stadium"),
            row.get("time_et"),
            row.get("date_et"),
        )
        rng = np.random.default_rng(seed)
        return {
            "yellows_per_game": float(np.clip(rng.normal(4.1, 0.45), 2.5, 6.5)),
            "reds_per_game": float(np.clip(rng.normal(0.20, 0.07), 0.0, 0.6)),
            "fouls_per_game": float(np.clip(rng.normal(24.5, 2.5), 18.0, 33.0)),
        }

    @classmethod
    def _random_weather_profile(cls, seed: int, venue_name: str) -> dict:
        rng = np.random.default_rng(seed)
        venue_l = str(venue_name).lower()

        base_temp = 24.0
        base_humidity = 55.0
        base_altitude = 200.0
        base_wind = 10.0
        venue_climate = "Temperate"

        if any(k in venue_l for k in ["mexico", "mexico city"]):
            base_temp = 27.0
            base_altitude = 2240.0
            venue_climate = "Hot"
        elif any(k in venue_l for k in ["monterrey"]):
            base_temp = 30.0
            base_altitude = 540.0
            venue_climate = "Hot"
        elif any(k in venue_l for k in ["guadalajara"]):
            base_temp = 28.0
            base_altitude = 1550.0
            venue_climate = "Warm"
        elif any(k in venue_l for k in ["vancouver", "toronto", "seattle"]):
            base_temp = 18.0
            base_humidity = 62.0
            venue_climate = "Temperate"

        return {
            "temperature_c": float(np.clip(rng.normal(base_temp, 3.0), 10.0, 38.0)),
            "humidity_pct": float(np.clip(rng.normal(base_humidity, 8.0), 20.0, 90.0)),
            "altitude_m": float(np.clip(rng.normal(base_altitude, 140.0), 0.0, 2500.0)),
            "wind_kph": float(np.clip(rng.normal(base_wind, 4.0), 0.0, 35.0)),
            "venue_climate": venue_climate,
        }

    def generate_fixtures(self, matchday: int):
        """
        Return fixtures in official schedule order, not group-assignment order.
        Requires fixture_features.csv to contain matchday and match_id if you want exact day-by-day replication.
        """
        fx = self._fixture_rows()

        if fx.empty:
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
                "days_rest": {1: 5.0, 2: 4.0, 3: 3.0}.get(matchday, 4.0),
                "venue_name": "",
                "venue_city": "",
                "venue_country": "",
                "venue_stadium": "",
                "date_et": None,
                "time_et": "",
                "travel_intensity_proxy": 1.0,
                "temperature_c": 24.0,
                "humidity_pct": 55.0,
                "altitude_m": 0.0,
                "wind_kph": 10.0,
                "venue_climate": "Temperate",
                "referee_profile": {
                    "yellows_per_game": 4.1,
                    "reds_per_game": 0.2,
                    "fouls_per_game": 24.5,
                },
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
                "days_rest": {1: 5.0, 2: 4.0, 3: 3.0}.get(matchday, 4.0),
                "venue_name": "",
                "venue_city": "",
                "venue_country": "",
                "venue_stadium": "",
                "date_et": None,
                "time_et": "",
                "travel_intensity_proxy": 1.0,
                "temperature_c": 24.0,
                "humidity_pct": 55.0,
                "altitude_m": 0.0,
                "wind_kph": 10.0,
                "venue_climate": "Temperate",
                "referee_profile": {
                    "yellows_per_game": 4.1,
                    "reds_per_game": 0.2,
                    "fouls_per_game": 24.5,
                },
            }

        row = candidates.sort_values(["match_id"]).iloc[0]

        venue_city = str(
            _default_if_missing(row.get("venue_city"), "")
        ).strip()

        venue_stadium = str(
            _default_if_missing(row.get("venue_stadium"), "")
        ).strip()

        venue_country = str(
            _default_if_missing(row.get("venue_country"), "")
        ).strip()
        
        venue_name = venue_stadium or venue_city or ""

        days_rest = row.get("days_rest", pd.NA)
        if pd.isna(days_rest):
            days_rest = {1: 5.0, 2: 4.0, 3: 3.0}.get(matchday, 4.0)

        venue_climate = row.get("venue_climate", "Temperate")

        if pd.isna(venue_climate) or not str(venue_climate).strip():
            venue_climate = "Temperate"
        return {
            "stage": "group",
            "group": group,
            "matchday": matchday,
            "match_id": int(row["match_id"]) if pd.notna(row.get("match_id")) else None,
            "days_rest": float(days_rest),
            "venue_name": venue_name,
            "venue_city": venue_city,
            "venue_country": venue_country,
            "venue_stadium": venue_stadium,
            "time_et": str(
                _default_if_missing(row.get("time_et"), "")
                ).strip(),

            "venue_climate": str(
                _default_if_missing(row.get("venue_climate"), "Temperate")
                ),

            "travel_intensity_proxy": float(
                _default_if_missing(row.get("travel_intensity_proxy"), 1.0)
            ),

            "temperature_c": float(
                _default_if_missing(row.get("temperature_c"), 24.0)
            ),
             
            "humidity_pct": float(
                _default_if_missing(row.get("humidity_pct"), 55.0)
            ),

            "altitude_m": float(
                _default_if_missing(row.get("altitude_m"), 0.0)
            ),

            "wind_kph": float(
                _default_if_missing(row.get("wind_kph"), 10.0)
            ),
            
            "referee_profile": self._referee_profile(row),
        }

    def build_knockout_context(self, match_number: int, stage: str, venue_name: str) -> dict:
        seed = self._stable_seed(match_number, stage, venue_name)
        weather = self._random_weather_profile(seed, venue_name)
        ref_rng = np.random.default_rng(seed + 17)

        days_rest = 4.0
        if stage in {"semi_finals", "final"}:
            days_rest = 5.0

        return {
            "stage": stage,
            "match_id": match_number,
            "days_rest": days_rest,
            "venue_name": venue_name,
            "venue_city": venue_name,
            "venue_country": "",
            "venue_stadium": venue_name,
            "travel_intensity_proxy": 1.0,
            **weather,
            "referee_profile": {
                "yellows_per_game": float(np.clip(ref_rng.normal(4.0, 0.4), 2.6, 6.2)),
                "reds_per_game": float(np.clip(ref_rng.normal(0.18, 0.06), 0.0, 0.6)),
                "fouls_per_game": float(np.clip(ref_rng.normal(24.0, 2.2), 18.0, 33.0)),
            },
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
