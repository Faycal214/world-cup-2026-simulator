from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass
class LiveUpdateConfig:
    points_win: int = 3
    points_draw: int = 1
    points_loss: int = 0


class LiveUpdateEngine:
    """
    Applies real tournament results to simulated tables.

    Useful when you want to:
    - replace simulated group-stage results with real ones,
    - rerun the rest of the tournament,
    - keep the forecast live.
    """

    def __init__(self, config: LiveUpdateConfig | None = None):
        self.config = config or LiveUpdateConfig()

    @staticmethod
    def load_results_csv(path: str | Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        required = {"match_id", "home_team", "away_team", "home_goals", "away_goals"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"results file missing columns: {sorted(missing)}")
        return df

    def apply_match_result(
        self,
        group_table: pd.DataFrame,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int,
    ) -> pd.DataFrame:
        table = group_table.copy()

        for team, gf, ga in [(home_team, home_goals, away_goals), (away_team, away_goals, home_goals)]:
            if team not in table.index:
                raise KeyError(f"Team not found in group table: {team}")

        # Update matches played
        table.loc[home_team, "MP"] += 1
        table.loc[away_team, "MP"] += 1

        # Update goals
        table.loc[home_team, "GF"] += int(home_goals)
        table.loc[home_team, "GA"] += int(away_goals)
        table.loc[away_team, "GF"] += int(away_goals)
        table.loc[away_team, "GA"] += int(home_goals)

        # Update results
        if home_goals > away_goals:
            table.loc[home_team, "W"] += 1
            table.loc[home_team, "Pts"] += self.config.points_win
            table.loc[away_team, "L"] += 1
            table.loc[away_team, "Pts"] += self.config.points_loss
        elif away_goals > home_goals:
            table.loc[away_team, "W"] += 1
            table.loc[away_team, "Pts"] += self.config.points_win
            table.loc[home_team, "L"] += 1
            table.loc[home_team, "Pts"] += self.config.points_loss
        else:
            table.loc[home_team, "D"] += 1
            table.loc[away_team, "D"] += 1
            table.loc[home_team, "Pts"] += self.config.points_draw
            table.loc[away_team, "Pts"] += self.config.points_draw

        table["GD"] = table["GF"] - table["GA"]
        return table

    def apply_results_to_group_tables(
        self,
        group_tables: dict[str, pd.DataFrame],
        results_df: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        """
        results_df should contain:
        - group
        - home_team
        - away_team
        - home_goals
        - away_goals
        """
        out = {g: tbl.copy() for g, tbl in group_tables.items()}

        required = {"group", "home_team", "away_team", "home_goals", "away_goals"}
        missing = required - set(results_df.columns)
        if missing:
            raise ValueError(f"results_df missing columns: {sorted(missing)}")

        for _, row in results_df.iterrows():
            g = str(row["group"])
            if g not in out:
                continue
            out[g] = self.apply_match_result(
                out[g],
                str(row["home_team"]),
                str(row["away_team"]),
                int(row["home_goals"]),
                int(row["away_goals"]),
            )

        return out

    def apply_results_from_csv(
        self,
        group_tables: dict[str, pd.DataFrame],
        path: str | Path,
    ) -> dict[str, pd.DataFrame]:
        results_df = self.load_results_csv(path)
        return self.apply_results_to_group_tables(group_tables, results_df)