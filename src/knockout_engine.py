from __future__ import annotations

from pathlib import Path
import pandas as pd

BRACKET_MAP_PATH = Path("data/raw/r32_bracket_map.csv")


class KnockoutEngine:
    @staticmethod
    def extract_and_rank_third_places(group_tables: dict) -> tuple[list[str], set[str]]:
        third_placed_teams = []
        for g_letter, table in group_tables.items():
            ranked = table.sort_values(by=["Pts", "GD", "GF", "GA"], ascending=[False, False, False, True])
            third_team_name = ranked.index[2]
            row = ranked.iloc[2].to_dict()
            row["Team"] = third_team_name
            row["Group"] = g_letter
            third_placed_teams.append(row)

        df_3rd = pd.DataFrame(third_placed_teams).sort_values(
            by=["Pts", "GD", "GF", "GA", "Team"],
            ascending=[False, False, False, True, True],
        ).reset_index(drop=True)

        advancing_3rd = df_3rd.head(8)
        return advancing_3rd["Team"].tolist(), set(advancing_3rd["Group"].tolist())

    @staticmethod
    def _resolve_source(source: str, qualified_top2: dict, advancing_3rd: list[str]) -> str:
        source = str(source).strip().upper()

        if source.startswith("3R"):
            idx = int(source[2:]) - 1
            return advancing_3rd[idx]

        group = source[0]
        seed = int(source[1:])
        return qualified_top2[group][seed - 1]

    @staticmethod
    def get_round_of_32_pairings(
        qualified_top2: dict,
        advancing_3d: list[str],
        groups_3rd_set: set,
    ) -> list[tuple[str, str]]:
        """
        Reads the 16-slot Round of 32 template from data/raw/r32_bracket_map.csv.
        The CSV must contain columns:
          slot, home_source, away_source
        with sources like A1, B2, 3R1, etc.
        """
        bracket = pd.read_csv(BRACKET_MAP_PATH)

        required = {"slot", "home_source", "away_source"}
        missing = required - set(bracket.columns)
        if missing:
            raise ValueError(f"r32_bracket_map.csv missing columns: {sorted(missing)}")

        bracket = bracket.sort_values("slot").reset_index(drop=True)

        pairings = []
        for _, row in bracket.iterrows():
            home = KnockoutEngine._resolve_source(row["home_source"], qualified_top2, advancing_3d)
            away = KnockoutEngine._resolve_source(row["away_source"], qualified_top2, advancing_3d)
            pairings.append((home, away))

        return pairings

    @staticmethod
    def get_round_of_16_pairings(r32_winners_by_match: dict[int, str]) -> list[tuple[int, str, str]]:
        return [
            (89, r32_winners_by_match[74], r32_winners_by_match[77]),
            (90, r32_winners_by_match[73], r32_winners_by_match[75]),
            (91, r32_winners_by_match[76], r32_winners_by_match[78]),
            (92, r32_winners_by_match[79], r32_winners_by_match[80]),
            (93, r32_winners_by_match[83], r32_winners_by_match[84]),
            (94, r32_winners_by_match[81], r32_winners_by_match[82]),
            (95, r32_winners_by_match[86], r32_winners_by_match[88]),
            (96, r32_winners_by_match[85], r32_winners_by_match[87]),
        ]

    @staticmethod
    def get_quarter_final_pairings(r16_winners_by_match: dict[int, str]) -> list[tuple[int, str, str]]:
        return [
            (97, r16_winners_by_match[89], r16_winners_by_match[90]),
            (98, r16_winners_by_match[93], r16_winners_by_match[94]),
            (99, r16_winners_by_match[91], r16_winners_by_match[92]),
            (100, r16_winners_by_match[95], r16_winners_by_match[96]),
        ]

    @staticmethod
    def get_semi_final_pairings(qf_winners_by_match: dict[int, str]) -> list[tuple[int, str, str]]:
        return [
            (101, qf_winners_by_match[97], qf_winners_by_match[98]),
            (102, qf_winners_by_match[99], qf_winners_by_match[100]),
        ]

    @staticmethod
    def get_third_place_playoff(sf_losers_by_match: dict[int, str]) -> tuple[int, str, str]:
        return (103, sf_losers_by_match[101], sf_losers_by_match[102])

    @staticmethod
    def get_final_pairing(sf_winners_by_match: dict[int, str]) -> tuple[int, str, str]:
        return (104, sf_winners_by_match[101], sf_winners_by_match[102])