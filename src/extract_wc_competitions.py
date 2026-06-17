from __future__ import annotations

from pathlib import Path
import pandas as pd

from config import RAW_DIR, PROCESSED_DIR, GROUPS
from naming import normalize_team_name

START_DATE = pd.Timestamp("2023-01-01")
END_DATE = pd.Timestamp("2026-06-10")

def get_worldcup_team_set() -> set[str]:
    teams = []
    for group_teams in GROUPS.values():
        teams.extend(group_teams)
    return {normalize_team_name(t) for t in teams}

def main() -> None:
    matches = pd.read_csv(RAW_DIR / "matches.csv")

    required = {"date", "home_team", "away_team", "tournament"}
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"matches.csv is missing columns: {sorted(missing)}")

    matches = matches.copy()
    matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
    matches["home_team"] = matches["home_team"].astype(str).map(normalize_team_name)
    matches["away_team"] = matches["away_team"].astype(str).map(normalize_team_name)
    matches["tournament"] = matches["tournament"].astype(str).str.strip()

    wc_teams = get_worldcup_team_set()

    filtered = matches.loc[
        (matches["date"] >= START_DATE)
        & (matches["date"] <= END_DATE)
        & (
            matches["home_team"].isin(wc_teams)
            | matches["away_team"].isin(wc_teams)
        )
    ].copy()

    filtered = filtered.sort_values("date").reset_index(drop=True)

    competitions = (
        filtered["tournament"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    competitions = competitions.loc[competitions.ne("")].drop_duplicates().sort_values().reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(PROCESSED_DIR / "filtered_wc_matches.csv", index=False)
    pd.DataFrame({"tournament": competitions}).to_csv(
        PROCESSED_DIR / "competition_names.csv",
        index=False,
    )

    print(f"Filtered matches: {len(filtered)}")
    print(f"Unique competitions: {len(competitions)}")
    print(competitions.to_string(index=False))

if __name__ == "__main__":
    main()