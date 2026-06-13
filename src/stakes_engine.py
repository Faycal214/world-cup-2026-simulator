from __future__ import annotations

def calculate_stakes(
    team_name: str,
    current_points: int,
    matches_played: int,
    stage: str,
    team_class: str = None,
) -> str:
    if stage in ["round_of_32", "round_of_16", "quarter_finals", "semi_finals", "final"]:
        return "Knockout_Tension"

    if matches_played < 3:
        if team_class == "Title_Contender":
            return "High_Expectations"
        return "Playing_for_1st"

    if current_points < 4:
        return "Must_Win"
    if current_points >= 6:
        return "Playing_for_Pride"
    return "Playing_for_1st"

def get_stakes_modifier(stakes: str) -> float:
    modifiers = {
        "Must_Win": 1.25,
        "Playing_for_1st": 1.10,
        "Playing_for_Pride": 0.72,
        "Knockout_Tension": 0.84,
        "High_Expectations": 1.15,
    }
    return modifiers.get(stakes, 1.0)

def format_stakes_display(stakes: str) -> str:
    labels = {
        "Must_Win": "Must Win",
        "Playing_for_1st": "Playing for 1st",
        "Playing_for_Pride": "Playing for Pride",
        "Knockout_Tension": "Knockout",
        "High_Expectations": "High Expectations",
    }
    return labels.get(stakes, stakes)