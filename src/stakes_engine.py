"""
Nuanced Stakes Calculator for 2026 World Cup Simulator
Determines match stakes and applies behavioral modifiers based on tournament context.
"""


def calculate_stakes(
    team_name: str,
    current_points: int,
    matches_played: int,
    stage: str,
    team_class: str = None
) -> str:
    """
    Determine match stakes for a team in a given tournament context.

    Args:
        team_name: Team name
        current_points: Current points in group (0-9)
        matches_played: Matches played so far (0-3)
        stage: Tournament stage ('group', 'group_stage_final', 'round_of_32', etc.)
        team_class: Team classification from team_mentality

    Returns:
        Stake category: 'Must_Win', 'Playing_for_1st', 'Playing_for_Pride',
                       'Knockout_Tension', 'High_Expectations'
    """

    # All knockout matches have same stakes
    if stage in ['round_of_32', 'round_of_16', 'quarter_finals', 'semi_finals', 'final']:
        return 'Knockout_Tension'

    # Group stage logic
    if matches_played < 3:
        # Earlier matchdays - Title Contenders face high expectations
        if team_class == 'Title_Contender':
            return 'High_Expectations'
        return 'Playing_for_1st'

    # Final matchday (matchday 3) - most nuanced stakes
    if current_points < 4:
        # 0-3 points: Elimination risk (0-1 win, or 1 draw max)
        return 'Must_Win'
    elif current_points >= 6:
        # 6+ points: Already qualified or fighting for 1st (3+ wins)
        return 'Playing_for_Pride'
    else:
        # 4-5 points: Mid-range teams fighting for qualification
        return 'Playing_for_1st'


def get_stakes_modifier(stakes: str) -> float:
    """
    Get lambda multiplier based on match stakes.

    Args:
        stakes: Stakes category from calculate_stakes()

    Returns:
        Lambda multiplier (0.7 to 1.25 range)
    """
    modifiers = {
        'Must_Win': 1.25,           # Desperate performance: +25% goal expectancy
        'Playing_for_1st': 1.1,     # Motivated: +10% goal expectancy
        'Playing_for_Pride': 0.7,   # Eliminated, reduced intensity: -30%
        'Knockout_Tension': 0.8,    # Defensive, high stakes: -20%
        'High_Expectations': 1.15   # Pressure to perform well: +15%
    }
    return modifiers.get(stakes, 1.0)


def format_stakes_display(stakes: str) -> str:
    """
    Format stakes for display in match reports.

    Returns:
        Human-readable stakes label
    """
    display_map = {
        'Must_Win': 'Must Win',
        'Playing_for_1st': 'Playing for 1st',
        'Playing_for_Pride': 'Playing for Pride',
        'Knockout_Tension': 'Knockout - Tension',
        'High_Expectations': 'High Expectations'
    }
    return display_map.get(stakes, stakes)
