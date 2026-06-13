"""
Team Classification & Mentality System for 2026 World Cup Simulator
Maps teams into competitive tiers and defines behavioral modifiers based on matchups.
"""

# All 48 teams classified into competitive tiers
TEAM_CLASSIFICATION = {
    'Title_Contender': [
        'Brazil', 'France', 'Argentina', 'Germany', 'England', 'Spain', 'Portugal', 'Netherlands'
    ],
    'Good_Team': [
        'Japan', 'USA', 'Mexico', 'Uruguay', 'Switzerland', 'Austria', 'Turkey',
        'South Korea', 'Belgium', 'Denmark'  # Denmark is a strong team historically
    ],
    'Average_Team': [
        'Scotland', 'Iran', 'Morocco', 'Ecuador', 'Australia', 'Paraguay',
        'Ivory Coast', 'South Africa', 'Senegal', 'Ghana', 'Croatia', 'Bosnia and Herzegovina',
        'Qatar', 'Czechia', 'New Zealand', 'Tunisia'
    ],
    'Underdog': [
        'Haiti', 'Cape Verde', 'Panama', 'Curaçao', 'Uzbekistan', 'Iraq',
        'Algeria', 'Jordan', 'Norway', 'DR Congo', 'Curaçao', 'Saudi Arabia',
        'Canada'  # Canada qualified but is not historically strong
    ]
}

# Mentality modifiers - how attacking team's lambda multiplies vs opponent class
# Range: 0.7 (defensive/cautious) to 1.1 (aggressive/high-intensity)
TEAM_MENTALITY = {
    'Title_Contender': {
        'vs_Title_Contender': 1.1,   # High intensity, big match mentality
        'vs_Good_Team': 1.05,        # Elevated play but not max intensity
        'vs_Average_Team': 1.0,      # Normal play
        'vs_Underdog': 0.9           # Complacency penalty - risk of underestimation
    },
    'Good_Team': {
        'vs_Title_Contender': 0.85,  # Tactical caution and defensive respect
        'vs_Good_Team': 1.0,         # Fair fight, neutral
        'vs_Average_Team': 1.05,     # Slight pressure and confidence
        'vs_Underdog': 1.0           # Standard approach
    },
    'Average_Team': {
        'vs_Title_Contender': 0.8,   # Strong defensive approach, park the bus mentality
        'vs_Good_Team': 0.95,        # Tactical caution
        'vs_Average_Team': 1.0,      # Fair fight
        'vs_Underdog': 1.1           # Confident and aggressive at home
    },
    'Underdog': {
        'vs_Title_Contender': 0.7,   # The "Parking the Bus" effect - extreme defensive focus
        'vs_Good_Team': 0.85,        # Cautious defensive approach
        'vs_Average_Team': 1.0,      # Fair fight
        'vs_Underdog': 1.0           # Even match between underdogs
    }
}

def get_team_class(team_name: str) -> str:
    """
    Classify a team into one of four competitive tiers.

    Args:
        team_name: Official team name

    Returns:
        'Title_Contender', 'Good_Team', 'Average_Team', or 'Underdog'
    """
    for tier, teams in TEAM_CLASSIFICATION.items():
        if team_name in teams:
            return tier

    # Default fallback if team not found
    return 'Average_Team'


def get_mentality_modifier(attacking_team_class: str, defending_team_class: str) -> float:
    """
    Get mentality-based lambda multiplier for attacking team vs defending team class.

    Args:
        attacking_team_class: Classification of team trying to score
        defending_team_class: Classification of opponent

    Returns:
        Multiplier for lambda (0.7 to 1.1 range)
    """
    key = f'vs_{defending_team_class}'
    return TEAM_MENTALITY.get(attacking_team_class, {}).get(key, 1.0)


def format_mentality_info(team_class: str, opponent_class: str) -> str:
    """
    Format mentality information for display.

    Returns:
        String describing the matchup mentality
    """
    modifier = get_mentality_modifier(team_class, opponent_class)

    if modifier > 1.05:
        return "High Intensity"
    elif modifier > 1.0:
        return "Elevated Play"
    elif modifier >= 0.95:
        return "Normal Play"
    elif modifier >= 0.85:
        return "Cautious"
    else:
        return "Defensive Focus"
