"""
Climate adaptation engine for the 2026 World Cup simulator.
Calculates how team performance is affected by venue climate conditions.
"""

from src.venue_data import VENUE_METADATA, TEAM_CLIMATE_PROFILES


def get_climate_factor(team_profile: dict, venue_name: str) -> dict:
    """
    Calculate climate adaptation factor and impact details.

    Returns:
        dict with 'factor' (float) and 'reason' (str) explaining the adjustment
    """
    venue_info = VENUE_METADATA.get(venue_name, {})
    if not venue_info:
        return {'factor': 1.0, 'reason': 'Unknown venue'}

    team_country = team_profile.get('country', '')
    team_climate = team_profile.get('climate_profile', 'Temperate')
    venue_climate = venue_info.get('climate', 'Temperate')
    venue_country = venue_info.get('country', '')

    if team_country == venue_country:
        return {
            'factor': 1.15,
            'reason': f'Home advantage ({team_country})',
            'type': 'home'
        }

    penalty_matrix = {
        ('Cool', 'Hot/Humid'): 0.85,
        ('Cool', 'Hot/Dry'): 0.87,
        ('Cool/Temperate', 'Hot/Humid'): 0.88,
        ('Cool/Temperate', 'Hot/Dry'): 0.90,
        ('Temperate', 'Hot/Humid'): 0.85,
        ('Temperate', 'Hot/Dry'): 0.88,
        ('Cool/Dry', 'Hot/Humid'): 0.87,
        ('Cool/Dry', 'Hot/Dry'): 0.95,
    }

    penalty = penalty_matrix.get((team_climate, venue_climate))
    if penalty and penalty < 1.0:
        return {
            'factor': penalty,
            'reason': f'{team_climate} team in {venue_climate} venue',
            'type': 'penalty'
        }

    return {
        'factor': 1.0,
        'reason': f'Climate neutral ({team_climate} → {venue_climate})',
        'type': 'neutral'
    }


def format_climate_impact(team_a_profile: dict, team_b_profile: dict, venue_name: str) -> str:
    """Format climate impact for pre-match reporting"""
    impact_a = get_climate_factor(team_a_profile, venue_name)
    impact_b = get_climate_factor(team_b_profile, venue_name)

    reason_a = impact_a['reason']
    reason_b = impact_b['reason']

    result = f"{team_a_profile['country']} ({reason_a}), {team_b_profile['country']} ({reason_b})"
    return result
