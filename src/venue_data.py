"""
2026 FIFA World Cup Venue & Climate Data
Extracted from official group stage schedule and Annex C tournament bracket
"""

VENUE_METADATA = {
    'Mexico City': {'climate': 'Hot/Humid', 'country': 'Mexico', 'altitude_m': 2250},
    'Guadalajara': {'climate': 'Temperate', 'country': 'Mexico', 'altitude_m': 1640},
    'Monterrey': {'climate': 'Hot/Humid', 'country': 'Mexico', 'altitude_m': 640},
    'Toronto': {'climate': 'Cool/Temperate', 'country': 'Canada', 'altitude_m': 75},
    'Vancouver': {'climate': 'Cool/Temperate', 'country': 'Canada', 'altitude_m': 0},
    'Seattle': {'climate': 'Cool/Temperate', 'country': 'USA', 'altitude_m': 0},
    'Las Vegas': {'climate': 'Hot/Dry', 'country': 'USA', 'altitude_m': 600},
    'Phoenix': {'climate': 'Hot/Dry', 'country': 'USA', 'altitude_m': 320},
    'Denver': {'climate': 'Cool/Dry', 'country': 'USA', 'altitude_m': 1600},
    'Dallas': {'climate': 'Hot/Humid', 'country': 'USA', 'altitude_m': 130},
    'Houston': {'climate': 'Hot/Humid', 'country': 'USA', 'altitude_m': 0},
    'Miami': {'climate': 'Hot/Humid', 'country': 'USA', 'altitude_m': 0},
    'Atlanta': {'climate': 'Temperate', 'country': 'USA', 'altitude_m': 300},
    'Boston': {'climate': 'Cool/Temperate', 'country': 'USA', 'altitude_m': 0},
    'Philadelphia': {'climate': 'Temperate', 'country': 'USA', 'altitude_m': 10},
    'New York': {'climate': 'Cool/Temperate', 'country': 'USA', 'altitude_m': 0},
    'Los Angeles': {'climate': 'Temperate', 'country': 'USA', 'altitude_m': 100},
    'San Francisco': {'climate': 'Cool/Temperate', 'country': 'USA', 'altitude_m': 50},
    'Kansas City': {'climate': 'Temperate', 'country': 'USA', 'altitude_m': 250},
    'Santa Fe': {'climate': 'Cool/Dry', 'country': 'USA', 'altitude_m': 2100},
    'Inglewood': {'climate': 'Temperate', 'country': 'USA', 'altitude_m': 100},
}

TEAM_CLIMATE_PROFILES = {
    'Mexico': 'Hot/Humid',
    'South Africa': 'Temperate',
    'South Korea': 'Temperate',
    'Czechia': 'Cool/Temperate',
    'Canada': 'Cool/Temperate',
    'Bosnia and Herzegovina': 'Cool/Temperate',
    'Qatar': 'Hot/Dry',
    'Switzerland': 'Cool/Temperate',
    'Brazil': 'Tropical',
    'Morocco': 'Hot/Dry',
    'Haiti': 'Tropical',
    'Scotland': 'Cool/Temperate',
    'United States': 'Temperate',
    'Paraguay': 'Temperate',
    'Australia': 'Temperate',
    'Turkey': 'Temperate',
    'Germany': 'Cool/Temperate',
    'Curaçao': 'Tropical',
    'Ivory Coast': 'Tropical',
    'Ecuador': 'Tropical',
    'Netherlands': 'Cool/Temperate',
    'Japan': 'Temperate',
    'Sweden': 'Cool/Temperate',
    'Tunisia': 'Hot/Dry',
    'Belgium': 'Cool/Temperate',
    'Egypt': 'Hot/Dry',
    'Iran': 'Hot/Dry',
    'New Zealand': 'Temperate',
    'Spain': 'Temperate',
    'Cape Verde': 'Hot/Dry',
    'Saudi Arabia': 'Hot/Dry',
    'Uruguay': 'Temperate',
    'France': 'Temperate',
    'Senegal': 'Tropical',
    'Iraq': 'Hot/Dry',
    'Norway': 'Cool/Temperate',
    'Argentina': 'Temperate',
    'Algeria': 'Hot/Dry',
    'Austria': 'Cool/Temperate',
    'Jordan': 'Hot/Dry',
    'Portugal': 'Temperate',
    'DR Congo': 'Tropical',
    'Uzbekistan': 'Hot/Dry',
    'Colombia': 'Tropical',
    'England': 'Cool/Temperate',
    'Croatia': 'Temperate',
    'Ghana': 'Tropical',
    'Panama': 'Tropical',
}

GROUP_STAGE_VENUES = {
    'A': ['Mexico City', 'Monterrey'],
    'B': ['Toronto', 'Vancouver'],
    'C': ['Los Angeles', 'San Francisco'],
    'D': ['Dallas', 'Houston'],
    'E': ['Kansas City', 'Denver'],
    'F': ['Denver', 'Kansas City'],
    'G': ['Las Vegas', 'Phoenix'],
    'H': ['Atlanta', 'Philadelphia'],
    'I': ['Boston', 'New York'],
    'J': ['New York', 'Boston'],
    'K': ['Miami', 'Atlanta'],
    'L': ['Guadalajara', 'Mexico City'],
}

KNOCKOUT_BRACKET_SCHEMA = {
    'round_of_32': {
        1: {'home': ('A', 1), 'away': ('B', 2)},
        2: {'home': ('B', 1), 'away': ('A', 2)},
        3: {'home': ('C', 1), 'away': ('D', 2)},
        4: {'home': ('D', 1), 'away': ('C', 2)},
        5: {'home': ('E', 1), 'away': ('F', 2)},
        6: {'home': ('F', 1), 'away': ('E', 2)},
        7: {'home': ('G', 1), 'away': ('H', 2)},
        8: {'home': ('H', 1), 'away': ('G', 2)},
        9: {'home': ('I', 1), 'away': ('I', 2)},
        10: {'home': ('J', 1), 'away': ('J', 2)},
        11: {'home': ('K', 1), 'away': ('K', 2)},
        12: {'home': ('L', 1), 'away': ('L', 2)},
        13: {'home': ('3rd', 1), 'away': ('A', 1)},
        14: {'home': ('3rd', 2), 'away': ('C', 1)},
        15: {'home': ('3rd', 3), 'away': ('E', 1)},
        16: {'home': ('3rd', 4), 'away': ('G', 1)},
    },
    'round_of_16': {
        1: {'from_match': (1, 16)},
        2: {'from_match': (8, 9)},
        3: {'from_match': (5, 12)},
        4: {'from_match': (4, 13)},
        5: {'from_match': (6, 11)},
        6: {'from_match': (3, 14)},
        7: {'from_match': (7, 10)},
        8: {'from_match': (2, 15)},
    },
    'quarter_finals': {
        1: {'from_match': (1, 5)},
        2: {'from_match': (4, 8)},
        3: {'from_match': (2, 6)},
        4: {'from_match': (3, 7)},
    },
    'semi_finals': {
        1: {'from_match': (1, 3)},
        2: {'from_match': (2, 4)},
    },
    'final': {
        1: {'from_match': (1, 2)},
    }
}

def get_all_venues():
    """Return list of all unique venues"""
    return list(VENUE_METADATA.keys())

def get_venue_by_group(group: str, matchday: int):
    """Return venue for a specific group matchday"""
    if group in GROUP_STAGE_VENUES:
        venues = GROUP_STAGE_VENUES[group]
        return venues[(matchday - 1) % len(venues)]
    return 'Neutral'
