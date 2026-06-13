from __future__ import annotations

VENUE_METADATA = {
    "Mexico City": {"country": "Mexico", "climate": "Hot/Humid", "altitude_m": 2250},
    "Guadalajara": {"country": "Mexico", "climate": "Temperate", "altitude_m": 1640},
    "Monterrey": {"country": "Mexico", "climate": "Hot/Humid", "altitude_m": 640},
    "Toronto": {"country": "Canada", "climate": "Cool/Temperate", "altitude_m": 75},
    "Vancouver": {"country": "Canada", "climate": "Cool/Temperate", "altitude_m": 0},
    "Seattle": {"country": "United States", "climate": "Cool/Temperate", "altitude_m": 0},
    "Los Angeles": {"country": "United States", "climate": "Temperate", "altitude_m": 100},
    "San Francisco Bay Area": {"country": "United States", "climate": "Cool/Temperate", "altitude_m": 50},
    "Dallas": {"country": "United States", "climate": "Hot/Humid", "altitude_m": 130},
    "Houston": {"country": "United States", "climate": "Hot/Humid", "altitude_m": 0},
    "Atlanta": {"country": "United States", "climate": "Temperate", "altitude_m": 300},
    "Boston": {"country": "United States", "climate": "Cool/Temperate", "altitude_m": 0},
    "Philadelphia": {"country": "United States", "climate": "Temperate", "altitude_m": 10},
    "New York/New Jersey": {"country": "United States", "climate": "Cool/Temperate", "altitude_m": 0},
    "Kansas City": {"country": "United States", "climate": "Temperate", "altitude_m": 250},
    "Miami": {"country": "United States", "climate": "Hot/Humid", "altitude_m": 0},
}

GROUP_STAGE_VENUES = {
    "A": ["Mexico City", "Monterrey"],
    "B": ["Toronto", "Vancouver"],
    "C": ["Los Angeles", "San Francisco Bay Area"],
    "D": ["Dallas", "Houston"],
    "E": ["Kansas City", "Seattle"],
    "F": ["Seattle", "Los Angeles"],
    "G": ["Miami", "New York/New Jersey"],
    "H": ["Atlanta", "Philadelphia"],
    "I": ["Boston", "New York/New Jersey"],
    "J": ["Guadalajara", "Mexico City"],
    "K": ["Houston", "Dallas"],
    "L": ["Philadelphia", "Atlanta"],
}

KNOCKOUT_ROTATION_VENUES = [
    "Miami",
    "Dallas",
    "New York/New Jersey",
    "Mexico City",
    "Los Angeles",
    "Houston",
    "Atlanta",
    "Philadelphia",
]

def get_venue_by_group(group: str, matchday: int) -> str:
    venues = GROUP_STAGE_VENUES.get(group, [])
    if not venues:
        return "Neutral"
    return venues[(matchday - 1) % len(venues)]

def get_knockout_venue(home: str, away: str) -> str:
    idx = abs(hash((home, away))) % len(KNOCKOUT_ROTATION_VENUES)
    return KNOCKOUT_ROTATION_VENUES[idx]