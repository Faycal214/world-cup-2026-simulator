from __future__ import annotations

TEAM_CLASSIFICATION = {
    "Title_Contender": [
        "Argentina", "Brazil", "France", "Germany", "England",
        "Spain", "Portugal", "Netherlands",
    ],
    "Good_Team": [
        "Japan", "United States", "Mexico", "Uruguay", "Switzerland",
        "Austria", "Turkey", "South Korea", "Belgium",
    ],
    "Balanced": [
        "Scotland", "Iran", "Morocco", "Ecuador", "Australia", "Paraguay",
        "Ivory Coast", "South Africa", "Senegal", "Ghana", "Croatia",
        "Bosnia and Herzegovina", "Qatar", "Czechia", "New Zealand", "Tunisia",
        "Algeria",
    ],
    "Underdog": [
        "Haiti", "Cape Verde", "Panama", "Curaçao", "Uzbekistan", "Iraq",
        "Jordan", "Norway", "DR Congo", "Saudi Arabia", "Canada",
    ],
}

TEAM_MENTALITY = {
    "Title_Contender": {
        "vs_Title_Contender": 1.10,
        "vs_Good_Team": 1.05,
        "vs_Balanced": 1.00,
        "vs_Underdog": 0.92,
    },
    "Good_Team": {
        "vs_Title_Contender": 0.88,
        "vs_Good_Team": 1.00,
        "vs_Balanced": 1.04,
        "vs_Underdog": 1.02,
    },
    "Balanced": {
        "vs_Title_Contender": 0.82,
        "vs_Good_Team": 0.95,
        "vs_Balanced": 1.00,
        "vs_Underdog": 1.08,
    },
    "Underdog": {
        "vs_Title_Contender": 0.74,
        "vs_Good_Team": 0.86,
        "vs_Balanced": 0.98,
        "vs_Underdog": 1.00,
    },
}

TEAM_CLIMATE_PROFILES = {
    "Mexico": "Hot/Humid",
    "South Africa": "Temperate",
    "South Korea": "Temperate",
    "Czechia": "Cool/Temperate",
    "Canada": "Cool/Temperate",
    "Bosnia and Herzegovina": "Cool/Temperate",
    "Qatar": "Hot/Dry",
    "Switzerland": "Cool/Temperate",
    "Brazil": "Tropical",
    "Morocco": "Hot/Dry",
    "Haiti": "Tropical",
    "Scotland": "Cool/Temperate",
    "United States": "Temperate",
    "Paraguay": "Temperate",
    "Australia": "Temperate",
    "Turkey": "Temperate",
    "Germany": "Cool/Temperate",
    "Curaçao": "Tropical",
    "Ivory Coast": "Tropical",
    "Ecuador": "Tropical",
    "Netherlands": "Cool/Temperate",
    "Japan": "Temperate",
    "Sweden": "Cool/Temperate",
    "Tunisia": "Hot/Dry",
    "Belgium": "Cool/Temperate",
    "Egypt": "Hot/Dry",
    "Iran": "Hot/Dry",
    "New Zealand": "Temperate",
    "Spain": "Temperate",
    "Cape Verde": "Hot/Dry",
    "Saudi Arabia": "Hot/Dry",
    "Uruguay": "Temperate",
    "France": "Temperate",
    "Senegal": "Tropical",
    "Iraq": "Hot/Dry",
    "Norway": "Cool/Temperate",
    "Argentina": "Temperate",
    "Algeria": "Hot/Dry",
    "Austria": "Cool/Temperate",
    "Jordan": "Hot/Dry",
    "Portugal": "Temperate",
    "DR Congo": "Tropical",
    "Uzbekistan": "Hot/Dry",
    "Colombia": "Tropical",
    "England": "Cool/Temperate",
    "Croatia": "Temperate",
    "Ghana": "Tropical",
    "Panama": "Tropical",
}

def get_team_class(team_name: str) -> str:
    for tier, teams in TEAM_CLASSIFICATION.items():
        if team_name in teams:
            return tier
    return "Balanced"

def get_mentality_modifier(attacking_team_class: str, defending_team_class: str) -> float:
    key = f"vs_{defending_team_class}"
    return TEAM_MENTALITY.get(attacking_team_class, {}).get(key, 1.0)