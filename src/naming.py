from __future__ import annotations

import re
from unidecode import unidecode

TEAM_ALIASES = {
    "usa": "United States",
    "u.s.a.": "United States",
    "united states of america": "United States",
    "united states": "United States",
    "korea republic": "South Korea",
    "republic of korea": "South Korea",
    "south korea": "South Korea",
    "czech republic": "Czechia",
    "czechia": "Czechia",
    "turkiye": "Turkey",
    "türkiye": "Turkey",
    "turkey": "Turkey",
    "cote d'ivoire": "Ivory Coast",
    "cote d ivoire": "Ivory Coast",
    "côte d'ivoire": "Ivory Coast",
    "ivory coast": "Ivory Coast",
    "congo dr": "DR Congo",
    "dr congo": "DR Congo",
    "democratic republic of congo": "DR Congo",
    "bosnia-herzegovina": "Bosnia and Herzegovina",
    "bosnia & herzegovina": "Bosnia and Herzegovina",
    "bosnia and herzegovina": "Bosnia and Herzegovina",
    "curacao": "Curaçao",
    "curaçao": "Curaçao",
    "cape verde": "Cape Verde",
    "saudi arabia": "Saudi Arabia",
    "new zealand": "New Zealand",
    "south africa": "South Africa",
    "cabo verde": "Cape Verde",
}

def normalize_text(value: str) -> str:
    if value is None:
        return ""
    s = unidecode(str(value))
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_team_name(value: str) -> str:
    s = normalize_text(value).lower()
    s = re.sub(r"\s*\([A-Z]{3}\)\s*$", "", s).strip()
    s = s.replace("&", "and")
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return TEAM_ALIASES.get(s, normalize_text(value))

def normalize_player_name(value: str) -> str:
    s = normalize_text(value)
    s = re.sub(r"\s*\([A-Z]{3}\)\s*$", "", s).strip()
    s = re.sub(r"[^A-Za-z0-9\s\-']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s