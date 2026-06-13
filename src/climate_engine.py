from __future__ import annotations

from src.naming import normalize_team_name
from src.team_mentality import TEAM_CLIMATE_PROFILES
from src.venue_data import VENUE_METADATA

def get_climate_factor(team_profile: dict, venue_name: str) -> dict:
    venue_info = VENUE_METADATA.get(venue_name, {})
    if not venue_info:
        return {"factor": 1.0, "reason": "Unknown venue", "type": "neutral"}

    team_country = normalize_team_name(team_profile.get("country", team_profile.get("team", "")))
    venue_country = normalize_team_name(venue_info.get("country", ""))
    team_climate = team_profile.get(
        "climate_profile",
        TEAM_CLIMATE_PROFILES.get(team_country, "Temperate"),
    )
    venue_climate = venue_info.get("climate", "Temperate")
    altitude_m = float(venue_info.get("altitude_m", 0) or 0)

    if team_country and venue_country and team_country == venue_country:
        return {
            "factor": 1.10,
            "reason": f"Home-country adaptation ({team_country})",
            "type": "home",
        }

    factor = 1.0
    reason_parts = []

    if "Hot" in venue_climate and "Cool" in team_climate:
        factor *= 0.88
        reason_parts.append(f"{team_climate} → {venue_climate}")
    elif "Tropical" in team_climate and "Cool" in venue_climate:
        factor *= 0.94
        reason_parts.append(f"{team_climate} → {venue_climate}")
    elif "Temperate" in team_climate and "Hot" in venue_climate:
        factor *= 0.90
        reason_parts.append(f"{team_climate} → {venue_climate}")

    if altitude_m >= 1500 and team_country != venue_country:
        factor *= 0.95
        reason_parts.append(f"high altitude {altitude_m:.0f}m")

    if not reason_parts:
        return {
            "factor": 1.0,
            "reason": f"Climate neutral ({team_climate} → {venue_climate})",
            "type": "neutral",
        }

    return {
        "factor": max(0.75, min(factor, 1.15)),
        "reason": ", ".join(reason_parts),
        "type": "penalty",
    }