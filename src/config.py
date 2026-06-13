from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

HOST_TEAMS = {"Mexico", "Canada", "United States"}

# Keep scenario bias at zero by default.
# Turn these on only for controlled experiments.
SCENARIO_BIAS = {
    "Argentina": 0.0,
    "Iran": 0.0,
    "Mexico": 0.0,
    "Canada": 0.0,
    "United States": 0.0,
}

GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czechia"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

GROUP_ORDER = list(GROUPS.keys())

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