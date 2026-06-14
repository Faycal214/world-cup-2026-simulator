import pandas as pd

features = pd.read_csv("data/processed/team_features.csv")

world_cup_teams = [
"Mexico","South Africa","South Korea","Czechia",
"Canada","Bosnia and Herzegovina","Qatar","Switzerland",
"Brazil","Morocco","Haiti","Scotland",
"United States","Paraguay","Australia","Turkey",
"Germany","Curaçao","Ivory Coast","Ecuador",
"Netherlands","Japan","Sweden","Tunisia",
"Belgium","Egypt","Iran","New Zealand",
"Spain","Cape Verde","Saudi Arabia","Uruguay",
"France","Senegal","Iraq","Norway",
"Argentina","Algeria","Austria","Jordan",
"Portugal","DR Congo","Uzbekistan","Colombia",
"England","Croatia","Ghana","Panama"
]

existing = set(features["team"])

missing = sorted(set(world_cup_teams) - existing)

print("Missing teams:")
print(missing)