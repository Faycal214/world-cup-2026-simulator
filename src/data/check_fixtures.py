import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
fixtures = pd.read_csv(ROOT / "data" / "raw" / "fixtures_raw.csv")

print(fixtures.shape)
print(fixtures.head())