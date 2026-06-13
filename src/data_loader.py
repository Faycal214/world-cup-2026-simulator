import pandas as pd
import numpy as np
from src.venue_data import TEAM_CLIMATE_PROFILES
from src.team_mentality import get_team_class

class DataProcessor:
    def __init__(self, filepath: str, current_year: int = 2026, gamma: float = 0.15):
        self.filepath = filepath
        self.current_year = current_year
        self.gamma = gamma
        
    def load_and_calibrate(self) -> pd.DataFrame:
        # Load Kaggle dataset
        df = pd.read_csv(self.filepath)
        
        # Calculate exponential decay weight based on the snapshot year
        df['weight'] = np.exp(-self.gamma * (self.current_year - df['year']))
        
        # Aggregate historical data using vectorized time-decay weights
        teams = df['country'].unique()
        calibrated_profiles = []
        
        for team in teams:
            team_data = df[df['country'] == team]
            latest_row = team_data.sort_values(by='snapshot_date').iloc[-1]
            
            # Weighted averages for goal capabilities
            total_weight = team_data['weight'].sum()
            weighted_gf = (team_data['goals_for'] * team_data['weight']).sum() / total_weight
            weighted_ga = (team_data['goals_against'] * team_data['weight']).sum() / total_weight
            weighted_matches = (team_data['matches_total'] * team_data['weight']).sum() / total_weight
            
            # Prevent division by zero
            base_attack = weighted_gf / max(weighted_matches, 1)
            base_defense = weighted_ga / max(weighted_matches, 1)
            
            calibrated_profiles.append({
                'country': team,
                'country_code': latest_row['country_code'],
                'current_elo': latest_row['rating'],
                'alpha_base': base_attack if base_attack > 0 else 1.0,
                'beta_base': base_defense if base_defense > 0 else 1.0,
                'is_host': latest_row['is_host'],
                'climate_profile': TEAM_CLIMATE_PROFILES.get(team, 'Temperate'),
                'classification': get_team_class(team)
            })
            
        return pd.DataFrame(calibrated_profiles)