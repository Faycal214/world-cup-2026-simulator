import pandas as pd
import numpy as np
from tabulate import tabulate
from src.climate_engine import format_climate_impact

class TournamentEngine:
    def __init__(self, teams_df: pd.DataFrame):
        self.teams_df = teams_df
        self.group_tables = {}
        self.group_assignments = {}
        self.flag_map = {}  # Dictionary to handle flag display seamlessly
        self.initialize_groups()
        
    def initialize_groups(self):
        # 2026 Official FIFA World Cup Groups mapping
        group_mapping = {
            'A': ['🇲🇽 Mexico', '🇿🇦 South Africa', '🇰🇷 South Korea', '🇨🇿 Czechia'],
            'B': ['🇨🇦 Canada', '🇧🇦 Bosnia and Herzegovina', '🇶🇦 Qatar', '🇨🇭 Switzerland'],
            'C': ['🇧🇷 Brazil', '🇲🇦 Morocco', '🇭🇹 Haiti', '🏴󠁧󠁢󠁳󠁣󠁴󠁿 Scotland'],
            'D': ['🇺🇸 United States', '🇵🇾 Paraguay', '🇦🇺 Australia', '🇹🇷 Turkey'],
            'E': ['🇩🇪 Germany', '🇨🇼 Curaçao', '🇨🇮 Ivory Coast', '🇪🇨 Ecuador'],
            'F': ['🇳🇱 Netherlands', '🇯🇵 Japan', '🇸🇪 Sweden', '🇹🇳 Tunisia'],
            'G': ['🇧🇪 Belgium', '🇪🇬 Egypt', '🇮🇷 Iran', '🇳🇿 New Zealand'],
            'H': ['🇪🇸 Spain', '🇨🇻 Cape Verde', '🇸🇦 Saudi Arabia', '🇺🇾 Uruguay'],
            'I': ['🇫🇷 France', '🇸🇳 Senegal', '🇮🇶 Iraq', '🇳🇴 Norway'],
            'J': ['🇦🇷 Argentina', '🇩🇿 Algeria', '🇦🇹 Austria', '🇯🇴 Jordan'],
            'K': ['🇵🇹 Portugal', '🇨🇩 DR Congo', '🇺🇿 Uzbekistan', '🇨🇴 Colombia'],
            'L': ['🏴󠁧󠁢󠁥󠁮󠁧󠁿 England', '🇭🇷 Croatia', '🇬🇭 Ghana', '🇵🇦 Panama']
        }
        
        for g_letter, team_list in group_mapping.items():
            valid_teams = []
            for t in team_list:
                # Extract the raw country name by splitting off the emoji flag
                country_name = t.split(' ', 1)[1]
                self.flag_map[country_name] = t
                valid_teams.append(country_name)
            
            self.group_assignments[g_letter] = valid_teams
            self.group_tables[g_letter] = pd.DataFrame({
                'Team': valid_teams, 'MP': 0, 'W': 0, 'D': 0, 'L': 0,
                'GF': 0, 'GA': 0, 'GD': 0, 'Pts': 0
            }).set_index('Team')

    def generate_fixtures(self, matchday: int):
        fixtures = []
        for g_letter, teams in self.group_assignments.items():
            if matchday == 1:
                fixtures.append((g_letter, teams[0], teams[1]))
                fixtures.append((g_letter, teams[2], teams[3]))
            elif matchday == 2:
                fixtures.append((g_letter, teams[0], teams[2]))
                fixtures.append((g_letter, teams[1], teams[3]))
            elif matchday == 3:
                fixtures.append((g_letter, teams[0], teams[3]))
                fixtures.append((g_letter, teams[1], teams[2]))
        return fixtures

    def update_table(self, group: str, team_a: str, team_b: str, sa: int, sb: int):
        ta = self.group_tables[group]
        ta.loc[team_a, 'MP'] += 1
        ta.loc[team_b, 'MP'] += 1
        ta.loc[team_a, 'GF'] += sa
        ta.loc[team_a, 'GA'] += sb
        ta.loc[team_b, 'GF'] += sb
        ta.loc[team_b, 'GA'] += sa
        ta.loc[team_a, 'GD'] = ta.loc[team_a, 'GF'] - ta.loc[team_a, 'GA']
        ta.loc[team_b, 'GD'] = ta.loc[team_b, 'GF'] - ta.loc[team_b, 'GA']
        
        if sa > sb:
            ta.loc[team_a, 'Pts'] += 3; ta.loc[team_a, 'W'] += 1; ta.loc[team_b, 'L'] += 1
        elif sb > sa:
            ta.loc[team_b, 'Pts'] += 3; ta.loc[team_b, 'W'] += 1; ta.loc[team_a, 'L'] += 1
        else:
            ta.loc[team_a, 'Pts'] += 1; ta.loc[team_b, 'Pts'] += 1; ta.loc[team_a, 'D'] += 1; ta.loc[team_b, 'D'] += 1

    def display_tables(self):
        for g_letter, table in self.group_tables.items():
            sorted_table = table.sort_values(by=['Pts', 'GD', 'GF'], ascending=False)

            # Create a display copy to inject emoji flags back into the index seamlessly
            display_table = sorted_table.copy()
            display_table.index = display_table.index.map(lambda x: self.flag_map.get(x, x))

            print(f"\n--- Group {g_letter} Standings ---")
            print(tabulate(display_table, headers='keys', tablefmt='psql'))

    @staticmethod
    def pre_match_report(home: str, away: str, context: dict = None):
        """
        Print match preview in sports broadcast format (simplified, single line).

        Format: [STAGE] | [VENUE] | [HOME] vs [AWAY] | STAKES: [Stake1] / [Stake2]

        Args:
            home: Home team name (with emoji flag)
            away: Away team name (with emoji flag)
            context: Dict with optional 'stage', 'venue_name', 'stakes_home', 'stakes_away'
        """
        if not context:
            return

        stage = context.get('stage', 'Match').replace('_', ' ').title()
        venue = context.get('venue_name', 'TBD')
        stakes_home = context.get('stakes_home', 'Regular')
        stakes_away = context.get('stakes_away', 'Regular')

        print(f"[{stage}] | {venue} | {home} vs {away} | STAKES: {stakes_home} / {stakes_away}")