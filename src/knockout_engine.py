import pandas as pd

class KnockoutEngine:
    @staticmethod
    def extract_and_rank_third_places(group_tables: dict) -> tuple:
        third_placed_teams = []
        for g_letter, table in group_tables.items():
            sorted_table = table.sort_values(by=['Pts', 'GD', 'GF'], ascending=False)
            third_team_name = sorted_table.index[2]
            row = sorted_table.iloc[2].to_dict()
            row['Team'] = third_team_name
            row['Group'] = g_letter
            third_placed_teams.append(row)
            
        df_3rd = pd.DataFrame(third_placed_teams)
        df_3rd = df_3rd.sort_values(by=['Pts', 'GD', 'GF'], ascending=False).reset_index(drop=True)
        
        advancing_3rd = df_3rd.head(8)
        return advancing_3rd['Team'].tolist(), set(advancing_3rd['Group'].tolist())

    @staticmethod
    def get_round_of_32_pairings(qualified_top2: dict, advancing_3rd: list, groups_3rd_set: set) -> list:
        """
        Generate exactly 16 matches for Round of 32:
        - 8 matches from groups A-H winners vs runners-up (cross-overs)
        - 4 matches from groups I-L winners vs runners-up (standard pairs)
        - 4 matches from 8 best third-placed teams paired among themselves
        """
        pairings = []

        # 8 matches: Groups A-H winners vs runners-up (cross-overs)
        pairings.append((qualified_top2['A'][0], qualified_top2['B'][1]))
        pairings.append((qualified_top2['B'][0], qualified_top2['A'][1]))
        pairings.append((qualified_top2['C'][0], qualified_top2['D'][1]))
        pairings.append((qualified_top2['D'][0], qualified_top2['C'][1]))
        pairings.append((qualified_top2['E'][0], qualified_top2['F'][1]))
        pairings.append((qualified_top2['F'][0], qualified_top2['E'][1]))
        pairings.append((qualified_top2['G'][0], qualified_top2['H'][1]))
        pairings.append((qualified_top2['H'][0], qualified_top2['G'][1]))

        # 4 matches: Groups I-L winners vs runners-up (standard pairs)
        pairings.append((qualified_top2['I'][0], qualified_top2['I'][1]))
        pairings.append((qualified_top2['J'][0], qualified_top2['J'][1]))
        pairings.append((qualified_top2['K'][0], qualified_top2['K'][1]))
        pairings.append((qualified_top2['L'][0], qualified_top2['L'][1]))

        # 4 matches: 8 best third-placed teams paired among themselves
        pool_3rd = list(advancing_3rd)
        for i in range(0, len(pool_3rd), 2):
            if i + 1 < len(pool_3rd):
                pairings.append((pool_3rd[i], pool_3rd[i + 1]))

        return pairings