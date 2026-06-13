import os
from src.data_loader import DataProcessor
from src.poisson_engine import MatchSimulator
from src.tournament_engine import TournamentEngine
from src.knockout_engine import KnockoutEngine
from src.venue_data import VENUE_METADATA, GROUP_STAGE_VENUES, get_venue_by_group
from src.stakes_engine import calculate_stakes, format_stakes_display

def print_tree(stage_name: str, matches: list):
    print(f"\n==================== {stage_name.upper()} RESULTS ====================")
    for m in matches:
        print(f"{m['home']} {m['score_home']} - {m['score_away']} {m['away']} {f'({m['meta']})' if m['meta'] else ''}")

def main():
    # Path configuration - Adjust if necessary for your environment
    csv_path = os.path.join("data", "elo_ratings_wc2026.csv")

    # Mock fallback dataset generation for independent file testing
    if not os.path.exists(csv_path):
        os.makedirs("data", exist_ok=True)
        import pandas as pd

        # Exact 48 Official Qualified Teams mapped to dataset naming conventions
        mock_teams = [
            'Mexico', 'South Africa', 'South Korea', 'Czechia',
            'Canada', 'Bosnia and Herzegovina', 'Qatar', 'Switzerland',
            'Brazil', 'Morocco', 'Haiti', 'Scotland',
            'United States', 'Paraguay', 'Australia', 'Turkey',
            'Germany', 'Curaçao', 'Ivory Coast', 'Ecuador',
            'Netherlands', 'Japan', 'Sweden', 'Tunisia',
            'Belgium', 'Egypt', 'Iran', 'New Zealand',
            'Spain', 'Cape Verde', 'Saudi Arabia', 'Uruguay',
            'France', 'Senegal', 'Iraq', 'Norway',
            'Argentina', 'Algeria', 'Austria', 'Jordan',
            'Portugal', 'DR Congo', 'Uzbekistan', 'Colombia',
            'England', 'Croatia', 'Ghana', 'Panama'
        ]

        mock_data = []
        for t in mock_teams:
            for y in range(2022, 2027):
                mock_data.append({
                    'year': y, 'snapshot_date': f"{y}-12-31", 'country': t, 'country_code': t[:2].upper(),
                    'rating': 1600 + (100 if t in ['Argentina', 'France', 'Brazil'] else 0),
                    'goals_for': 40, 'goals_against': 30, 'matches_total': 25,
                    'is_host': 1 if t in ['Mexico', 'United States', 'Canada'] else 0
                })
        pd.DataFrame(mock_data).to_csv(csv_path, index=False)

    print("🚀 Initializing 2026 FIFA World Cup Analytics Engine...")
    processor = DataProcessor(csv_path)
    teams_df = processor.load_and_calibrate()

    tournament = TournamentEngine(teams_df)
    simulator = MatchSimulator()

    # --- GROUP STAGE LOGIC LOOP ---
    for matchday in [1, 2, 3]:
        print(f"\n⚡ LIVE TOURNAMENT FEED: SIMULATING MATCHDAY {matchday}")
        fixtures = tournament.generate_fixtures(matchday)

        for group, home, away in fixtures:
            profile_home = teams_df[teams_df['country'] == home].iloc[0].to_dict()
            profile_away = teams_df[teams_df['country'] == away].iloc[0].to_dict()

            # Calculate stakes for both teams
            home_pts = tournament.group_tables[group].loc[home, 'Pts']
            away_pts = tournament.group_tables[group].loc[away, 'Pts']

            stakes_home = calculate_stakes(
                home, home_pts, matchday,
                'group_stage_final' if matchday == 3 else 'group',
                profile_home.get('classification')
            )

            stakes_away = calculate_stakes(
                away, away_pts, matchday,
                'group_stage_final' if matchday == 3 else 'group',
                profile_away.get('classification')
            )

            venue_name = get_venue_by_group(group, matchday)

            # Pre-match report (simplified format)
            context_report = {
                'stage': f'Group {group} - Matchday {matchday}',
                'venue_name': venue_name,
                'stakes_home': format_stakes_display(stakes_home),
                'stakes_away': format_stakes_display(stakes_away)
            }

            TournamentEngine.pre_match_report(
                tournament.flag_map.get(home, home),
                tournament.flag_map.get(away, away),
                context_report
            )

            # Match simulation with team classes and stakes
            context_sim = {
                'stage': 'group_stage_final' if matchday == 3 else 'group',
                'venue_name': venue_name
            }

            res = simulator.simulate_match(
                profile_home, profile_away, context_sim,
                team_class_a=profile_home.get('classification'),
                team_class_b=profile_away.get('classification'),
                stakes_a=stakes_home,
                stakes_b=stakes_away
            )

            tournament.update_table(group, home, away, res['score_a'], res['score_b'])

            # Result output (simplified)
            result_str = f"RESULT: {res['score_a']} - {res['score_b']}"
            if res['extra_time']:
                result_str += " (AET)"
            if res['penalty_winner']:
                winner = home if res['penalty_winner'] == 'A' else away
                result_str += f" ({winner} on PKs)"
            print(f"{result_str}\n")

        tournament.display_tables()
        input(f"▶️ Matchday {matchday} Complete. Press Enter to proceed...")

    # --- ADVANCEMENT COMPUTATIONS ---
    top2_qualifiers = {}
    for g_letter, table in tournament.group_tables.items():
        sorted_table = table.sort_values(by=['Pts', 'GD', 'GF'], ascending=False)
        top2_qualifiers[g_letter] = sorted_table.index[:2].tolist()

    adv_3rd, groups_3rd_set = KnockoutEngine.extract_and_rank_third_places(tournament.group_tables)
    print("\n⭐ SUCCESSFUL THIRD-PLACE ADVANCING TEAMS (Ranked):")
    for i, team in enumerate(adv_3rd, 1):
        print(f"  {i}. {tournament.flag_map.get(team, team)}")

    # --- TOURNAMENT VALIDATION ---
    total_group_qualifiers = len(top2_qualifiers) * 2
    total_3rd_qualifiers = len(adv_3rd)
    total_teams = total_group_qualifiers + total_3rd_qualifiers

    assert total_group_qualifiers == 24, f"Expected 24 group qualifiers, got {total_group_qualifiers}"
    assert total_3rd_qualifiers == 8, f"Expected 8 third-place qualifiers, got {total_3rd_qualifiers}"
    assert total_teams == 32, f"Expected 32 total teams for Round of 32, got {total_teams}"
    print(f"\n✅ TOURNAMENT VALIDATION PASSED: {total_group_qualifiers} group qualifiers + {total_3rd_qualifiers} third-place qualifiers = {total_teams} teams for Round of 32\n")

    # --- KNOCKOUT SETUP ---
    r32_pairings = KnockoutEngine.get_round_of_32_pairings(top2_qualifiers, adv_3rd, groups_3rd_set)

    def run_knockout_stage(pairings, stage_name, stage_context, venue_mapper=None):
        next_stage_teams = []
        display_logs = []
        for home, away in pairings:
            p_home = teams_df[teams_df['country'] == home].iloc[0].to_dict()
            p_away = teams_df[teams_df['country'] == away].iloc[0].to_dict()

            # Assign venue for knockout matches
            venue_name = venue_mapper(home, away) if venue_mapper else 'Neutral'

            # Pre-match report (knockout uses standard stakes)
            context_report = {
                'stage': stage_name,
                'venue_name': venue_name,
                'stakes_home': 'Knockout',
                'stakes_away': 'Knockout'
            }

            TournamentEngine.pre_match_report(
                tournament.flag_map.get(home, home),
                tournament.flag_map.get(away, away),
                context_report
            )

            # Match simulation with team classes
            context_sim = {
                'stage': stage_context,
                'venue_name': venue_name
            }

            res = simulator.simulate_match(
                p_home, p_away, context_sim,
                team_class_a=p_home.get('classification'),
                team_class_b=p_away.get('classification'),
                stakes_a='Knockout_Tension',
                stakes_b='Knockout_Tension'
            )

            meta_str = ""
            if res['extra_time']:
                meta_str = "AET"
            if res['penalty_winner']:
                winner = home if res['penalty_winner'] == 'A' else away
                meta_str += f" ({winner} on PKs)"
            else:
                winner = home if res['score_a'] > res['score_b'] else away

            next_stage_teams.append(winner)

            home_flagged = tournament.flag_map.get(home, home)
            away_flagged = tournament.flag_map.get(away, away)

            display_logs.append({
                'home': home_flagged, 'away': away_flagged,
                'score_home': res['score_a'], 'score_away': res['score_b'], 'meta': meta_str
            })

            # Simplified result output
            result_str = f"RESULT: {res['score_a']} - {res['score_b']}"
            if res['extra_time']:
                result_str += " (AET)"
            if res['penalty_winner']:
                result_str += f" ({meta_str})"
            print(f"{result_str}\n")

        return next_stage_teams, display_logs

    def get_knockout_venue(home, away):
        """Rotate through major venues for knockout matches"""
        venues = ['Miami', 'Dallas', 'New York', 'Mexico City', 'Los Angeles', 'Houston']
        hash_val = hash((home, away)) % len(venues)
        return venues[hash_val]

    # Execute Bracket Tree
    r32_winners, logs = run_knockout_stage(r32_pairings, "Round of 32", "round_of_32", get_knockout_venue)
    print_tree("Round of 32", logs)
    r16_pairings = list(zip(r32_winners[::2], r32_winners[1::2]))

    r16_winners, logs = run_knockout_stage(r16_pairings, "Round of 16", "round_of_16", get_knockout_venue)
    print_tree("Round of 16", logs)
    qf_pairings = list(zip(r16_winners[::2], r16_winners[1::2]))

    qf_winners, logs = run_knockout_stage(qf_pairings, "Quarter-Finals", "quarter_finals", get_knockout_venue)
    print_tree("Quarter-Finals", logs)
    sf_pairings = list(zip(qf_winners[::2], qf_winners[1::2]))

    sf_winners, logs = run_knockout_stage(sf_pairings, "Semi-Finals", "semi_finals", get_knockout_venue)
    print_tree("Semi-Finals", logs)
    final_pairing = [(sf_winners[0], sf_winners[1])]

    champion, logs = run_knockout_stage(final_pairing, "World Cup Final", "final", get_knockout_venue)
    print_tree("World Cup Final", logs)

    champ_flagged = tournament.flag_map.get(champion[0], champion[0])
    print(f"\n🏆 🏆 🏆 THE 2026 WORLD CUP CHAMPION IS: {champ_flagged.upper()} 🏆 🏆 🏆")

if __name__ == "__main__":
    main()