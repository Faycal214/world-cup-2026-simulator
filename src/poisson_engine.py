import numpy as np
from src.climate_engine import get_climate_factor
from src.team_mentality import get_mentality_modifier
from src.stakes_engine import get_stakes_modifier

# Bias coefficients for realistic match outcomes
REFEREE_BIAS_COEFFICIENT = 0.15  # Host nation referee bias
CHAMPION_LUCK_COEFFICIENT = 0.10  # Argentina reigning champion luck


def generate_score_from_probability(goal_expectancy: float) -> int:
    """
    Generate realistic goal count using probability distribution instead of raw Poisson.

    Base distribution (goal_expectancy = 2.0):
    - 0 goals: 20%
    - 1 goal: 35%
    - 2 goals: 25%
    - 3 goals: 15%
    - 4+ goals: 5%

    Adjusted by goal_expectancy scaling to reflect stronger/weaker teams.

    Args:
        goal_expectancy: Expected goal value (typically 0.5-4.0 after all modifiers)

    Returns:
        Number of goals (0-4+)
    """
    # Normalize to factor relative to base expectancy of 2.0
    exp_factor = np.clip(goal_expectancy / 2.0, 0.5, 2.0)

    # Base probabilities
    base_probs = [0.20, 0.35, 0.25, 0.15, 0.05]

    # Adjust probabilities by expectancy factor
    adjusted_probs = []
    for i, prob in enumerate(base_probs):
        if i == 0:
            # Lower expected goals → higher 0-goal probability
            adjusted_probs.append(base_probs[0] / (exp_factor ** 0.5))
        elif i <= 2:
            # Scale 1-2 goals linearly with expectancy
            adjusted_probs.append(base_probs[i] * exp_factor)
        else:
            # Diminishing returns for 3+ goals
            adjusted_probs.append(base_probs[i] * (exp_factor ** 0.7))

    # Normalize to probability distribution
    total = sum(adjusted_probs)
    probs = [p / total for p in adjusted_probs]

    # Draw from distribution
    return np.random.choice([0, 1, 2, 3, 4], p=probs)


class MatchSimulator:
    def __init__(self, elo_scaling_factor: float = 0.0015):
        self.c = elo_scaling_factor

    def calculate_lambda(
        self,
        team_profile: dict,
        opponent_profile: dict,
        context: dict,
        team_class: str = None,
        opponent_class: str = None,
        stakes: str = None
    ) -> float:
        """
        Calculate adjusted lambda for Poisson-like distribution using comprehensive modifiers.

        Formula: λ_final = λ_base × C_climate × S_stakes × M_mentality × B_bias × K_tension

        Args:
            team_profile: Attacking team's profile dict
            opponent_profile: Defending team's profile dict
            context: Match context (venue, stage, etc.)
            team_class: Team classification (Title_Contender, Good_Team, etc.)
            opponent_class: Opponent classification
            stakes: Match stakes (Must_Win, Playing_for_Pride, etc.)

        Returns:
            Adjusted lambda value for scoring intensity
        """
        elo_diff = team_profile['current_elo'] - opponent_profile['current_elo']

        # Host advantage (structural +100 Elo)
        if team_profile['is_host']:
            elo_diff += 100
        if opponent_profile['is_host']:
            elo_diff -= 100

        # Base lambda from Elo difference
        lambda_base = team_profile['alpha_base'] * opponent_profile['beta_base'] * np.exp(self.c * elo_diff)
        lambda_base = np.clip(lambda_base, 0.2, 5.0)

        # Climate modifier
        venue_name = context.get('venue_name', '')
        climate_modifier = 1.0
        if venue_name:
            climate_info = get_climate_factor(team_profile, venue_name)
            climate_modifier = climate_info['factor']

        # Stakes modifier
        stakes_modifier = 1.0
        if stakes:
            stakes_modifier = get_stakes_modifier(stakes)

        # Mentality modifier
        mentality_modifier = 1.0
        if team_class and opponent_class:
            mentality_modifier = get_mentality_modifier(team_class, opponent_class)

        # Bias modifier (referee bias + champion luck)
        bias_modifier = self._apply_bias_modifier(team_profile, opponent_profile, context)

        # Knockout tension modifier
        knockout_modifier = self._apply_knockout_modifier(context.get('stage', 'group'))

        # Calculate final lambda
        lambda_final = (
            lambda_base
            * climate_modifier
            * stakes_modifier
            * mentality_modifier
            * bias_modifier
            * knockout_modifier
        )

        return np.clip(lambda_final, 0.2, 5.0)

    def _apply_bias_modifier(self, team_profile: dict, opponent_profile: dict, context: dict) -> float:
        """Apply referee bias and luck modifiers"""
        bias = 1.0

        # Referee bias for hosts (+0.15 Elo advantage converted to lambda boost)
        if team_profile['is_host']:
            bias *= (1 + REFEREE_BIAS_COEFFICIENT * 0.05)  # ~5% goal expectancy boost

        # Champion's luck for Argentina (+0.10)
        if team_profile['country'] == 'Argentina':
            bias *= (1 + CHAMPION_LUCK_COEFFICIENT * 0.03)  # ~3% goal expectancy boost

        return bias

    def _apply_stakes_modifier(self, team_profile: dict, stage: str) -> float:
        """Apply motivation modifier based on match stakes"""
        if stage == 'group_stage_final' and team_profile.get('needs_points'):
            return 1.25
        elif stage == 'group_stage_final' and team_profile.get('already_qualified'):
            return 0.8

        return 1.0

    def _apply_knockout_modifier(self, stage: str) -> float:
        """Apply defensive caution in knockout matches"""
        if stage in ['round_of_32', 'round_of_16', 'quarter_finals', 'semi_finals', 'final']:
            return 0.8

        return 1.0

    def penalty_shootout(self, team_a: dict, team_b: dict) -> str:
        """
        Simulate penalty shootout based on Elo ratings.
        Returns 'A' or 'B' for the winner.
        """
        elo_diff = team_a['current_elo'] - team_b['current_elo']
        p_a = 0.70 + (elo_diff * 0.0001)
        p_a = np.clip(p_a, 0.50, 0.85)
        return 'A' if np.random.binomial(1, p_a) == 1 else 'B'

    def simulate_match(
        self,
        team_a: dict,
        team_b: dict,
        context: dict = None,
        team_class_a: str = None,
        team_class_b: str = None,
        stakes_a: str = None,
        stakes_b: str = None
    ) -> dict:
        """
        Simulate a match with realistic probability-based scoring.

        Args:
            team_a: Home team profile dict
            team_b: Away team profile dict
            context: Dict with 'venue_name', 'stage', and match context info
            team_class_a: Classification of team A
            team_class_b: Classification of team B
            stakes_a: Stakes for team A (Must_Win, etc.)
            stakes_b: Stakes for team B
        """
        if context is None:
            context = {'stage': 'group', 'venue_name': 'Neutral'}

        # Calculate adjusted lambdas with all modifiers
        lambda_a = self.calculate_lambda(
            team_a, team_b, context,
            team_class=team_class_a,
            opponent_class=team_class_b,
            stakes=stakes_a
        )

        lambda_b = self.calculate_lambda(
            team_b, team_a, {**context, 'is_away': True},
            team_class=team_class_b,
            opponent_class=team_class_a,
            stakes=stakes_b
        )

        # Generate 90-minute scores using probability distribution (realistic scoring)
        score_a = generate_score_from_probability(lambda_a)
        score_b = generate_score_from_probability(lambda_b)

        extra_time = False
        penalty_winner = None

        # Knockout tie-breaker resolution
        is_knockout = context.get('stage', 'group') not in ['group', 'group_stage_final']
        if is_knockout and score_a == score_b:
            extra_time = True
            # Extra time has reduced intensity (lambdas divided by 3)
            score_et_a = generate_score_from_probability(lambda_a / 3.0)
            score_et_b = generate_score_from_probability(lambda_b / 3.0)
            score_a += score_et_a
            score_b += score_et_b

            # Penalty shootout if still tied
            if score_a == score_b:
                penalty_winner = self.penalty_shootout(team_a, team_b)

        return {
            'score_a': score_a, 'score_b': score_b,
            'extra_time': extra_time, 'penalty_winner': penalty_winner
        }
