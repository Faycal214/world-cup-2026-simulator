from src.data_loader import load_team_profiles, load_fixture_features
from src.simulator import WorldCupSimulator


def main() -> None:
    team_profiles = load_team_profiles()
    fixture_features = load_fixture_features()

    sim = WorldCupSimulator(
        team_profiles=team_profiles,
        fixture_features=fixture_features,
        random_state=42,
    )
    sim.run()


if __name__ == "__main__":
    main()