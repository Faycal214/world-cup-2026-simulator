from __future__ import annotations

import argparse
from bayesian_calibration import run_bayesian_calibration


def main() -> None:
    parser = argparse.ArgumentParser(description="Bayesian calibration for World Cup simulator")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--cores", type=int, default=2)
    parser.add_argument("--target_accept", type=float, default=0.92)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    priors, report, _ = run_bayesian_calibration(
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        cores=args.cores,
        target_accept=args.target_accept,
        seed=args.seed,
    )

    print(priors.head(10).to_string(index=False))
    print(report)


if __name__ == "__main__":
    main()