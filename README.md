# FWC26 World Cup Simulator — Live Web UI

A FastAPI interface for the existing Python simulator. The web layer instruments the existing `WorldCupSimulator` at its match-simulation boundary, so the current goal model, fixture handling, group assignments and knockout logic remain the source of truth.

## Architecture

    app.py                 FastAPI routes, WebSocket event streaming, simulation adapter
    templates/index.html   Responsive dark UI, CSS and browser JavaScript
    src/                   Existing simulation engines (unchanged)
    data/                  Existing team, fixture and model inputs

## Run locally

From the repository root, using the project's existing virtual environment if available:

    python3 -m pip install -r requirements.txt
    uvicorn app:app --reload

Then open http://127.0.0.1:8000 in your browser and select **Simulate tournament**. You can change the random seed to reproduce a run.

The app needs the same input files as the command-line simulator, particularly `data/processed/team_features.csv`, `data/processed/team_strength_priors.csv`, and `data/processed/fixture_features.csv` (or the existing raw-fixture fallback).

## API

- `GET /` — web interface
- `GET /api/health` — health check
- `GET /api/status` — latest run status
- `GET /api/results` — latest streamed event history
- `WS /ws/simulate?seed=42` — starts a tournament and streams `status`, `match`, `standings`, `complete` or `error` JSON events

The simulator runs in a worker thread so the event loop can continue sending WebSocket updates. Each match event includes the phase, teams, score, venue, extra-time/penalty metadata and winner where applicable.

## Notes

- The existing tournament runner simulates the group stage, Round of 32, Round of 16, quarter-finals, semi-finals and final. This interface streams the stages that the engine currently implements; it does not add a separate third-place match.
- Results are stochastic and seed-based; they are simulations, not official FIFA predictions.
- A single tournament is run per connection at a time. Avoid starting a second run while one is active.
