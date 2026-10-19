from __future__ import annotations

import asyncio
import logging
import sys
import threading
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
TEMPLATE_DIR = ROOT / "templates"

# The existing engines use imports such as `from config import ...`.
# Put src/ on the path so those imports continue working without refactoring them.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

app = FastAPI(
    title="FWC26 Tournament Simulator",
    description="Live web interface for the existing World Cup 2026 simulator.",
    version="1.0.0",
)
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

simulation_lock = asyncio.Lock()
state_lock = threading.RLock()
LAST_RUN: dict[str, Any] = {
    "status": "idle",
    "seed": None,
    "champion": None,
    "matches_played": 0,
    "events": [],
    "error": None,
}

STAGE_LABELS = {
    "round_of_32": "Round of 32",
    "round_of_16": "Round of 16",
    "quarter_finals": "Quarter-finals",
    "semi_finals": "Semi-finals",
    "final": "Final",
}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request=request, name="index.html", context={})


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "FWC26 Tournament Simulator"}


@app.get("/api/status")
async def status() -> dict[str, Any]:
    with state_lock:
        return {
            "status": LAST_RUN["status"],
            "seed": LAST_RUN["seed"],
            "champion": LAST_RUN["champion"],
            "matches_played": LAST_RUN["matches_played"],
            "event_count": len(LAST_RUN["events"]),
            "error": LAST_RUN["error"],
        }


@app.get("/api/results")
async def results() -> dict[str, Any]:
    """Return the latest run's streamed events and final status."""
    with state_lock:
        return dict(LAST_RUN)


def _standings_payload(tournament: Any) -> dict[str, Any]:
    groups: list[dict[str, Any]] = []
    for group in tournament.group_tables:
        table = tournament.sorted_group_table(group).reset_index()
        rows = []
        for _, row in table.iterrows():
            rows.append(
                {
                    "team": str(row["Team"]),
                    "mp": int(row["MP"]),
                    "w": int(row["W"]),
                    "d": int(row["D"]),
                    "l": int(row["L"]),
                    "gf": int(row["GF"]),
                    "ga": int(row["GA"]),
                    "gd": int(row["GD"]),
                    "pts": int(row["Pts"]),
                }
            )
        groups.append({"group": group, "rows": rows})
    return {"type": "standings", "groups": groups}


def _run_existing_simulator(seed: int, emit: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    """Run the repository's existing tournament engine and publish structured events.

    The simulator is instrumented at its MatchSimulator boundary rather than copied
    or rewritten: its scoring model, group assignments, and knockout logic remain
    the source of truth.
    """
    from data_loader import load_fixture_features, load_team_profiles
    from simulator import WorldCupSimulator

    team_profiles = load_team_profiles()
    fixture_features = load_fixture_features()
    simulator = WorldCupSimulator(
        team_profiles=team_profiles,
        fixture_features=fixture_features,
        random_state=seed,
    )

    current_matchday: dict[str, int | None] = {"value": None}
    original_generate_fixtures = simulator.tournament.generate_fixtures

    def tracked_generate_fixtures(matchday: int):
        current_matchday["value"] = int(matchday)
        return original_generate_fixtures(matchday)

    simulator.tournament.generate_fixtures = tracked_generate_fixtures

    original_display_tables = simulator.tournament.display_tables

    def streamed_display_tables() -> None:
        original_display_tables()
        emit(_standings_payload(simulator.tournament))

    simulator.tournament.display_tables = streamed_display_tables

    original_simulate_match = simulator.match_simulator.simulate_match

    def streamed_simulate_match(
        team_a: dict,
        team_b: dict,
        context: dict | None = None,
        team_class_a: str | None = None,
        team_class_b: str | None = None,
        stakes_a: str | None = None,
        stakes_b: str | None = None,
    ) -> dict:
        result = original_simulate_match(
            team_a,
            team_b,
            context=context,
            team_class_a=team_class_a,
            team_class_b=team_class_b,
            stakes_a=stakes_a,
            stakes_b=stakes_b,
        )
        context = context or {}
        stage_key = str(context.get("stage", "group"))
        is_group = stage_key in {"group", "group_stage_final"}

        if is_group:
            group = str(context.get("group", ""))
            matchday = current_matchday["value"]
            stage_label = f"Group {group} · Matchday {matchday or '?'}"
            phase = "groups"
        else:
            group = None
            matchday = None
            stage_label = STAGE_LABELS.get(stage_key, stage_key.replace("_", " ").title())
            phase = "knockout"

        home = str(team_a.get("team", team_a.get("country", "Team A")))
        away = str(team_b.get("team", team_b.get("country", "Team B")))
        if result.get("penalty_winner") == "A":
            winner = home
        elif result.get("penalty_winner") == "B":
            winner = away
        elif result["score_a"] > result["score_b"]:
            winner = home
        elif result["score_b"] > result["score_a"]:
            winner = away
        else:
            winner = None

        penalty_score = result.get("penalty_score")
        event = {
            "type": "match",
            "phase": phase,
            "stage_key": "group" if is_group else stage_key,
            "stage": stage_label,
            "group": group,
            "matchday": matchday,
            "home": home,
            "away": away,
            "score_home": int(result["score_a"]),
            "score_away": int(result["score_b"]),
            "winner": winner,
            "extra_time": bool(result.get("extra_time", False)),
            "penalty_winner": winner if result.get("penalty_winner") else None,
            "penalty_score": list(penalty_score) if penalty_score is not None else None,
            "method": str(result.get("method", "90min")),
            "venue": str(context.get("venue_name", "Venue not recorded") or "Venue not recorded"),
        }
        emit(event)
        return result

    simulator.match_simulator.simulate_match = streamed_simulate_match
    simulator.run()

    with state_lock:
        champion = LAST_RUN.get("champion")
    return {"champion": champion, "matches_played": len([1 for e in LAST_RUN["events"] if e.get("type") == "match"])}


@app.websocket("/ws/simulate")
async def simulate(websocket: WebSocket) -> None:
    await websocket.accept()
    raw_seed = websocket.query_params.get("seed", "42")
    try:
        seed = int(raw_seed)
    except ValueError:
        seed = 42

    try:
        async with simulation_lock:
            with state_lock:
                LAST_RUN.update(
                    {
                        "status": "running",
                        "seed": seed,
                        "champion": None,
                        "matches_played": 0,
                        "events": [],
                        "error": None,
                    }
                )

            await websocket.send_json({"type": "status", "status": "running", "seed": seed})
            loop = asyncio.get_running_loop()

            def emit(event: dict[str, Any]) -> None:
                with state_lock:
                    LAST_RUN["events"].append(event)
                    if event.get("type") == "match":
                        LAST_RUN["matches_played"] += 1
                        if event.get("stage_key") == "final":
                            LAST_RUN["champion"] = event.get("winner")
                try:
                    asyncio.run_coroutine_threadsafe(websocket.send_json(event), loop).result(timeout=45)
                except Exception:
                    # Keep the simulation running if the browser closes the socket.
                    pass

            outcome = await asyncio.to_thread(_run_existing_simulator, seed, emit)
            with state_lock:
                LAST_RUN["status"] = "complete"
                LAST_RUN["champion"] = outcome.get("champion") or LAST_RUN.get("champion")
                LAST_RUN["matches_played"] = outcome.get("matches_played", LAST_RUN["matches_played"])
            await websocket.send_json(
                {
                    "type": "complete",
                    "status": "complete",
                    "champion": outcome.get("champion") or LAST_RUN.get("champion"),
                    "matches_played": outcome.get("matches_played", LAST_RUN["matches_played"]),
                }
            )
    except WebSocketDisconnect:
        logging.info("WebSocket client disconnected")
    except Exception as exc:
        logging.exception("Tournament simulation failed")
        message = str(exc) or exc.__class__.__name__
        with state_lock:
            LAST_RUN["status"] = "error"
            LAST_RUN["error"] = message
        try:
            await websocket.send_json({"type": "error", "message": message})
        except Exception:
            pass
        try:
            await websocket.close(code=1011)
        except Exception:
            pass
