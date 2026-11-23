"""Replay one layout many times and append each finished fight to CSV files."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from src.ai import LLMClient, LLMError, play_turn
from src.encounter import Encounter

FIGHT_FIELDS = [
    "fight_id",
    "label",
    "policy",
    "seed",
    "winner",
    "rounds",
    "party_size",
    "enemy_count",
    "party_hp_remaining",
    "enemy_hp_remaining",
    "party_down",
    "enemy_down",
    "damage_by_party",
    "damage_by_enemy",
    "llm_turns",
    "fallback_turns",
    "mean_enemy_cr",
    "mean_attack_bonus",
    "start_distance",
    "party_win",
    "party_slots",
    "enemy_slots",
    "party_slot_levels",
    "enemy_slot_levels",
    "party_slots_by_level",
    "enemy_slots_by_level",
]

TURN_FIELDS = [
    "fight_id",
    "round",
    "actor_id",
    "actor_name",
    "side",
    "action_kind",
    "action_label",
    "source",
    "target_id",
    "damage",
    "healing",
    "detail",
]


def _append_row(path: Path, fields: list[str], row: dict) -> None:
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def _write_status(run_dir: Path, completed: int, total: int, state: str, detail: str = "") -> None:
    payload = {"completed": completed, "total": total, "state": state, "detail": detail}
    (run_dir / "status.json").write_text(json.dumps(payload), encoding="utf-8")


def run_batch(
    setup: dict,
    *,
    count: int,
    policy: str,
    seed: int,
    run_dir: Path,
    client: LLMClient | None = None,
    on_progress=None,
    should_stop=None,
) -> dict:
    """Run ``count`` fights. Party members use the heuristic. Enemies use ``policy``."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "setup.json").write_text(json.dumps(setup), encoding="utf-8")
    if policy == "llm" and client is None:
        client = LLMClient()
    fights_path = run_dir / "fights.csv"
    turns_path = run_dir / "turns.csv"
    completed = 0
    _write_status(run_dir, 0, count, "running")
    for index in range(count):
        if should_stop and should_stop():
            _write_status(run_dir, completed, count, "stopped")
            break
        fight_setup = json.loads(json.dumps(setup))
        fight_setup["seed"] = seed + index
        fight_setup["enemy_policy"] = policy
        encounter = Encounter.from_setup(fight_setup)
        encounter.reaction_policy = "heuristic"
        encounter.start()
        safety = 0
        while not encounter.over and safety < 500:
            safety += 1
            if should_stop and should_stop():
                encounter.over = True
                encounter.winner = encounter.winner or "stopped"
                break
            encounter.begin_turn()
            if encounter.over:
                break
            actor = encounter.current()
            actor_policy = policy if actor.side == "enemy" else "heuristic"
            try:
                play_turn(encounter, actor_policy, client if actor_policy == "llm" else None)
            except LLMError:
                _write_status(run_dir, completed, count, "error", "Missing or rejected API key.")
                raise
        if not encounter.over:
            encounter.over = True
            encounter.winner = encounter.winner or "draw"
        fight_id = f"{run_dir.name}-{index + 1}"
        _append_row(fights_path, FIGHT_FIELDS, encounter.summary(fight_id))
        for turn in encounter.turns:
            _append_row(turns_path, TURN_FIELDS, {"fight_id": fight_id, **turn})
        completed += 1
        _write_status(run_dir, completed, count, "running")
        if on_progress:
            on_progress(completed, count, encounter.winner)
    if (run_dir / "status.json").exists():
        current = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        if current.get("state") == "running":
            _write_status(run_dir, completed, count, "done")
    return {"fights": str(fights_path), "turns": str(turns_path), "completed": completed}
