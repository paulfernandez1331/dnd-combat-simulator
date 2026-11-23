"""Summary statistics for saved fight logs."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from src.dice import parse_dice

NUMERIC_COLUMNS = [
    "party_win",
    "rounds",
    "damage_by_party",
    "damage_by_enemy",
    "enemy_count",
    "mean_enemy_cr",
    "party_size",
    "mean_attack_bonus",
    "start_distance",
    "fallback_turns",
    "llm_turns",
    "policy_llm",
]


def load_runs(runs_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    fight_frames = []
    turn_frames = []
    if runs_dir.exists():
        for path in sorted(runs_dir.glob("*/fights.csv")):
            frame = pd.read_csv(path)
            frame["run_id"] = path.parent.name
            fight_frames.append(frame)
            turns_path = path.parent / "turns.csv"
            if turns_path.exists():
                turns = pd.read_csv(turns_path)
                turns["run_id"] = path.parent.name
                turn_frames.append(turns)
    fights = pd.concat(fight_frames, ignore_index=True) if fight_frames else pd.DataFrame()
    turns = pd.concat(turn_frames, ignore_index=True) if turn_frames else pd.DataFrame()
    return fights, turns


def _numeric(fights: pd.DataFrame) -> pd.DataFrame:
    frame = fights.copy()
    if "policy" in frame.columns:
        frame["policy_llm"] = (frame["policy"] == "llm").astype(int)
    for column in NUMERIC_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def win_rate(fights: pd.DataFrame) -> float | None:
    if fights.empty or "party_win" not in fights.columns:
        return None
    decided = pd.to_numeric(fights["party_win"], errors="coerce").dropna()
    if decided.empty:
        return None
    return float(decided.mean())


_DESCRIBE_STATS = ("count", "mean", "std", "min", "25%", "50%", "75%", "max")
_SUMMARY_MEASURES = (
    ("Fight", "Rounds", "rounds"),
    ("Party", "Damage dealt", "damage_by_party"),
    ("Enemy", "Damage dealt", "damage_by_enemy"),
    ("Party", "Hit points remaining", "party_hp_remaining"),
    ("Enemy", "Hit points remaining", "enemy_hp_remaining"),
)


def _empty_summary() -> pd.DataFrame:
    frame = pd.DataFrame(columns=list(_DESCRIBE_STATS))
    frame.index = pd.MultiIndex.from_tuples([], names=["Winner", "Measure", "Side"])
    return frame


def describe_fights(fights: pd.DataFrame) -> pd.DataFrame:
    """Describe rounds, damage, and hit points by winner, measure, and side.

    The index is ``(Winner, Measure, Side)`` and the columns are the usual
    ``describe`` statistics. Rounds stay on the Fight side. Winner ``all`` is
    the whole batch. A winner with no fights is left out, and party sits
    directly above enemy for the same measure.
    """
    if fights.empty:
        return _empty_summary()
    frame = _numeric(fights)
    pieces = []
    for side, measure, column in _SUMMARY_MEASURES:
        if column not in frame.columns:
            continue
        part = pd.DataFrame({
            "Measure": measure,
            "Side": side,
            "value": pd.to_numeric(frame[column], errors="coerce").to_numpy(),
        })
        if "winner" in frame.columns:
            part["winner"] = frame["winner"].to_numpy()
        pieces.append(part)
    if not pieces:
        return _empty_summary()
    long = pd.concat(pieces, ignore_index=True)
    blocks = [long.assign(Winner="all")]
    if "winner" in long.columns:
        labeled = long["winner"].map(lambda value: "" if pd.isna(value) else str(value))
        for winner in ("party", "enemy", "draw"):
            part = long.loc[labeled == winner]
            if not part.empty:
                blocks.append(part.assign(Winner=winner))
    described = pd.concat(blocks, ignore_index=True).groupby(
        ["Winner", "Measure", "Side"],
        sort=False,
    )["value"].describe()
    order = [
        (winner, measure, side)
        for winner in ("all", "party", "enemy", "draw")
        for measure, side in (
            ("Rounds", "Fight"),
            ("Damage dealt", "Party"),
            ("Damage dealt", "Enemy"),
            ("Hit points remaining", "Party"),
            ("Hit points remaining", "Enemy"),
        )
        if (winner, measure, side) in described.index
    ]
    described = described.reindex(pd.MultiIndex.from_tuples(order, names=["Winner", "Measure", "Side"]))
    return described.reindex(columns=list(_DESCRIBE_STATS))


def variation_notes(fights: pd.DataFrame) -> dict[str, str]:
    frame = _numeric(fights)
    notes = {}
    for column in NUMERIC_COLUMNS:
        if column not in frame.columns:
            continue
        if frame[column].nunique(dropna=True) < 2:
            notes[column] = "no variation across these fights, so correlation is undefined"
    return notes


def correlation_table(fights: pd.DataFrame) -> pd.DataFrame:
    frame = _numeric(fights)
    columns = [column for column in NUMERIC_COLUMNS if column in frame.columns]
    if len(frame) < 2 or not columns:
        return pd.DataFrame()
    return frame[columns].corr(method="pearson")


def latest_run_id(runs_dir: Path, session_dir: str | Path | None = None) -> str | None:
    """Folder name of the batch to show.

    A session directory that already contains fights.csv wins, even when a
    different run was written later. With no session directory, the newest
    fights.csv wins.
    """
    if session_dir:
        folder = Path(session_dir)
        if (folder / "fights.csv").is_file():
            return folder.name
        return None
    if not runs_dir.exists():
        return None
    candidates = [path for path in runs_dir.glob("*/fights.csv") if path.is_file()]
    if not candidates:
        return None
    newest = max(candidates, key=lambda path: path.stat().st_mtime)
    return newest.parent.name


def filter_run(fights: pd.DataFrame, turns: pd.DataFrame, run_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if fights.empty or "run_id" not in fights.columns:
        empty_turns = turns.iloc[0:0].copy() if not turns.empty else turns.copy()
        return fights.iloc[0:0].copy(), empty_turns
    chosen = fights[fights["run_id"] == run_id].copy()
    if turns.empty:
        return chosen, turns.copy()
    if "run_id" in turns.columns:
        kept = turns[turns["run_id"] == run_id].copy()
    elif "fight_id" in turns.columns and "fight_id" in chosen.columns:
        kept = turns[turns["fight_id"].isin(set(chosen["fight_id"]))].copy()
    else:
        kept = turns.iloc[0:0].copy()
    return chosen, kept


_BOOKKEEPING = {"skip", "end"}
_HOSTILE = {"attack", "spell", "feature"}
LIVING_ACTIONS = {"attack", "spell", "move", "feature", "dash", "disengage", "dodge"}


def living_actions(turns: pd.DataFrame) -> pd.DataFrame:
    """Actions a living creature took. End closes a turn. Skip is a creature at 0 hit points."""
    if turns is None:
        return pd.DataFrame()
    if turns.empty or "action_kind" not in turns.columns:
        return turns.iloc[0:0].copy()
    kept = turns["action_kind"].astype(str).isin(LIVING_ACTIONS)
    return turns.loc[kept].copy()
_MEASURES = (
    ("party_turns", "Party actions"),
    ("enemy_turns", "Enemy actions"),
    ("party_hostile", "Party attacks and spells"),
    ("enemy_hostile", "Enemy attacks and spells"),
)


def _other_side(side: str) -> str:
    if side == "party":
        return "enemy"
    if side == "enemy":
        return "party"
    return side


def _sentences(detail: str) -> list[str]:
    return [part.strip() for part in str(detail or "").split(".") if part.strip()]


def _dropped_names(detail: str) -> list[str]:
    marker = "drops to 0 hit points"
    names = []
    for sentence in _sentences(detail):
        if sentence.endswith(marker):
            name = sentence[: -len(marker)].strip()
            if name:
                names.append(name)
    return names


def _turn_table(turns: pd.DataFrame) -> pd.DataFrame:
    frame = turns.copy() if turns is not None and not turns.empty else pd.DataFrame()
    for column in ("fight_id", "round", "actor_name", "side", "action_kind", "detail"):
        if column not in frame.columns:
            frame[column] = ""
    frame["_ord"] = range(len(frame))
    return frame


def _blank_factor(fight_id: str) -> dict:
    return {
        "fight_id": fight_id,
        "first_side": None,
        "first_blood_round": None,
        "first_blood_victim": None,
        "first_blood_scorer": None,
        "party_turns": 0,
        "enemy_turns": 0,
        "party_hostile": 0,
        "enemy_hostile": 0,
        "max_swing": 0,
        "concentration_fails": 0,
        "concentration_holds": 0,
        "concentration_drops": 0,
    }


def _known_side(fight_id: str, name: str, by_fight: dict, any_fight: dict) -> str | None:
    side = by_fight.get(fight_id, {}).get(name) or any_fight.get(name)
    if side in {"party", "enemy"}:
        return side
    return None


def _score_drop(fight_id: str, victim: str, actor_name: str, actor_side: str, by_fight: dict, any_fight: dict) -> tuple[str, str] | None:
    if victim == actor_name and actor_side in {"party", "enemy"}:
        return actor_side, _other_side(actor_side)
    known = _known_side(fight_id, victim, by_fight, any_fight)
    if known:
        return known, _other_side(known)
    if actor_side in {"party", "enemy"}:
        return _other_side(actor_side), actor_side
    return None


def fight_factors(fights: pd.DataFrame, turns: pd.DataFrame) -> pd.DataFrame:
    """One row per fight: initiative, first knockout, actions, and concentration."""
    columns = list(_blank_factor("").keys())
    if fights.empty or "fight_id" not in fights.columns:
        return pd.DataFrame(columns=columns)
    table = _turn_table(turns)
    by_fight: dict[str, dict[str, str]] = {}
    any_fight: dict[str, str] = {}
    for row in table.itertuples(index=False):
        name = str(row.actor_name or "")
        side = str(row.side or "")
        fight_id = str(row.fight_id)
        if not name or side not in {"party", "enemy"}:
            continue
        by_fight.setdefault(fight_id, {}).setdefault(name, side)
        any_fight.setdefault(name, side)
    grouped = {
        str(fight_id): group.sort_values("_ord")
        for fight_id, group in table.groupby("fight_id", sort=False)
    }
    rows = []
    for fight_id in fights["fight_id"]:
        factor = _blank_factor(str(fight_id))
        group = grouped.get(str(fight_id))
        if group is None:
            rows.append(factor)
            continue
        for row in group.itertuples(index=False):
            kind = str(row.action_kind or "")
            side = str(row.side or "")
            detail = str(row.detail or "")
            if factor["first_side"] is None and kind != "skip" and side in {"party", "enemy"}:
                factor["first_side"] = side
            if kind not in _BOOKKEEPING and side == "party":
                factor["party_turns"] += 1
            elif kind not in _BOOKKEEPING and side == "enemy":
                factor["enemy_turns"] += 1
            if kind in _HOSTILE and side == "party":
                factor["party_hostile"] += 1
            elif kind in _HOSTILE and side == "enemy":
                factor["enemy_hostile"] += 1
            dropped = _dropped_names(detail)
            factor["max_swing"] = max(factor["max_swing"], len(dropped))
            if factor["first_blood_scorer"] is None and dropped:
                scored = _score_drop(
                    str(fight_id),
                    dropped[0],
                    str(row.actor_name or ""),
                    side,
                    by_fight,
                    any_fight,
                )
                if scored is not None:
                    factor["first_blood_victim"], factor["first_blood_scorer"] = scored
                    try:
                        factor["first_blood_round"] = int(row.round)
                    except (TypeError, ValueError):
                        factor["first_blood_round"] = None
            factor["concentration_fails"] += detail.count("fails the concentration save")
            factor["concentration_holds"] += detail.count("holds concentration")
            factor["concentration_drops"] += detail.count("loses concentration on ")
        rows.append(factor)
    return pd.DataFrame(rows)


def _with_win(fights: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    columns = ["fight_id"]
    for name in ("party_win", "winner"):
        if name in fights.columns:
            columns.append(name)
    if "fight_id" not in fights.columns or factors.empty:
        merged = factors.copy()
        merged["party_win"] = pd.NA
        merged["winner"] = "unfinished"
        return merged
    return factors.merge(fights[columns], on="fight_id", how="left")


def split_win_rate(fights: pd.DataFrame, factors: pd.DataFrame, column: str) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["group", "fights", "party_win"])
    if factors.empty or column not in factors.columns:
        return empty
    merged = _with_win(fights, factors)
    merged = merged[merged[column].notna() & (merged[column].astype(str) != "")]
    if merged.empty:
        return empty
    present = list(dict.fromkeys(merged[column].tolist()))
    keys = [key for key in ("party", "enemy") if key in present]
    keys.extend(key for key in present if key not in keys)
    rows = []
    for key in keys:
        group = merged[merged[column] == key]
        rows.append({"group": key, "fights": int(len(group)), "party_win": win_rate(group)})
    return pd.DataFrame(rows)


def action_economy_points(fights: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """One row per fight and measure, for a box plot rather than a mean bar."""
    empty = pd.DataFrame(columns=["winner", "measure", "count"])
    if factors.empty:
        return empty
    merged = _with_win(fights, factors)
    winner = merged["winner"] if "winner" in merged.columns else pd.Series(["unfinished"] * len(merged), index=merged.index)
    merged = merged.copy()
    merged["_winner"] = winner.fillna("unfinished")
    frames = []
    for column, label in _MEASURES:
        frames.append(pd.DataFrame({
            "winner": merged["_winner"].to_numpy(),
            "measure": label,
            "count": pd.to_numeric(merged[column], errors="coerce").to_numpy(),
        }))
    return pd.concat(frames, ignore_index=True)


def swing_summary(factors: pd.DataFrame) -> dict:
    if factors.empty or "max_swing" not in factors.columns:
        return {"fights": 0, "area_fights": 0, "largest": 0}
    swing = pd.to_numeric(factors["max_swing"], errors="coerce").fillna(0)
    return {
        "fights": int(len(factors)),
        "area_fights": int((swing >= 2).sum()),
        "largest": int(swing.max()),
    }


def concentration_summary(fights: pd.DataFrame, factors: pd.DataFrame) -> dict:
    if factors.empty:
        return {
            "fails": 0,
            "holds": 0,
            "drops": 0,
            "broke_fights": 0,
            "held_fights": 0,
            "broke_win_rate": None,
            "held_win_rate": None,
        }
    merged = _with_win(fights, factors)
    broke = (merged["concentration_fails"] > 0) | (merged["concentration_drops"] > 0)
    held = (merged["concentration_holds"] > 0) & ~broke
    return {
        "fails": int(merged["concentration_fails"].sum()),
        "holds": int(merged["concentration_holds"].sum()),
        "drops": int(merged["concentration_drops"].sum()),
        "broke_fights": int(broke.sum()),
        "held_fights": int(held.sum()),
        "broke_win_rate": win_rate(merged.loc[broke]),
        "held_win_rate": win_rate(merged.loc[held]),
    }


def pace_points(fights: pd.DataFrame) -> pd.DataFrame:
    """Damage per round for each fight and side. A fight with no rounds is left out."""
    empty = pd.DataFrame(columns=["side", "damage_per_round"])
    if fights.empty or "rounds" not in fights.columns:
        return empty
    rounds = pd.to_numeric(fights["rounds"], errors="coerce")
    valid = rounds > 0
    frames = []
    for column, side in (("damage_by_party", "Party"), ("damage_by_enemy", "Enemy")):
        if column not in fights.columns:
            continue
        damage = pd.to_numeric(fights[column], errors="coerce")
        frames.append(pd.DataFrame({
            "side": side,
            "damage_per_round": (damage[valid] / rounds[valid]).to_numpy(),
        }))
    if not frames:
        return empty
    return pd.concat(frames, ignore_index=True)


def damage_per_round(fights: pd.DataFrame, column: str) -> float | None:
    if fights.empty or column not in fights.columns or "rounds" not in fights.columns:
        return None
    rounds = pd.to_numeric(fights["rounds"], errors="coerce")
    damage = pd.to_numeric(fights[column], errors="coerce")
    valid = rounds > 0
    if not valid.any():
        return None
    return float((damage[valid] / rounds[valid]).mean())


def dropped_fraction(fights: pd.DataFrame, count_column: str, down_column: str) -> float | None:
    if fights.empty or count_column not in fights.columns or down_column not in fights.columns:
        return None
    count = pd.to_numeric(fights[count_column], errors="coerce")
    down = pd.to_numeric(fights[down_column], errors="coerce")
    valid = count > 0
    if not valid.any():
        return None
    return float((down[valid] / count[valid]).mean())


def headcount_sentence(fights: pd.DataFrame) -> str | None:
    if fights.empty or not {"enemy_count", "party_size"}.issubset(fights.columns):
        return None
    party = pd.to_numeric(fights["party_size"], errors="coerce")
    enemy = pd.to_numeric(fights["enemy_count"], errors="coerce")
    valid = party.notna() & enemy.notna() & (party > 0)
    if not valid.any():
        return None
    if party[valid].nunique() == 1 and enemy[valid].nunique() == 1:
        members = int(party[valid].iloc[0])
        foes = int(enemy[valid].iloc[0])
        foe_word = "enemy" if foes == 1 else "enemies"
        member_word = "party member" if members == 1 else "party members"
        return (
            f"The layout opens with {foes} {foe_word} and {members} {member_word} "
            f"({foes / members:.1f} enemies per character)."
        )
    ratio = float((enemy[valid] / party[valid]).mean())
    return f"Across these layouts the mean opening ratio is {ratio:.1f} enemies per character."


def layout_summary(fights: pd.DataFrame) -> pd.DataFrame:
    if fights.empty or "label" not in fights.columns:
        return pd.DataFrame()
    rows = []
    for label, group in fights.groupby("label", sort=True):
        rows.append({
            "label": label,
            "fights": len(group),
            "party_win": win_rate(group),
            "mean_rounds": float(pd.to_numeric(group["rounds"], errors="coerce").mean()) if "rounds" in group else None,
            "party_size": float(pd.to_numeric(group["party_size"], errors="coerce").mean()) if "party_size" in group else None,
            "enemy_count": float(pd.to_numeric(group["enemy_count"], errors="coerce").mean()) if "enemy_count" in group else None,
            "mean_enemy_cr": float(pd.to_numeric(group["mean_enemy_cr"], errors="coerce").mean()) if "mean_enemy_cr" in group else None,
        })
    return pd.DataFrame(rows)


def _rate_phrase(frame: pd.DataFrame, labels: dict[str, str]) -> str:
    if frame.empty:
        return "none"
    parts = []
    for row in frame.itertuples(index=False):
        rate = "undefined" if row.party_win is None or pd.isna(row.party_win) else f"{float(row.party_win):.2f}"
        count = int(row.fights)
        noun = "fight" if count == 1 else "fights"
        parts.append(f"{labels.get(row.group, row.group)} in {count} {noun}, party win rate {rate}")
    return "; ".join(parts)


def _factor_lines(fights: pd.DataFrame, factors: pd.DataFrame) -> list[str]:
    lines = [
        "First turn: " + _rate_phrase(split_win_rate(fights, factors, "first_side"), {
            "party": "party went first",
            "enemy": "enemy went first",
        }),
        "First knockout: " + _rate_phrase(split_win_rate(fights, factors, "first_blood_scorer"), {
            "party": "party scored it",
            "enemy": "enemy scored it",
        }),
    ]
    if factors.empty:
        return lines
    lines.append(
        "Mean actions, excluding End and skipped turns: "
        f"party {factors['party_turns'].mean():.2f}, enemy {factors['enemy_turns'].mean():.2f}. "
        "Mean attacks and spells: "
        f"party {factors['party_hostile'].mean():.2f}, enemy {factors['enemy_hostile'].mean():.2f}."
    )
    swing = swing_summary(factors)
    lines.append(
        f"One action dropped two or more creatures in {swing['area_fights']} of {swing['fights']} fights. "
        f"Largest swing: {swing['largest']}."
    )
    conc = concentration_summary(fights, factors)
    lines.append(
        f"Concentration saves failed: {conc['fails']}. Held: {conc['holds']}. Lost on a knockout: {conc['drops']}."
    )
    if conc["fails"] or conc["holds"] or conc["drops"]:
        lines.append(
            f"Fights that broke concentration: {conc['broke_fights']}, party win rate {conc['broke_win_rate']}. "
            f"Fights that held it without a break: {conc['held_fights']}, party win rate {conc['held_win_rate']}."
        )
    else:
        lines.append("No concentration save or lost concentration is in these fights.")
    return lines


def outlier_mask(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    q1 = numeric.quantile(0.25)
    q3 = numeric.quantile(0.75)
    spread = q3 - q1
    if pd.isna(spread) or spread == 0:
        return pd.Series(False, index=series.index)
    return (numeric < q1 - 1.5 * spread) | (numeric > q3 + 1.5 * spread)


def outlier_fights(fights: pd.DataFrame) -> pd.DataFrame:
    if fights.empty or "rounds" not in fights.columns:
        return fights.iloc[0:0]
    mask = outlier_mask(fights["rounds"])
    if "damage_by_party" in fights.columns:
        mask = mask | outlier_mask(fights["damage_by_party"])
    columns = [column for column in ("fight_id", "label", "policy", "winner", "rounds", "damage_by_party", "damage_by_enemy", "party_win") if column in fights.columns]
    return fights.loc[mask, columns]


_ATTACK_D20 = re.compile(
    r"Attack\s+(?:advantage |disadvantage )?(\d+(?: and \d+)*)(?:, keeps \d+)? \+"
)
_ROLL_D20 = re.compile(r"Roll\s+(?:(\d+(?: and \d+)*), keeps \d+,|(\d+)) \+")
_SAVE_NAME = re.compile(r"(?:Failed save|Saved)\s+(.+?)\s+vs DC")
_NAMED_D20 = re.compile(
    r"([^.(\n]{1,80}?) (?:holds concentration|fails the concentration save"
    r"|resists Topple|is knocked prone|resists Telekinetic Shove"
    r"|succeeds on the Constitution save|fails the Constitution save"
    r"|is shoved to \([^)]*\)) \((?:(?:advantage|disadvantage) )?(\d+(?: and \d+)*)(?:, keeps \d+)?,?\s*\+"
)
_OLD_SAVE_D20 = re.compile(
    r"([A-Z][^.]{0,80}?) (?:fails|succeeds) a DC \d+ [A-Za-z]+ save against .*?\((\d+)\+(-?\d+)="
)
_DICE_FACES = re.compile(r"(\d*d\d+(?:[+-]\d+)?)(?: critical)?: dice (\d+(?: \+ \d+)*)")
_FURY_FACES = re.compile(r"Divine Fury(?: critical)?: (\d*d\d+) dice (\d+(?: \+ \d+)*)")
_ARROW_DICE = re.compile(r"Dice (\d*d\d+) -> (\d+)")
_SAVAGE = re.compile(r"Savage Attacker keeps (\d+) \(other roll (\d+)\)")
_WAYS: dict[tuple[int, int], dict[int, int]] = {}


def _ways(count: int, sides: int) -> dict[int, int]:
    key = (count, sides)
    cached = _WAYS.get(key)
    if cached is not None:
        return cached
    ways = {0: 1}
    for _ in range(count):
        nxt: dict[int, int] = {}
        for total, copies in ways.items():
            for face in range(1, sides + 1):
                nxt[total + face] = nxt.get(total + face, 0) + copies
        ways = nxt
    _WAYS[key] = ways
    return ways


def _percentile(count: int, sides: int, total: int) -> float | None:
    """Mid-rank percentile of a dice sum. A fair roll of any size averages 50."""
    if count < 1 or sides < 2 or total < count or total > count * sides:
        return None
    ways = _ways(count, sides)
    universe = sum(ways.values())
    equal = ways.get(total, 0)
    if not equal or not universe:
        return None
    below = sum(copies for value, copies in ways.items() if value < total)
    return (below + 0.5 * equal) / universe * 100


def _die_sides(expression: str) -> int | None:
    try:
        _count, sides, _modifier = parse_dice(expression)
    except ValueError:
        return None
    if sides < 2:
        return None
    return sides


def _faces(text: str) -> list[int]:
    return [int(part) for part in text.split(" and ") if part.isdigit()]


def _sentence_list(detail: str) -> list[str]:
    sentences = []
    for line in str(detail or "").splitlines():
        sentences.extend(part.strip() for part in re.split(r"(?<=\.)\s+", line.strip()) if part.strip())
    return sentences


def _damage_roll(expression: str, total: int, count: int | None = None) -> float | None:
    sides = _die_sides(expression)
    if sides is None:
        return None
    if count is None:
        try:
            count, parsed_sides, _modifier = parse_dice(expression)
        except ValueError:
            return None
        sides = parsed_sides
    return _percentile(int(count), sides, int(total))


def _read_luck(detail: str, actor_side: str | None, name_side: dict[str, str | None]) -> tuple[list[tuple[str | None, int]], list[tuple[str | None, float]]]:
    """d20 faces and damage percentiles in one detail. Side is the roller, not always the actor."""
    d20s: list[tuple[str | None, int]] = []
    damage: list[tuple[str | None, float]] = []
    remembered: list[tuple[int, int, int]] = []
    saver: str | None = None

    def known(name: str | None) -> str | None:
        cleaned = " ".join(str(name or "").split())
        if not cleaned:
            return None
        return name_side.get(cleaned)

    for sentence in _sentence_list(detail):
        if "regains" in sentence:
            continue
        named = _SAVE_NAME.search(sentence)
        if named:
            saver = " ".join(named.group(1).split())
        for match in _ATTACK_D20.finditer(sentence):
            for face in _faces(match.group(1)):
                if 1 <= face <= 20:
                    d20s.append((actor_side, face))
        for match in _ROLL_D20.finditer(sentence):
            text = match.group(1) or match.group(2) or ""
            for face in _faces(text):
                if 1 <= face <= 20:
                    d20s.append((known(saver), face))
            saver = None
        for pattern in (_NAMED_D20, _OLD_SAVE_D20):
            for match in pattern.finditer(sentence):
                text = match.group(2)
                faces = _faces(text) if "and" in text else [int(text)]
                for face in faces:
                    if 1 <= face <= 20:
                        d20s.append((known(match.group(1)), face))
        for pattern in (_DICE_FACES, _FURY_FACES):
            for match in pattern.finditer(sentence):
                expression = match.group(1)
                faces = [int(part) for part in match.group(2).split(" + ")]
                sides = _die_sides(expression)
                if sides is None or any(face < 1 or face > sides for face in faces):
                    continue
                score = _percentile(len(faces), sides, sum(faces))
                if score is None:
                    continue
                damage.append((actor_side, score))
                remembered.append((len(faces), sides, sum(faces)))
        for match in _ARROW_DICE.finditer(sentence):
            expression = match.group(1)
            if "+" in expression or "-" in expression:
                continue
            score = _damage_roll(expression, int(match.group(2)))
            if score is None:
                continue
            damage.append((actor_side, score))
        for match in _SAVAGE.finditer(sentence):
            kept = int(match.group(1))
            other = int(match.group(2))
            found = next((item for item in remembered if item[2] == kept), None)
            if found is None:
                continue
            score = _percentile(found[0], found[1], other)
            if score is not None:
                damage.append((actor_side, score))
    return d20s, damage


def _name_sides(turns: pd.DataFrame) -> dict[str, str | None]:
    found: dict[str, set[str]] = {}
    if turns.empty or "actor_name" not in turns.columns or "side" not in turns.columns:
        return {}
    for name, side in zip(turns["actor_name"], turns["side"], strict=False):
        cleaned = " ".join(str(name or "").split())
        label = str(side or "")
        if not cleaned or label not in {"party", "enemy"}:
            continue
        found.setdefault(cleaned, set()).add(label)
    return {name: next(iter(sides)) if len(sides) == 1 else None for name, sides in found.items()}


def _winner_of(fights: pd.DataFrame) -> dict[str, str]:
    if fights is None or fights.empty or "fight_id" not in fights.columns or "winner" not in fights.columns:
        return {}
    winners = {}
    for fight_id, winner in zip(fights["fight_id"], fights["winner"], strict=False):
        if pd.isna(winner):
            continue
        label = str(winner)
        if label in {"party", "enemy", "draw"}:
            winners[str(fight_id)] = label
    return winners


def luck_summary(turns: pd.DataFrame, fights: pd.DataFrame | None = None) -> dict:
    """Mean natural d20 and mean damage percentile, overall and by side and winner."""
    groups = pd.DataFrame(columns=["side", "winner", "d20_mean", "d20_count", "damage_percentile", "damage_count"])
    empty = {"d20_mean": None, "d20_count": 0, "damage_percentile": None, "damage_count": 0, "groups": groups}
    if turns is None or turns.empty or "detail" not in turns.columns:
        return empty
    names = _name_sides(turns)
    winners = _winner_of(fights if fights is not None else pd.DataFrame())
    d20_values: list[int] = []
    damage_values: list[float] = []
    records: list[dict] = []
    for row in turns.itertuples(index=False):
        detail = getattr(row, "detail", "")
        actor_side = str(getattr(row, "side", "") or "")
        if actor_side not in {"party", "enemy"}:
            actor_side = None
        fight_id = str(getattr(row, "fight_id", "") or "")
        winner = winners.get(fight_id)
        faces, rolls = _read_luck("" if pd.isna(detail) else str(detail), actor_side, names)
        for side, face in faces:
            d20_values.append(face)
            if side in {"party", "enemy"} and winner:
                records.append({"side": side, "winner": winner, "kind": "d20", "value": face})
        for side, score in rolls:
            damage_values.append(score)
            if side in {"party", "enemy"} and winner:
                records.append({"side": side, "winner": winner, "kind": "damage", "value": score})
    if records:
        frame = pd.DataFrame(records)
        rows = []
        order = {"party": 0, "enemy": 1}
        outcome = {"party": 0, "enemy": 1, "draw": 2}
        keys = sorted(frame.groupby(["side", "winner"], sort=False).groups, key=lambda item: (order.get(item[0], 9), outcome.get(item[1], 9)))
        for side, winner in keys:
            part = frame[(frame["side"] == side) & (frame["winner"] == winner)]
            d20 = part.loc[part["kind"] == "d20", "value"]
            rolled = part.loc[part["kind"] == "damage", "value"]
            rows.append({
                "side": "Party" if side == "party" else "Enemy",
                "winner": winner,
                "d20_mean": None if d20.empty else float(d20.mean()),
                "d20_count": int(len(d20)),
                "damage_percentile": None if rolled.empty else float(rolled.mean()),
                "damage_count": int(len(rolled)),
            })
        groups = pd.DataFrame(rows)
    return {
        "d20_mean": None if not d20_values else float(sum(d20_values) / len(d20_values)),
        "d20_count": len(d20_values),
        "damage_percentile": None if not damage_values else float(sum(damage_values) / len(damage_values)),
        "damage_count": len(damage_values),
        "groups": groups,
    }


def _luck_lines(summary: dict) -> list[str]:
    lines = []
    if summary["d20_count"]:
        lines.append(
            f"Mean d20: {summary['d20_mean']:.2f} over {summary['d20_count']} faces (fair is 10.5)."
        )
    if summary["damage_count"]:
        lines.append(
            "Mean damage-roll percentile: "
            f"{summary['damage_percentile']:.1f} over {summary['damage_count']} rolls (fair is 50)."
        )
    groups = summary["groups"]
    if groups is None or groups.empty:
        return lines
    for row in groups.itertuples(index=False):
        d20 = "no d20 faces" if not row.d20_count else f"mean d20 {row.d20_mean:.2f} over {int(row.d20_count)} faces"
        if row.damage_count:
            damage = f"damage percentile {row.damage_percentile:.1f} over {int(row.damage_count)} rolls"
        else:
            damage = "no damage rolls"
        lines.append(f"Luck for {row.side} in fights won by {row.winner}: {d20}, {damage}.")
    return lines


def report_brief(fights: pd.DataFrame, coverage: dict | None = None, turns: pd.DataFrame | None = None) -> str:
    """Plain-text brief the report model is allowed to cite."""
    if fights.empty:
        return "No fights have been saved."
    lines = [
        f"Saved fights: {len(fights)}",
        f"Party win rate (draws omitted): {win_rate(fights)}",
    ]
    if "rounds" in fights.columns:
        rounds = pd.to_numeric(fights["rounds"], errors="coerce")
        lines.append(f"Mean rounds: {rounds.mean():.2f}")
        lines.append(f"Median rounds: {rounds.median():.2f}")
    party_pace = damage_per_round(fights, "damage_by_party")
    enemy_pace = damage_per_round(fights, "damage_by_enemy")
    if party_pace is not None:
        lines.append(f"Mean party damage per round: {party_pace:.1f}")
    if enemy_pace is not None:
        lines.append(f"Mean enemy damage per round: {enemy_pace:.1f}")
    party_lost = dropped_fraction(fights, "party_size", "party_down")
    enemy_lost = dropped_fraction(fights, "enemy_count", "enemy_down")
    if party_lost is not None:
        lines.append(f"Mean fraction of the party dropped: {party_lost:.2f}")
    if enemy_lost is not None:
        lines.append(f"Mean fraction of the enemies dropped: {enemy_lost:.2f}")
    opening = headcount_sentence(fights)
    if opening:
        lines.append(opening)
    if "policy" in fights.columns:
        for policy, group in fights.groupby("policy"):
            lines.append(f"Policy {policy}: {len(group)} fights, party win rate {win_rate(group)}")
    if turns is not None:
        lines.extend(_factor_lines(fights, fight_factors(fights, turns)))
        lines.extend(_luck_lines(luck_summary(turns, fights)))
    outliers = outlier_fights(fights)
    lines.append(f"Outlier fights by the 1.5 IQR rule on rounds or party damage: {len(outliers)}")
    if not outliers.empty:
        lines.append(outliers.head(8).to_csv(index=False).strip())
    if coverage:
        lines.append("API coverage: " + json.dumps(coverage))
    return "\n".join(lines)
