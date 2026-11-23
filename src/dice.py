"""Dice expressions from the SRD API, such as ``2d6+1`` or ``8d6``."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

_SPACES = re.compile(r"\s+")
_DICE = re.compile(r"^(?:(\d*)d(\d+))?([+-]\d+)?$")


@dataclass(frozen=True)
class RollResult:
    expression: str
    rolls: tuple[int, ...]
    modifier: int
    total: int
    critical: bool


def normalize_expression(expression: str) -> str:
    text = _SPACES.sub("", expression.strip().lower())
    text = text.replace("+-", "-").replace("-+", "-")
    return text


def parse_dice(expression: str) -> tuple[int, int, int]:
    """Return ``(count, sides, modifier)``. A flat number has ``sides == 0``."""
    text = normalize_expression(expression)
    if re.fullmatch(r"[+-]?\d+", text):
        return 0, 0, int(text)
    match = _DICE.fullmatch(text)
    if not match or match.group(2) is None:
        raise ValueError(f"Cannot parse dice expression: {expression}")
    count = int(match.group(1) or 1)
    sides = int(match.group(2))
    modifier = int(match.group(3) or 0)
    if count < 0 or count > 40 or sides < 2 or sides > 100:
        raise ValueError(f"Dice expression is out of range: {expression}")
    return count, sides, modifier


def average(expression: str) -> float:
    count, sides, modifier = parse_dice(expression)
    if sides == 0:
        return float(modifier)
    return count * (sides + 1) / 2 + modifier


def critical_average(expression: str) -> float:
    """Crits double the dice and leave the modifier alone."""
    count, sides, modifier = parse_dice(expression)
    if sides == 0:
        return float(modifier)
    return (count * 2) * (sides + 1) / 2 + modifier


def roll_dice(
    expression: str,
    rng: random.Random,
    *,
    critical: bool = False,
) -> RollResult:
    count, sides, modifier = parse_dice(expression)
    times = count * (2 if critical and sides else 1)
    rolls = tuple(rng.randint(1, sides) for _ in range(times)) if sides else ()
    total = sum(rolls) + modifier
    return RollResult(
        expression=normalize_expression(expression),
        rolls=rolls,
        modifier=modifier,
        total=total,
        critical=critical and bool(sides),
    )


def substitute_mod(expression: str, modifier: int) -> str:
    """Replace the API token ``MOD`` with a spellcasting modifier."""
    text = _SPACES.sub("", expression.strip())
    text = text.replace("MOD", str(modifier))
    return normalize_expression(text)
