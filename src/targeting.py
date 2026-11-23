"""Group the flat action list into abilities a player can aim on the map.

The engine still builds one action per target, square, facing, and slot.
This module only picks among those actions. It does not add targets.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.actions import GameAction, _zone_aim_cells, targets_for
from src.grid import area_squares

_SLOT = re.compile(r"^(?P<name>.+?) (?:cantrip|L\d+)(?: |$)")


@dataclass(frozen=True)
class Ability:
    key: str
    label: str
    budget: str
    kind: str
    attack_name: str | None
    spell_index: str | None
    slots: tuple[int, ...]
    needs: str


def ability_key(action: GameAction) -> str:
    return "|".join((action.budget, action.kind, action.attack_name or "", action.spell_index or ""))


def abilities_from(actions: list[GameAction]) -> list[Ability]:
    """One entry per attack or spell. Moves stay on the map, not on the bar."""
    grouped: dict[str, list[GameAction]] = {}
    order: list[str] = []
    for action in actions:
        if action.kind == "move":
            continue
        key = ability_key(action)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(action)
    return [_ability(key, grouped[key]) for key in order]


def chosen_slot(ability: Ability, slot: int | None) -> int | None:
    if not ability.slots:
        return None
    if slot in ability.slots:
        return slot
    return ability.slots[0]


def movement_squares(actions: list[GameAction]) -> set[tuple[int, int]]:
    return {action.destination for action in actions if action.kind == "move" and action.destination is not None}


def actions_for(actions: list[GameAction], ability: Ability, slot: int | None) -> list[GameAction]:
    picked = chosen_slot(ability, slot)
    rows = [action for action in actions if ability_key(action) == ability.key]
    if picked is None:
        return rows
    return [action for action in rows if action.slot_level == picked]


def match_instant(actions: list[GameAction], ability: Ability, slot: int | None) -> GameAction | None:
    rows = actions_for(actions, ability, slot)
    if len(rows) == 1:
        return rows[0]
    return None


def match_creature(actions: list[GameAction], ability: Ability, slot: int | None, target_id: str) -> GameAction | None:
    for action in actions_for(actions, ability, slot):
        if action.target_id == target_id:
            return action
    return None


def match_square(actions: list[GameAction], ability: Ability, slot: int | None, square: tuple[int, int]) -> GameAction | None:
    square = (int(square[0]), int(square[1]))
    for action in actions_for(actions, ability, slot):
        if action.aim == square:
            return action
    return None


def facing_for_square(origin: tuple[int, int], square: tuple[int, int]) -> tuple[int, int] | None:
    """Snap a clicked square to one of the eight compass facings."""
    dx = square[0] - origin[0]
    dy = square[1] - origin[1]
    if dx == 0 and dy == 0:
        return None
    step_x = 0 if dx == 0 else (1 if dx > 0 else -1)
    step_y = 0 if dy == 0 else (1 if dy > 0 else -1)
    return step_x, step_y


def match_facing(
    actions: list[GameAction],
    ability: Ability,
    slot: int | None,
    origin: tuple[int, int],
    square: tuple[int, int],
) -> GameAction | None:
    facing = facing_for_square(origin, square)
    if facing is None:
        return None
    for action in actions_for(actions, ability, slot):
        if action.facing == facing:
            return action
    return None


def _weapon_needs(option) -> str:
    if (option.rider or {}).get("aimed"):
        return "square"
    if option.area_type:
        return "facing"
    return "creature"


def offer_weapons(encounter, actor, abilities: list[Ability]) -> list[Ability]:
    """List every weapon under Attack, including ones that currently reach nobody."""
    if actor is None or getattr(actor, "hp", 0) <= 0:
        return abilities
    from src.actions import _available, attack_window, nick_weapon_name

    present = {ability.key for ability in abilities}
    added = list(abilities)
    window = attack_window(actor)

    def inject(name: str, budget: str, needs: str) -> None:
        key = f"{budget}|attack|{name}|"
        if key in present:
            return
        present.add(key)
        added.append(
            Ability(
                key=key,
                label=name,
                budget=budget,
                kind="attack",
                attack_name=name,
                spell_index=None,
                slots=(),
                needs=needs,
            )
        )

    if window == "open" and actor.multiattack:
        inject("Multiattack", "action", "creature")
    if window == "nick":
        nick = nick_weapon_name(actor)
        if nick:
            inject(nick, "action", "creature")
    elif window in {"open", "extra"}:
        for option in actor.attacks:
            if option.activation != "action" or option.resolution == "disengage" or not _available(actor, option):
                continue
            if window == "extra" and (option.resolution != "attack" or option.area_type):
                continue
            needs = _weapon_needs(option)
            inject(option.name, "action", needs)
    if actor.bonus_available:
        for option in actor.attacks:
            if option.activation == "bonus" and option.resolution == "attack" and _available(actor, option):
                needs = _weapon_needs(option)
                inject(option.name, "bonus", needs)
    return added


def preview_squares(encounter, actor, action: GameAction) -> set[tuple[int, int]]:
    """Cells the template covers. Empty when the action has no area."""
    if action.kind == "feature" and action.attack_name == "Land's Aid" and action.aim is not None:
        return set(area_squares(tuple(action.aim), "sphere", 10, None))
    if action.kind == "spell":
        spell = actor.spell_named(action.spell_index)
        if spell is not None and spell.kind == "zone":
            return set(_zone_aim_cells(encounter, actor, spell, action))
        if spell is None or not spell.area_type:
            return set()
        if spell.range_mode == "self":
            origin = tuple(actor.position)
        else:
            origin = tuple(action.aim or actor.position)
        return set(area_squares(origin, spell.area_type, spell.area_size or 0, action.facing))
    if action.kind == "attack" and action.attack_name and action.aim is not None:
        option = actor.attack_named(action.attack_name)
        if option is not None and option.area_type and (option.rider or {}).get("aimed"):
            return set(area_squares(tuple(action.aim), option.area_type, option.area_size or 0, None))
    if action.kind == "attack" and action.attack_name and action.facing:
        option = actor.attack_named(action.attack_name)
        if option is None or not option.area_type:
            return set()
        return set(area_squares(tuple(actor.position), option.area_type, option.area_size or 0, action.facing))
    return set()


def board_marks(encounter, actor, actions, ability: Ability | None, slot: int | None, pending: tuple[int, int] | None):
    """Movement, empty aim squares, outlined squares, and the amber template."""
    if ability is None:
        return movement_squares(actions), set(), set(), set()
    rows = actions_for(actions, ability, slot)
    pending_action = _pending(actions, ability, slot, tuple(actor.position), pending)
    preview = preview_squares(encounter, actor, pending_action) if pending_action is not None else set()
    if ability.needs == "creature":
        targets = set()
        for action in rows:
            if not action.target_id:
                continue
            target = encounter.by_id(action.target_id)
            if target is not None and target.hp > 0:
                targets.update(target.squares())
        return set(), set(), targets, preview
    if ability.needs == "square":
        squares = {action.aim for action in rows if action.aim is not None}
        aim, targets = _split_occupied(encounter, squares)
        return set(), aim, targets, preview
    if ability.needs == "facing":
        squares: set[tuple[int, int]] = set()
        for action in rows:
            squares |= preview_squares(encounter, actor, action)
        aim, targets = _split_occupied(encounter, squares)
        return set(), aim, targets, preview
    if ability.needs == "confirm":
        action = rows[0] if len(rows) == 1 else None
        shown = preview_squares(encounter, actor, action) if action is not None else set()
        return set(), set(), set(), shown
    return set(), set(), set(), set()


def affected_names(encounter, actor, action: GameAction) -> str:
    preview = targets_for(encounter, actor, action)
    if not preview:
        return "The template does not reach anyone."
    who = ", ".join(f"{target.name} ({target.hp} HP)" for target in preview)
    return f"Reaches {who}."


def _pending(actions, ability: Ability, slot, origin, pending: tuple[int, int] | None) -> GameAction | None:
    if ability.needs == "square" and pending is not None:
        return match_square(actions, ability, slot, pending)
    if ability.needs == "facing" and pending is not None:
        return match_facing(actions, ability, slot, origin, pending)
    return None


def _split_occupied(encounter, squares: set[tuple[int, int]]):
    occupied: set[tuple[int, int]] = set()
    for creature in encounter.combatants:
        if creature.hp > 0:
            occupied.update(creature.squares())
    return squares - occupied, squares & occupied


def _ability(key: str, rows: list[GameAction]) -> Ability:
    sample = rows[0]
    slots = tuple(sorted({row.slot_level for row in rows if row.slot_level is not None}))
    return Ability(
        key=key,
        label=_label(sample),
        budget=sample.budget,
        kind=sample.kind,
        attack_name=sample.attack_name,
        spell_index=sample.spell_index,
        slots=slots,
        needs=_needs(rows),
    )


def _label(action: GameAction) -> str:
    if action.kind == "feature" and action.attack_name == "Spiritual Weapon":
        return "Move Spiritual Weapon"
    if action.kind == "feature" and action.attack_name:
        return action.attack_name
    if action.kind == "spell":
        match = _SLOT.match(action.label)
        if match:
            return match.group("name")
    if action.attack_name and action.kind in {"attack", "disengage"}:
        return action.attack_name
    return action.label


def _needs(rows: list[GameAction]) -> str:
    sample = rows[0]
    if sample.kind in {"dash", "dodge", "end", "disengage"}:
        return "none"
    facings = {row.facing for row in rows if row.facing is not None}
    if len(facings) > 1:
        return "facing"
    if any(row.target_id for row in rows):
        return "creature"
    aims = {row.aim for row in rows if row.aim is not None}
    if len(aims) > 1:
        return "square"
    if len(facings) == 1 or len(aims) == 1:
        return "confirm"
    return "none"
