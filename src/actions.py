"""Legal actions for one creature. Automated lists stay small enough to show a model."""

from __future__ import annotations

from dataclasses import dataclass

from src.bestiary import spell_damage_parts, spell_heal_expression
from src.conditions import (
    affliction,
    base_speed,
    can_see,
    can_stand,
    charmed_by,
    closer_to_source,
    frightened_sources,
    is_incapacitated,
    pending_order,
    size_rank,
    speed_locked,
    stand_cost,
    step_cost,
)
from src.grid import FACINGS, area_squares, chebyshev, footprint, reachable_anchors, space_gap, step_path
from src.zones import (
    anchor_cost,
    blocking_cells,
    bolt_cells,
    cells_of,
    centered_block,
    compass,
    difficult_costs,
    footprint_free,
    foe_points,
    side_band,
    travel_cost,
    zone_spec,
)

@dataclass(frozen=True)
class GameAction:
    id: int
    kind: str
    label: str
    attack_name: str | None = None
    spell_index: str | None = None
    target_id: str | None = None
    destination: tuple[int, int] | None = None
    aim: tuple[int, int] | None = None
    facing: tuple[int, int] | None = None
    slot_level: int | None = None
    budget: str = "action"
    tag: str | None = None


# These two emanations follow the caster and save on entry, not at the start of the caster's turn.
EMANATION_INDEXES = {"spirit-guardians", "conjure-woodland-beings"}


def same_action(left: GameAction, right: GameAction) -> bool:
    return (
        left.kind == right.kind
        and left.attack_name == right.attack_name
        and left.spell_index == right.spell_index
        and left.target_id == right.target_id
        and left.destination == right.destination
        and left.aim == right.aim
        and left.facing == right.facing
        and left.slot_level == right.slot_level
        and left.budget == right.budget
        and left.tag == right.tag
    )


def facing_name(facing: tuple[int, int] | None) -> str:
    for name, value in FACINGS.items():
        if value == facing:
            return name
    return str(facing)


def _living(encounter, *, side: str | None = None) -> list:
    rows = [creature for creature in encounter.combatants if creature.hp > 0]
    if side is None:
        return rows
    return [creature for creature in rows if creature.side == side]


def _foes(encounter, actor) -> list:
    return [creature for creature in _living(encounter) if creature.side != actor.side]


def adjacent_foe(encounter, actor, square):
    """Nearest living enemy within 5 feet of a square. Ties use the lower id."""
    origin = (int(square[0]), int(square[1]))
    best = None
    best_key = None
    for foe in _foes(encounter, actor):
        gap = space_gap(origin, 1, tuple(foe.position), foe.span)
        if gap != 1:
            continue
        key = (gap, foe.id)
        if best_key is None or key < best_key:
            best = foe
            best_key = key
    return best


def _occupied(encounter) -> set[tuple[int, int]]:
    taken: set[tuple[int, int]] = set()
    for creature in encounter.combatants:
        if creature.hp > 0:
            taken.update(creature.squares())
    return taken


def _allies(encounter, actor, *, include_self: bool) -> list:
    rows = []
    for creature in encounter.combatants:
        if creature.side != actor.side:
            continue
        if creature.id == actor.id and not include_self:
            continue
        rows.append(creature)
    return rows


def turn_speed(actor) -> int:
    if speed_locked(actor):
        return 0
    return base_speed(actor)


def movement_budget(actor) -> int:
    """Feet still available. None means the turn has not spent any yet."""
    if actor.movement_left is None:
        return turn_speed(actor)
    return int(actor.movement_left)


def _available(actor, option) -> bool:
    if option.name in actor.recharging:
        return False
    if option.uses is not None and actor.uses_left.get(option.name, 0) <= 0:
        return False
    return True


def weapon_reaches(actor, target, option) -> bool:
    distance = actor.gap_to(target)
    if distance == 0:
        return False
    feet = distance * 5
    if option.reach_feet is not None and feet <= option.reach_feet:
        return True
    return option.range_feet is not None and feet <= option.range_feet


def using_melee(actor, target, option) -> bool:
    if option.reach_feet is None:
        return False
    distance = actor.gap_to(target)
    return distance > 0 and distance * 5 <= option.reach_feet


def _tactical(actor, foes, squares: set[tuple[int, int]], limit: int = 8) -> list[tuple[int, int]]:
    if not foes or not squares:
        return []

    def sort_key(square: tuple[int, int]) -> tuple[int, int]:
        nearest = min(space_between(square, actor, foe) for foe in foes)
        return nearest, chebyshev(square, tuple(actor.position))

    ordered = sorted(squares, key=sort_key)
    adjacent = [square for square in ordered if any(space_between(square, actor, foe) == 1 for foe in foes)]
    picks = adjacent[:limit]
    if ordered[0] not in picks:
        picks.append(ordered[0])
    return picks


def space_between(anchor: tuple[int, int], actor, other) -> int:
    return space_gap(anchor, actor.span, tuple(other.position), other.span)


def _reachable(
    encounter,
    actor,
    speed_feet: int,
    origin: tuple[int, int] | None = None,
) -> set[tuple[int, int]]:
    start = tuple(actor.position if origin is None else origin)
    blocked: set[tuple[int, int]] = set()
    for foe in _foes(encounter, actor):
        blocked.update(foe.squares())
    passable: set[tuple[int, int]] = set()
    for creature in encounter.combatants:
        if creature.id != actor.id:
            passable.update(creature.squares())
    blocked.update(blocking_cells(encounter))
    step = step_cost(encounter, actor)
    return reachable_anchors(
        start,
        actor.span,
        speed_feet,
        width=encounter.width,
        height=encounter.height,
        blocked=blocked,
        passable=passable,
        step_cost=step,
    )


@dataclass(frozen=True)
class EmanationZone:
    """One hostile caster's emanation. ``saved`` means this creature already saved against it this turn."""

    caster: object
    spell: object
    cells: frozenset
    saved: bool


def _emanation_marked(caster, mark: list[int], creature_id: str) -> bool:
    saved = caster.traits.get("emanation_saved")
    if not isinstance(saved, dict) or list(saved.get("turn") or []) != mark:
        return False
    return str(creature_id) in {str(item) for item in saved.get("ids") or []}


def hostile_zones(encounter, actor) -> list[EmanationZone]:
    """Living enemy casters whose emanation is still up. An ally's area is not included."""
    mark = [int(encounter.round), int(encounter.turn_index)]
    zones = []
    for caster in encounter.combatants:
        if caster.hp <= 0 or caster.side == actor.side or caster.id == actor.id:
            continue
        index = caster.concentrating_on
        if index not in EMANATION_INDEXES:
            continue
        spell = caster.spell_named(index)
        if spell is None or not spell.ongoing or not spell.area_type or not spell.area_size:
            continue
        cells = area_squares(
            (int(caster.position[0]), int(caster.position[1])),
            spell.area_type,
            int(spell.area_size),
            None,
        )
        zones.append(EmanationZone(
            caster=caster,
            spell=spell,
            cells=frozenset(cells),
            saved=_emanation_marked(caster, mark, actor.id),
        ))
    return zones


def path_enters(actor, origin: tuple[int, int], dest: tuple[int, int], cells) -> bool:
    """True when the straight step path newly overlaps ``cells``. Standing inside does not count."""
    previous = (int(origin[0]), int(origin[1]))
    occupied = set(cells)
    previous_hit = bool(footprint(previous, actor.span) & occupied)
    for square in step_path(previous, dest):
        hit = bool(footprint(square, actor.span) & occupied)
        if hit and not previous_hit:
            return True
        previous = square
        previous_hit = hit
    return False


def _body_in(actor, square: tuple[int, int], cells) -> bool:
    return bool(footprint(square, actor.span) & set(cells))


def _unpaid_cells(zones: list[EmanationZone]) -> list[frozenset]:
    return [zone.cells for zone in zones if not zone.saved]


def _clear_square(actor, origin: tuple[int, int], square: tuple[int, int], zones: list[EmanationZone]) -> bool:
    for cells in _unpaid_cells(zones):
        if _body_in(actor, square, cells) or path_enters(actor, origin, square, cells):
            return False
    return True


def _with_exits(encounter, actor, reachable: set[tuple[int, int]], chosen: list[tuple[int, int]]):
    """Add the nearest square outside every unpaid emanation, and the exit nearest a foe."""
    zones = hostile_zones(encounter, actor)
    if not _unpaid_cells(zones):
        return chosen
    origin = tuple(actor.position)
    clear = [square for square in reachable if _clear_square(actor, origin, square, zones)]
    if not clear:
        return chosen
    picks = list(chosen)

    def add(square: tuple[int, int]) -> None:
        if square not in picks:
            picks.append(square)

    add(min(clear, key=lambda square: (chebyshev(square, origin), square)))
    foes = _foes(encounter, actor)
    if foes:
        def foe_key(square: tuple[int, int]):
            nearest = min(space_between(square, actor, foe) for foe in foes)
            return nearest, chebyshev(square, origin), square

        add(min(clear, key=foe_key))
    return picks


def spell_range_feet(actor, spell) -> int:
    """Spell Sniper adds 60 feet to an attack spell whose range is already 10 feet or more."""
    reach = int(getattr(spell, "range_feet", 0) or 0)
    if actor.traits.get("spell_sniper") and getattr(spell, "kind", "") == "attack" and reach >= 10:
        return reach + 60
    return reach


def _slot_levels(actor, spell, automated: bool) -> list[int]:
    if spell.level == 0:
        return [0]
    available = [level for level, count in sorted(actor.slots.items()) if count > 0 and level >= spell.level]
    free_name = (actor.traits.get("free_spells") or {}).get(spell.index)
    if free_name and int(actor.uses_left.get(free_name, 0)) > 0 and spell.level not in available:
        available = [spell.level, *available]
    if automated:
        return available[:1]
    return available


def _square_in_range(actor, square: tuple[int, int], spell) -> bool:
    distance = actor.gap_to_square(square)
    if spell.range_mode == "self":
        return distance == 0
    if spell.range_mode == "touch":
        return distance <= 1
    return distance * 5 <= spell_range_feet(actor, spell)


def _creature_in_range(actor, target, spell) -> bool:
    if spell.area_type and spell.range_mode == "self":
        return False
    distance = actor.gap_to(target)
    if spell.range_mode == "touch":
        return distance <= 1
    if spell.range_mode == "self":
        return target.id == actor.id
    reach = spell_range_feet(actor, spell)
    if spell.kind == "heal":
        return distance * 5 <= reach
    return distance > 0 and distance * 5 <= reach


def _zone_aim_cells(encounter, actor, spell, action) -> set[tuple[int, int]]:
    spec = zone_spec(spell.index)
    if spec is None or action.aim is None:
        return set()
    origin = (int(action.aim[0]), int(action.aim[1]))
    form = action.attack_name or ""
    placed = {"index": spec.index, "x": origin[0], "y": origin[1], "form": form, "dx": 0, "dy": 0}
    if form == "line":
        step_x, step_y = compass(tuple(actor.position), origin)
        placed["dx"], placed["dy"] = step_x, step_y
    if spec.bolt:
        return bolt_cells(origin, spec.bolt)
    cells = cells_of(spec, placed)
    if spec.side:
        cells = side_band(cells, foe_points(encounter, actor), spec.side)
    return cells


def _token_target(encounter, actor, square, spec):
    """The nearest foe the token can strike from ``square``."""
    if square is None:
        return None
    origin = (int(square[0]), int(square[1]))
    best = None
    best_key = None
    reach = spec.near or 5
    for foe in _foes(encounter, actor):
        gap = space_gap(origin, spec.span, tuple(foe.position), foe.span)
        if gap * 5 > reach:
            continue
        key = (gap, foe.id)
        if best_key is None or key < best_key:
            best = foe
            best_key = key
    return best


def _add_zone_spell(actions, encounter, actor, spell, slot, automated, budget, tag, prefix, slot_text) -> None:
    spec = zone_spec(spell.index)
    if spec is None:
        return
    if spec.shape == "token":
        _add_token_spell(actions, encounter, actor, spell, spec, slot, automated, budget, tag, prefix, slot_text)
        return
    if spec.shape == "wall":
        _add_wall_spell(actions, encounter, actor, spell, spec, slot, automated, budget, tag, prefix, slot_text)
        return
    foes = _foes(encounter, actor)
    if automated:
        candidates = [square for foe in foes for square in sorted(foe.squares())]
    else:
        candidates = [(x, y) for x in range(encounter.width) for y in range(encounter.height)]
    seen: set[tuple[int, int]] = set()
    for square in candidates:
        if square in seen or not _square_in_range(actor, square, spell):
            continue
        seen.add(square)
        probe = GameAction(id=-1, kind="spell", label="", spell_index=spell.index, aim=square)
        cells = _zone_aim_cells(encounter, actor, spell, probe)
        if not any(creature.in_cells(cells) for creature in foes):
            continue
        _append(
            actions,
            kind="spell",
            label=f"{prefix}{spell.name} {slot_text} at ({square[0]}, {square[1]})",
            spell_index=spell.index,
            aim=square,
            slot_level=slot,
            budget=budget,
            tag=tag,
        )


def _add_token_spell(actions, encounter, actor, spell, spec, slot, automated, budget, tag, prefix, slot_text) -> None:
    for x in range(encounter.width):
        for y in range(encounter.height):
            square = (x, y)
            if not _square_in_range(actor, square, spell):
                continue
            if not footprint_free(encounter, square, spec.span):
                continue
            if automated and _token_target(encounter, actor, square, spec) is None:
                continue
            _append(
                actions,
                kind="spell",
                label=f"{prefix}{spell.name} {slot_text} at ({square[0]}, {square[1]})",
                spell_index=spell.index,
                aim=square,
                slot_level=slot,
                budget=budget,
                tag=tag,
            )


def _wall_hits(encounter, actor, spell, square, form) -> bool:
    probe = GameAction(
        id=-1,
        kind="spell",
        label="",
        spell_index=spell.index,
        aim=square,
        attack_name=form,
    )
    cells = _zone_aim_cells(encounter, actor, spell, probe)
    return any(creature.in_cells(cells) for creature in _foes(encounter, actor))


def _add_wall_spell(actions, encounter, actor, spell, spec, slot, automated, budget, tag, prefix, slot_text) -> None:
    forms = []
    if spec.ring and (not automated or spec.ring):
        forms.append("ring")
    if spec.length and not (automated and spec.ring):
        forms.append("line")
    if automated and spec.ring:
        forms = ["ring"]
    for form in forms:
        candidates = [(x, y) for x in range(encounter.width) for y in range(encounter.height)]
        for square in candidates:
            if not _square_in_range(actor, square, spell):
                continue
            if not _wall_hits(encounter, actor, spell, square, form):
                continue
            _append(
                actions,
                kind="spell",
                label=f"{prefix}{spell.name} {form} {slot_text} at ({square[0]}, {square[1]})",
                spell_index=spell.index,
                attack_name=form,
                aim=square,
                slot_level=slot,
                budget=budget,
                tag=tag,
            )


def targets_for(encounter, actor, action: GameAction) -> list:
    if action.kind == "attack":
        option = actor.attack_named(action.attack_name) if action.attack_name != "Multiattack" else None
        if option and option.area_type and (option.rider or {}).get("aimed") and action.aim is not None:
            cells = area_squares(tuple(action.aim), option.area_type, option.area_size or 0, None)
            return [
                creature
                for creature in encounter.combatants
                if creature.hp > 0 and creature.id != actor.id and creature.in_cells(cells)
            ]
        if option and option.area_type and action.facing:
            cells = area_squares(tuple(actor.position), option.area_type, option.area_size or 0, action.facing)
            return [
                creature
                for creature in encounter.combatants
                if creature.hp > 0 and creature.id != actor.id and creature.in_cells(cells)
            ]
        if action.target_id:
            target = encounter.by_id(action.target_id)
            return [target] if target and target.hp > 0 else []
        return []
    if action.kind != "spell":
        return []
    spell = actor.spell_named(action.spell_index)
    if spell is None:
        return []
    if spell.kind == "zone":
        cells = _zone_aim_cells(encounter, actor, spell, action)
        picked = [creature for creature in encounter.combatants if creature.hp > 0 and creature.in_cells(cells)]
        spec = zone_spec(spell.index)
        if spec is not None and spec.skip_self:
            picked = [creature for creature in picked if creature.id != actor.id]
        if spec is not None and spec.attack:
            foe = _token_target(encounter, actor, action.aim, spec)
            return [foe] if foe is not None else []
        return picked
    if spell.kind == "afflict" and spell.area_type == "cube" and action.aim is not None:
        cells = centered_block(tuple(action.aim), int(spell.area_size or 0))
        return [creature for creature in encounter.combatants if creature.hp > 0 and creature.in_cells(cells)]
    if spell.area_type:
        origin = tuple(actor.position) if spell.range_mode == "self" else tuple(action.aim or actor.position)
        cells = area_squares(origin, spell.area_type, spell.area_size or 0, action.facing)
        picked = [creature for creature in encounter.combatants if creature.hp > 0 and creature.in_cells(cells)]
        if spell.kind == "heal":
            return [creature for creature in picked if creature.side == actor.side]
        if spell.index in EMANATION_INDEXES:
            return [creature for creature in picked if creature.side != actor.side]
        return picked
    if action.target_id:
        target = encounter.by_id(action.target_id)
        return [target] if target else []
    return []


def _append(actions: list[GameAction], **kwargs) -> None:
    actions.append(GameAction(id=len(actions), **kwargs))


def _add_spiritual_weapon(actions, encounter, actor, spell, slot, automated, budget, tag, prefix, slot_text) -> None:
    """Empty squares within the spell's range. Automated play only aims next to a foe."""
    taken = _occupied(encounter)
    for x in range(encounter.width):
        for y in range(encounter.height):
            square = (x, y)
            if square in taken or not _square_in_range(actor, square, spell):
                continue
            if automated and adjacent_foe(encounter, actor, square) is None:
                continue
            _append(
                actions,
                kind="spell",
                label=f"{prefix}{spell.name} {slot_text} at ({square[0]}, {square[1]})",
                spell_index=spell.index,
                aim=square,
                slot_level=slot,
                budget=budget,
                tag=tag,
            )


def _add_moves(actions, encounter, actor, automated: bool) -> None:
    budget = movement_budget(actor)
    if budget < 5:
        return
    foes = _foes(encounter, actor)
    squares = _reachable(encounter, actor, budget)
    costs = {} if actor.traits.get("freedom") else difficult_costs(encounter)
    if costs and "prone" not in actor.conditions:
        origin = tuple(actor.position)
        squares = {
            square
            for square in squares
            if travel_cost(origin, square, actor.span, costs) <= budget
        }
    sources = frightened_sources(encounter, actor)
    if sources:
        squares = {
            square
            for square in squares
            if not any(closer_to_source(actor, square, source) for source in sources)
        }
    chosen = _tactical(actor, foes, squares) if automated else sorted(squares)
    if automated:
        chosen = _with_exits(encounter, actor, squares, chosen)
    for square in chosen:
        _append(
            actions,
            kind="move",
            label=f"Move to ({square[0]}, {square[1]})",
            destination=square,
            budget="move",
        )


def _add_aimed_save(actions, encounter, actor, option, activation: str, automated: bool) -> None:
    """A hurled sphere or cube. The square is the center, and the range is the throw."""
    reach = int(option.range_feet or 0)
    foes = _foes(encounter, actor)
    if automated:
        candidates = [square for creature in foes for square in sorted(creature.squares())]
    else:
        candidates = [(x, y) for x in range(encounter.width) for y in range(encounter.height)]
    seen: set[tuple[int, int]] = set()
    for square in candidates:
        if square in seen:
            continue
        if reach and actor.gap_to_square(square) * 5 > reach:
            continue
        seen.add(square)
        cells = area_squares(square, option.area_type, option.area_size or 0, None)
        caught = [creature for creature in foes if creature.in_cells(cells)]
        if not caught:
            continue
        if option.needs_sight and not any(can_see(actor, creature) for creature in caught):
            continue
        _append(
            actions,
            kind="attack",
            label=f"{option.name} at ({square[0]}, {square[1]})",
            attack_name=option.name,
            aim=square,
            budget=activation,
        )


def _add_attacks(
    actions,
    encounter,
    actor,
    activation: str,
    *,
    automated: bool = False,
    swings_only: bool = False,
    only_name: str | None = None,
) -> None:
    foes = _foes(encounter, actor)
    origin = tuple(actor.position)
    if activation == "action" and actor.multiattack and not swings_only and only_name is None:
        for foe in sorted(foes, key=lambda creature: creature.id):
            if any(
                weapon_reaches(actor, foe, option)
                for name, _count in actor.multiattack
                if (option := actor.attack_named(name)) and option.resolution == "attack" and _available(actor, option)
            ):
                _append(
                    actions,
                    kind="attack",
                    label=f"Multiattack -> {foe.name}",
                    attack_name="Multiattack",
                    target_id=foe.id,
                    budget="action",
                )
    for option in actor.attacks:
        if option.activation != activation or option.resolution == "disengage" or not _available(actor, option):
            continue
        if option.resolution == "invisible":
            _append(
                actions,
                kind="attack",
                label=option.name,
                attack_name=option.name,
                budget=activation,
            )
            continue
        if only_name and option.name != only_name:
            continue
        rider = option.rider or {}
        contest = bool(rider.get("grapple") or rider.get("shove"))
        if swings_only and not contest and (option.resolution != "attack" or option.area_type):
            continue
        if option.resolution == "attack":
            for foe in sorted(foes, key=lambda creature: creature.id):
                if option.needs_sight and not can_see(actor, foe):
                    continue
                if charmed_by(actor, foe):
                    continue
                if weapon_reaches(actor, foe, option):
                    _append(
                        actions,
                        kind="attack",
                        label=f"{option.name} -> {foe.name}",
                        attack_name=option.name,
                        target_id=foe.id,
                        budget=activation,
                    )
            continue
        if option.area_type and (option.rider or {}).get("aimed"):
            _add_aimed_save(actions, encounter, actor, option, activation, automated)
            continue
        if option.area_type:
            for name, facing in FACINGS.items():
                cells = area_squares(origin, option.area_type, option.area_size or 0, facing)
                if any(creature.hp > 0 and creature.id != actor.id and creature.in_cells(cells) for creature in foes):
                    _append(
                        actions,
                        kind="attack",
                        label=f"{option.name} facing {name}",
                        attack_name=option.name,
                        facing=facing,
                        budget=activation,
                    )
            continue
        for foe in sorted(foes, key=lambda creature: creature.id):
            if option.needs_sight and not can_see(actor, foe):
                continue
            if charmed_by(actor, foe):
                continue
            if contest and size_rank(getattr(foe, "size", None)) > size_rank(getattr(actor, "size", None)) + int(rider.get("size_step") or 0):
                continue
            if weapon_reaches(actor, foe, option) or (
                option.reach_feet is None and option.range_feet is None and actor.gap_to(foe) == 1
            ):
                _append(
                    actions,
                    kind="attack",
                    label=f"{option.name} -> {foe.name}",
                    attack_name=option.name,
                    target_id=foe.id,
                    budget=activation,
                )


_SKIPPED_SPELLS = {"hex", "hunters-mark", "divine-smite"}


def _afflict_pool(encounter, actor, spell, spec: dict) -> list:
    if spec.get("self_cast"):
        return [actor]
    if spec.get("ally") or spell.kind == "heal":
        pool = [creature for creature in _allies(encounter, actor, include_self=True) if creature.hp > 0]
    else:
        pool = _foes(encounter, actor)
    rows = []
    for target in pool:
        if target.traits.get("banished") or target.traits.get("removed"):
            continue
        if spec.get("humanoid") and not _humanoid(target):
            continue
        if spec.get("sight") and not can_see(actor, target):
            continue
        if not spec.get("ally") and spell.kind != "heal" and charmed_by(actor, target):
            continue
        if spec.get("each"):
            reach = int(spell.range_feet or 0)
            if spell.range_mode == "self":
                reach = int(spec.get("area_feet") or reach)
            if reach and actor.gap_to(target) * 5 > reach:
                continue
            rows.append(target)
            continue
        if spec.get("after_hit") and actor.traits.get("last_hit") != target.id:
            continue
        if not _creature_in_range(actor, target, spell) and not spec.get("area"):
            continue
        if spec.get("area") == "cube":
            if not _square_in_range(actor, tuple(target.position), spell):
                continue
        rows.append(target)
    return rows


def _humanoid(target) -> bool:
    from src.conditions import is_humanoid

    return is_humanoid(target)


def _add_afflict(actions, encounter, actor, spell, slot, automated, budget, tag, prefix, slot_text) -> None:
    spec = affliction(spell.index) or {}
    choices = list(spec.get("choices") or [""])
    if spec.get("special") in {"lesser", "greater"}:
        choices = list(spec.get("choices") or [])
    pool = _afflict_pool(encounter, actor, spell, spec)
    if spec.get("self_cast"):
        _append(
            actions,
            kind="spell",
            label=f"{prefix}{spell.name} {slot_text}",
            spell_index=spell.index,
            target_id=actor.id,
            slot_level=slot,
            budget=budget,
            tag=tag,
        )
        return
    if spec.get("each") or (spec.get("area") in {"cone", "line"} and spell.range_mode == "self"):
        if spec.get("area") in {"cone", "line"}:
            for name, facing in FACINGS.items():
                cells = area_squares(tuple(actor.position), spec["area"], int(spec.get("area_feet") or 0), facing)
                if not any(creature.in_cells(cells) for creature in pool):
                    continue
                _append(
                    actions,
                    kind="spell",
                    label=f"{prefix}{spell.name} {slot_text} facing {name}",
                    spell_index=spell.index,
                    facing=facing,
                    aim=tuple(actor.position),
                    slot_level=slot,
                    budget=budget,
                    tag=tag,
                )
            return
        if pool:
            _append(
                actions,
                kind="spell",
                label=f"{prefix}{spell.name} {slot_text}",
                spell_index=spell.index,
                slot_level=slot,
                budget=budget,
                tag=tag,
            )
        return
    if spec.get("area") == "sphere":
        seen = set()
        for target in pool:
            square = tuple(target.position)
            if square in seen or not _square_in_range(actor, square, spell):
                continue
            seen.add(square)
            cells = area_squares(square, "sphere", int(spec.get("area_feet") or 0), None)
            if not any(creature.in_cells(cells) for creature in pool):
                continue
            _append(
                actions,
                kind="spell",
                label=f"{prefix}{spell.name} {slot_text} at ({square[0]}, {square[1]})",
                spell_index=spell.index,
                aim=square,
                slot_level=slot,
                budget=budget,
                tag=tag,
            )
        return
    if spec.get("area") == "cube":
        seen = set()
        for target in pool:
            square = tuple(target.position)
            if square in seen:
                continue
            seen.add(square)
            _append(
                actions,
                kind="spell",
                label=f"{prefix}{spell.name} {slot_text} at ({square[0]}, {square[1]})",
                spell_index=spell.index,
                aim=square,
                slot_level=slot,
                budget=budget,
                tag=tag,
            )
        return
    for choice in choices:
        for target in sorted(pool, key=lambda creature: creature.id):
            if spec.get("special") in {"lesser", "greater"} and choice not in _removable(target, choice):
                continue
            if not _creature_in_range(actor, target, spell):
                continue
            label = f"{prefix}{spell.name}"
            if choice:
                label += f" ({choice})"
            label += f" {slot_text} -> {target.name}"
            _append(
                actions,
                kind="spell",
                label=label,
                spell_index=spell.index,
                attack_name=choice or None,
                target_id=target.id,
                slot_level=slot,
                budget=budget,
                tag=tag,
            )


def _removable(target, choice: str) -> set[str]:
    if choice == "exhaustion":
        return {"exhaustion"} if int(target.conditions.get("exhaustion") or 0) else set()
    if choice in target.conditions:
        return {choice}
    return set()


def _add_spells(
    actions,
    encounter,
    actor,
    automated: bool,
    activation: str,
    *,
    budget: str | None = None,
    tag: str | None = None,
) -> None:
    budget = budget or activation
    if budget == "action" and actor.traits.get("surge_blocks_magic"):
        return
    foes = _foes(encounter, actor)
    allies = [creature for creature in _allies(encounter, actor, include_self=True) if creature.hp < creature.max_hp]
    prefix = "Quickened " if tag == "quickened" else ""
    for spell in actor.spells:
        if spell.index in _SKIPPED_SPELLS or spell.kind == "excluded" or spell.activation != activation:
            continue
        if spell.verbal and actor.traits.get("inside_silence"):
            continue
        if spell.level > 0 and actor.traits.get("leveled_cast"):
            continue
        for slot in _slot_levels(actor, spell, automated):
            slot_text = "cantrip" if slot == 0 else f"L{slot}"
            found = affliction(spell.index)
            if found and found.get("mode") == "cast" and zone_spec(spell.index) is None:
                _add_afflict(actions, encounter, actor, spell, slot, automated, budget, tag, prefix, slot_text)
                continue
            if spell.index == "spiritual-weapon":
                _add_spiritual_weapon(actions, encounter, actor, spell, slot, automated, budget, tag, prefix, slot_text)
                continue
            if zone_spec(spell.index) is not None:
                _add_zone_spell(actions, encounter, actor, spell, slot, automated, budget, tag, prefix, slot_text)
                continue
            if spell.area_type and spell.range_mode == "self" and spell.area_type in {"sphere", "cylinder"}:
                cells = area_squares(tuple(actor.position), spell.area_type, spell.area_size or 0, None)
                hits_foe = any(creature.in_cells(cells) for creature in foes)
                hits_ally = any(creature.in_cells(cells) and creature.hp < creature.max_hp for creature in allies)
                if (spell.kind == "heal" and hits_ally) or (spell.kind != "heal" and hits_foe):
                    _append(
                        actions,
                        kind="spell",
                        label=f"{prefix}{spell.name} {slot_text}",
                        spell_index=spell.index,
                        aim=tuple(actor.position),
                        slot_level=slot,
                        budget=budget,
                        tag=tag,
                    )
                continue
            if spell.area_type and spell.range_mode == "self" and spell.area_type in {"cone", "line", "cube"}:
                for name, facing in FACINGS.items():
                    cells = area_squares(tuple(actor.position), spell.area_type, spell.area_size or 0, facing)
                    hits = [creature for creature in foes if creature.in_cells(cells)]
                    if not hits:
                        continue
                    _append(
                        actions,
                        kind="spell",
                        label=f"{prefix}{spell.name} {slot_text} facing {name}",
                        spell_index=spell.index,
                        facing=facing,
                        slot_level=slot,
                        budget=budget,
                        tag=tag,
                    )
                continue
            if spell.area_type:
                if automated:
                    pool = allies if spell.kind == "heal" else foes
                    candidates = [square for creature in pool for square in sorted(creature.squares())]
                else:
                    candidates = [
                        (x, y)
                        for x in range(encounter.width)
                        for y in range(encounter.height)
                    ]
                seen = set()
                for square in candidates:
                    if square in seen or not _square_in_range(actor, square, spell):
                        continue
                    seen.add(square)
                    cells = area_squares(square, spell.area_type, spell.area_size or 0, None)
                    if spell.kind == "heal":
                        if not any(creature.in_cells(cells) and creature.hp < creature.max_hp for creature in allies):
                            continue
                    elif not any(creature.in_cells(cells) for creature in foes):
                        continue
                    _append(
                        actions,
                        kind="spell",
                        label=f"{prefix}{spell.name} {slot_text} at ({square[0]}, {square[1]})",
                        spell_index=spell.index,
                        aim=square,
                        slot_level=slot,
                        budget=budget,
                        tag=tag,
                    )
                continue
            pool = allies if spell.kind == "heal" else foes
            for target in sorted(pool, key=lambda creature: creature.id):
                if not _creature_in_range(actor, target, spell):
                    continue
                if spell.needs_sight and not can_see(actor, target):
                    continue
                if spell.kind != "heal" and charmed_by(actor, target):
                    continue
                _append(
                    actions,
                    kind="spell",
                    label=f"{prefix}{spell.name} {slot_text} -> {target.name}",
                    spell_index=spell.index,
                    target_id=target.id,
                    slot_level=slot,
                    budget=budget,
                    tag=tag,
                )


def _add_order(actions, encounter, actor, mark: dict) -> None:
    name = str(mark.get("order") or "")
    source = None
    try:
        source = encounter.by_id(str(mark.get("source") or ""))
    except KeyError:
        source = None
    if name in {"grovel", "drop", "halt"} or source is None:
        _append(actions, kind="order", label=name.title() or "Obey", attack_name=name, budget="action")
        return
    budget = movement_budget(actor)
    if name == "away":
        budget += turn_speed(actor)
    squares = _reachable(encounter, actor, budget) if budget >= 5 else set()
    dest = None
    if squares:
        if name == "approach":
            ranked = sorted(squares, key=lambda square: (space_between(square, actor, source), square))
            dest = ranked[0]
            for square in ranked:
                if space_between(square, actor, source) <= 1:
                    dest = square
                    break
            if actor.gap_to(source) <= 1:
                dest = None
        else:
            ranked = sorted(squares, key=lambda square: (space_between(square, actor, source), square), reverse=True)
            dest = ranked[0]
            if space_between(dest, actor, source) <= actor.gap_to(source):
                dest = None
    if name == "away" and dest is None:
        return
    _append(
        actions,
        kind="order",
        label=name.title(),
        attack_name=name,
        destination=dest,
        budget="action",
    )


def _add_escape(actions, actor) -> None:
    from src.conditions import marks_of

    if not actor.action_available or is_incapacitated(actor):
        return
    for mark in marks_of(actor):
        if mark.get("grapple") or mark.get("spell") == "ensnaring-strike":
            _append(actions, kind="escape", label="Escape (Athletics)", attack_name="athletics", budget="action")
            _append(actions, kind="escape", label="Escape (Acrobatics)", attack_name="acrobatics", budget="action")
            return


def _add_shake(actions, encounter, actor) -> None:
    from src.conditions import marks_of

    if not actor.action_available or is_incapacitated(actor):
        return
    for other in encounter.combatants:
        if other.id == actor.id or other.hp <= 0 or actor.gap_to(other) > 1:
            continue
        if any(mark.get("shake") for mark in marks_of(other)):
            _append(
                actions,
                kind="shake",
                label=f"Shake {other.name}",
                target_id=other.id,
                budget="action",
            )


def attack_window(actor) -> str:
    """open, extra, nick, or closed. Extra Attack keeps the action for weapon swings."""
    per = max(1, int(actor.attacks_per_action or 1))
    nick = nick_weapon_name(actor)
    if actor.attacks_made <= 0 and actor.action_available:
        return "open"
    if actor.attacks_made < per and actor.action_available:
        return "extra"
    if nick:
        return "nick"
    return "closed"


def nick_weapon_name(actor) -> str | None:
    used = str(actor.traits.get("light_used") or "")
    if not used or actor.traits.get("nick_used"):
        return None
    nicks = [
        option
        for option in actor.attacks
        if option.mastery == "nick" and option.light and option.activation == "action"
    ]
    for option in nicks:
        if option.name != used:
            return option.name
    if sum(1 for option in nicks if option.name == used) >= 2:
        return used
    return None


def legal_actions(encounter, actor, *, automated: bool) -> list[GameAction]:
    from src.features import add_character_actions

    actions: list[GameAction] = []
    if actor.hp <= 0 or actor.traits.get("banished") or actor.traits.get("removed"):
        return actions
    order = pending_order(actor)
    if order and not is_incapacitated(actor):
        _add_order(actions, encounter, actor, order)
        if actions:
            return actions
    if is_incapacitated(actor) or actor.traits.get("lock_action"):
        _add_moves(actions, encounter, actor, automated)
        _append(actions, kind="end", label="End turn", budget="end")
        return actions
    if can_stand(actor):
        _append(actions, kind="stand", label="Stand", budget="move")
    _add_escape(actions, actor)
    _add_shake(actions, encounter, actor)
    _add_moves(actions, encounter, actor, automated)
    window = attack_window(actor)
    if window == "open":
        _add_attacks(actions, encounter, actor, "action", automated=automated)
        _add_spells(actions, encounter, actor, automated, "action")
        _append(actions, kind="dash", label="Dash", budget="action")
        _append(actions, kind="disengage", label="Disengage", budget="action")
        _append(actions, kind="dodge", label="Dodge", budget="action")
    elif window == "extra":
        _add_attacks(actions, encounter, actor, "action", swings_only=True)
    elif window == "nick":
        nick = nick_weapon_name(actor)
        if nick:
            _add_attacks(actions, encounter, actor, "action", swings_only=True, only_name=nick)
    add_character_actions(actions, encounter, actor, automated, _append)
    _add_monk_strikes(actions, encounter, actor)
    if actor.bonus_available:
        _add_attacks(actions, encounter, actor, "bonus")
        _add_spells(actions, encounter, actor, automated, "bonus")
        for option in actor.attacks:
            if option.activation == "bonus" and option.resolution == "disengage":
                _append(
                    actions,
                    kind="disengage",
                    label=f"{option.name}: Disengage",
                    attack_name=option.name,
                    budget="bonus",
                )
    _append(actions, kind="end", label="End turn", budget="end")
    return actions


def _add_monk_strikes(actions, encounter, actor) -> None:
    """Bonus unarmed strike after an Attack action, and the two Flurry strikes."""
    profile = actor.traits.get("monk") or {}
    if not isinstance(profile, dict):
        return
    option = actor.attack_named("Unarmed Strike")
    if option is None:
        return
    foes = [
        foe
        for foe in sorted(_foes(encounter, actor), key=lambda creature: creature.id)
        if weapon_reaches(actor, foe, option) and not charmed_by(actor, foe)
    ]
    if int(actor.traits.get("flurry_left") or 0) > 0:
        for foe in foes:
            _append(
                actions,
                kind="attack",
                label=f"Flurry: Unarmed Strike -> {foe.name}",
                attack_name="Unarmed Strike",
                target_id=foe.id,
                budget="bonus",
                tag="flurry",
            )
    if actor.bonus_available and actor.traits.get("monk_attacked") and profile.get("bonus_unarmed"):
        for foe in foes:
            _append(
                actions,
                kind="attack",
                label=f"Bonus Unarmed Strike -> {foe.name}",
                attack_name="Unarmed Strike",
                target_id=foe.id,
                budget="bonus",
                tag="monk",
            )


def spell_parts(actor, spell, slot_level: int) -> list[tuple[str, str]]:
    from src.dice import substitute_mod

    resolved = []
    for damage_type, expression in spell_damage_parts(spell, slot_level, actor.caster_level):
        if "MOD" in expression.upper():
            expression = substitute_mod(expression, actor.spellcasting_mod)
        resolved.append((damage_type, expression))
    return resolved


def heal_expression(actor, spell, slot_level: int) -> str | None:
    return spell_heal_expression(spell, slot_level, actor.spellcasting_mod)


def distance_feet(left, right) -> int:
    return left.gap_to(right) * 5
