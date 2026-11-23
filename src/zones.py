"""Placed zones and conjured tokens.

The size, the trigger, and the shape come from this catalog. A radius mentioned
later in a spell description, such as the light around Flaming Sphere, is not
the damage area. Spirit Guardians and Conjure Woodland Beings are not here.
They stay self-centered emanations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.grid import footprint, in_bounds, space_gap, sphere, step_path

NOT_AURA = {"flame-blade", "shining-smite"}

Square = tuple[int, int]


@dataclass(frozen=True)
class ZoneSpec:
    """One lasting spell. ``when`` lists cast, enter, end, start, push, and squares."""

    index: str
    shape: str
    size_feet: int = 0
    cost: int = 0
    save: str = ""
    success: str = "half"
    damage: str = ""
    damage_type: str = ""
    scale: int = 0
    scale_n: int = 1
    condition: str = ""
    escape: bool = False
    hold: str = ""
    skip_self: bool = False
    when: str = ""
    attack: bool = False
    move: str = ""
    move_feet: int = 0
    span: int = 1
    near: int = 0
    blocks: bool = False
    bolt: int = 0
    ring: int = 0
    length: int = 0
    side: int = 0
    later: str = ""
    later_type: str = ""
    later_save: bool = True
    appear: str = ""
    appear_type: str = ""
    linger: int = 0
    lock_action: bool = False
    until: str = ""
    drop_conc: bool = False
    blind_inside: bool = False
    deaf_inside: bool = False
    silence: bool = False
    band: int = 0

    def allows(self, trigger: str) -> bool:
        return trigger in self.when.split(",")

    @property
    def area_type(self) -> str | None:
        if self.shape in {"sphere", "cylinder", "square", "cube"}:
            return self.shape
        return None

    @property
    def area_size(self) -> int | None:
        if self.area_type is None:
            return None
        return self.size_feet


def _spec(**kwargs) -> ZoneSpec:
    return ZoneSpec(**kwargs)


ZONES: dict[str, ZoneSpec] = {
    item.index: item
    for item in (
        _spec(index="spike-growth", shape="sphere", size_feet=20, cost=2, damage="2d4", damage_type="piercing", when="squares"),
        _spec(index="insect-plague", shape="sphere", size_feet=20, cost=2, save="constitution", damage="4d10", damage_type="piercing", scale=10, when="cast,enter,end"),
        _spec(index="cloudkill", shape="sphere", size_feet=20, save="constitution", damage="5d8", damage_type="poison", scale=8, when="cast,enter,end,push", move="drift", move_feet=10),
        _spec(index="incendiary-cloud", shape="sphere", size_feet=20, save="dexterity", damage="10d8", damage_type="fire", when="cast,enter,end,push", move="drift", move_feet=10),
        _spec(index="moonbeam", shape="cylinder", size_feet=5, save="constitution", damage="2d10", damage_type="radiant", scale=10, when="cast,enter,end,push", move="action", move_feet=60),
        _spec(index="black-tentacles", shape="square", size_feet=20, cost=2, save="strength", success="none", damage="3d6", damage_type="bludgeoning", condition="restrained", escape=True, hold="spell", when="cast,enter,end"),
        _spec(index="web", shape="cube", size_feet=20, cost=2, save="dexterity", success="none", condition="restrained", escape=True, hold="inside", when="enter,start"),
        _spec(index="entangle", shape="square", size_feet=20, cost=2, save="strength", success="none", condition="restrained", escape=True, hold="spell", skip_self=True, when="cast,enter,end"),
        _spec(index="flaming-sphere", shape="token", save="dexterity", damage="2d6", damage_type="fire", scale=6, near=5, when="end,push", move="bonus", move_feet=30),
        _spec(index="conjure-animals", shape="token", span=2, save="dexterity", success="none", damage="3d10", damage_type="slashing", scale=10, near=10, when="enter,end,push", move="follow", move_feet=30),
        _spec(index="conjure-fey", shape="token", attack=True, damage="3d12+MOD", damage_type="psychic", scale=12, near=5, condition="frightened", when="cast,push", move="bonus", move_feet=30),
        _spec(index="conjure-elemental", shape="token", span=2, save="dexterity", success="none", damage="8d8", damage_type="fire", later="4d8", later_type="fire", scale=8, near=5, condition="restrained", hold="spell", when="enter,start"),
        _spec(index="arcane-sword", shape="token", attack=True, damage="4d12+MOD", damage_type="force", near=5, when="cast,push", move="bonus", move_feet=30),
        _spec(index="call-lightning", shape="cloud", size_feet=60, save="dexterity", damage="3d10", damage_type="lightning", scale=10, bolt=5, when="cast", move="bolt", move_feet=0),
        _spec(index="wall-of-fire", shape="wall", save="dexterity", damage="5d8", damage_type="fire", scale=8, ring=10, length=60, side=10, later_save=False, when="cast,enter,end"),
        _spec(index="wall-of-thorns", shape="wall", cost=4, save="dexterity", appear="7d8", appear_type="piercing", later="7d8", later_type="slashing", scale=8, ring=10, length=60, when="cast,enter,end"),
        _spec(index="wall-of-ice", shape="wall", save="dexterity", damage="10d6", damage_type="cold", scale=6, scale_n=2, ring=10, length=100, blocks=True, when="cast"),
        _spec(index="blade-barrier", shape="wall", cost=2, save="dexterity", damage="6d10", damage_type="force", ring=30, length=100, when="cast,enter,end"),
        _spec(index="wind-wall", shape="wall", save="strength", damage="4d8", damage_type="bludgeoning", length=50, when="cast"),
        _spec(index="wall-of-force", shape="wall", ring=10, length=100, blocks=True),
        _spec(index="wall-of-stone", shape="wall", length=100, blocks=True),
        _spec(index="faerie-fire", shape="cube", size_feet=20, save="dexterity", success="none", condition="outlined", when="cast"),
        _spec(index="grease", shape="square", size_feet=10, cost=2, save="dexterity", success="none", condition="prone", when="cast,enter,end", linger=10),
        _spec(index="stinking-cloud", shape="sphere", size_feet=20, save="constitution", success="none", condition="poisoned", when="start", lock_action=True, until="end"),
        _spec(index="sleet-storm", shape="cylinder", size_feet=20, cost=2, save="dexterity", success="none", condition="prone", when="enter,start", drop_conc=True, blind_inside=True),
        _spec(index="silence", shape="sphere", size_feet=20, deaf_inside=True, silence=True),
        _spec(index="earthquake", shape="sphere", size_feet=100, cost=2, save="dexterity", success="none", condition="prone", when="cast,caster-end", drop_conc=True),
        _spec(index="prismatic-wall", shape="wall", length=90, save="constitution", success="none", condition="blinded", when="start", linger=100, until="rounds", band=20),
    )
}


def zone_spec(index: str | None) -> ZoneSpec | None:
    if not index:
        return None
    return ZONES.get(index)


def is_zone(index: str | None) -> bool:
    return index in ZONES


def _scale(expr: str, extra: int, sides: int) -> str:
    if extra <= 0 or sides <= 0 or not expr:
        return expr
    text = expr.replace(" ", "")
    suffix = ""
    for token in ("+MOD", "-MOD"):
        if text.endswith(token):
            suffix = token
            text = text[: -len(token)]
            break
    match = re.fullmatch(r"(\d+)d(\d+)([+-]\d+)?", text)
    if not match or int(match.group(2)) != sides:
        return expr
    count = int(match.group(1)) + extra
    return f"{count}d{sides}{match.group(3) or ''}{suffix}"


def effect_parts(spec: ZoneSpec, slot: int, level: int, phase: str) -> list[tuple[str, str]]:
    """Dice for one trigger. ``later`` is the enter/end/repeat damage when it differs."""
    if phase == "later" and spec.later:
        expr, kind = spec.later, spec.later_type or spec.damage_type
    elif phase == "appear" and spec.appear:
        expr, kind = spec.appear, spec.appear_type or spec.damage_type
    else:
        expr, kind = spec.damage, spec.damage_type
    if not expr or not kind:
        return []
    expr = _scale(expr, max(0, int(slot) - int(level)) * max(1, spec.scale_n), spec.scale)
    return [(kind, expr)]


def compass(origin: Square, aim: Square) -> tuple[int, int]:
    dx = int(aim[0]) - int(origin[0])
    dy = int(aim[1]) - int(origin[1])
    if dx == 0 and dy == 0:
        return (1, 0)
    step_x = 0 if dx == 0 else (1 if dx > 0 else -1)
    step_y = 0 if dy == 0 else (1 if dy > 0 else -1)
    return step_x, step_y


def centered_block(center: Square, size_feet: int) -> set[Square]:
    """An axis-aligned square centered on ``center``. An even side leans positive."""
    side = max(1, int(size_feet) // 5)
    start = -(side // 2) if side % 2 else -(side // 2) + 1
    cx, cy = int(center[0]), int(center[1])
    return {(cx + dx, cy + dy) for dx in range(start, start + side) for dy in range(start, start + side)}


def ring_cells(center: Square, radius_feet: int) -> set[Square]:
    radius = max(1, int(radius_feet) // 5)
    cx, cy = int(center[0]), int(center[1])
    return {
        (cx + dx, cy + dy)
        for dx in range(-radius, radius + 1)
        for dy in range(-radius, radius + 1)
        if max(abs(dx), abs(dy)) == radius
    }


def line_cells(origin: Square, facing: tuple[int, int], length_feet: int) -> set[Square]:
    count = max(1, int(length_feet) // 5)
    fx, fy = facing
    ox, oy = int(origin[0]), int(origin[1])
    return {(ox + fx * step, oy + fy * step) for step in range(count)}


def bolt_cells(center: Square, radius_feet: int) -> set[Square]:
    return sphere((int(center[0]), int(center[1])), int(radius_feet))


def cells_of(spec: ZoneSpec, placed: dict) -> set[Square]:
    """Squares the zone occupies. A cloud is the cloud, not the lightning bolt."""
    origin = (int(placed.get("x", 0)), int(placed.get("y", 0)))
    form = str(placed.get("form") or "")
    if spec.shape == "sphere":
        return sphere(origin, spec.size_feet)
    if spec.shape == "cylinder":
        return sphere(origin, spec.size_feet)
    if spec.shape in {"square", "cube"}:
        return centered_block(origin, spec.size_feet)
    if spec.shape == "token":
        return footprint(origin, spec.span)
    if spec.shape == "cloud":
        return sphere(origin, spec.size_feet)
    if spec.shape == "wall":
        if form == "line" or (not spec.ring and spec.length):
            facing = (int(placed.get("dx") or 1), int(placed.get("dy") or 0))
            return line_cells(origin, facing, spec.length)
        if spec.ring:
            return ring_cells(origin, spec.ring)
        return line_cells(origin, (int(placed.get("dx") or 1), int(placed.get("dy") or 0)), spec.length)
    return set()


def side_band(wall: set[Square], foes: list, band_feet: int) -> set[Square]:
    """Squares on the side of the wall nearest a foe, plus the wall itself."""
    if not wall or band_feet <= 0:
        return set(wall)
    band = max(1, int(band_feet) // 5)

    def nearest(cell: Square) -> int:
        return min(max(abs(cell[0] - foe[0]), abs(cell[1] - foe[1])) for foe in foes)

    if not foes:
        return set(wall)
    extra: set[Square] = set()
    for cell in wall:
        base = nearest(cell)
        for dx in range(-band, band + 1):
            for dy in range(-band, band + 1):
                if max(abs(dx), abs(dy)) > band or (dx, dy) == (0, 0):
                    continue
                square = (cell[0] + dx, cell[1] + dy)
                if nearest(square) < base:
                    extra.add(square)
    return set(wall) | extra


def each_zone(encounter) -> list[tuple]:
    """Concentration zones and timed areas such as Grease."""
    found = []
    for creature in encounter.combatants:
        zone = read_zone(creature)
        if zone is not None:
            found.append((creature, zone[0], zone[1]))
        for placed in list(creature.traits.get("lingers") or []):
            if not isinstance(placed, dict):
                continue
            spec = zone_spec(str(placed.get("index") or ""))
            if spec is None or int(placed.get("rounds") or 0) <= 0:
                continue
            found.append((creature, spec, placed))
    return found


def read_zone(creature) -> tuple[ZoneSpec, dict] | None:
    placed = creature.traits.get("zone")
    if creature.hp <= 0 or not isinstance(placed, dict):
        return None
    if creature.concentrating_on != placed.get("index"):
        return None
    spec = zone_spec(str(placed.get("index") or ""))
    if spec is None:
        return None
    return spec, placed


def anchor_cost(anchor: Square, span: int, costs: dict[Square, int]) -> int:
    """Movement units of 5 feet for stepping onto ``anchor``. One step, one charge."""
    found = [costs[cell] for cell in footprint(anchor, span) if cell in costs]
    return max(found) if found else 1


def difficult_costs(encounter) -> dict[Square, int]:
    costs: dict[Square, int] = {}
    for _creature, spec, placed in each_zone(encounter):
        if spec.cost <= 1:
            continue
        for cell in cells_of(spec, placed):
            costs[cell] = max(costs.get(cell, 1), spec.cost)
    return costs


def blocking_cells(encounter) -> set[Square]:
    cells: set[Square] = set()
    for _creature, spec, placed in each_zone(encounter):
        if spec.blocks:
            cells.update(cells_of(spec, placed))
    return cells


def travel_cost(origin: Square, dest: Square, span: int, costs: dict[Square, int]) -> int:
    """Feet charged along the straight step path. The origin square is free."""
    total = 0
    for square in step_path(origin, dest):
        total += anchor_cost(square, span, costs) * 5
    return total


def foe_points(encounter, caster) -> list[Square]:
    points = []
    for creature in encounter.combatants:
        if creature.hp <= 0 or creature.side == caster.side:
            continue
        points.append((int(creature.position[0]), int(creature.position[1])))
    return points


def paint_cells(encounter, caster, spec: ZoneSpec, placed: dict) -> set[Square]:
    cells = cells_of(spec, placed)
    if spec.side:
        cells = side_band(cells, foe_points(encounter, caster), spec.side)
    return cells


def _body_hits(anchor: Square, span: int, cells: set[Square]) -> bool:
    return bool(footprint(anchor, span) & cells)


def _grow(cells: set[Square], feet: int) -> set[Square]:
    reach = max(0, int(feet) // 5)
    if reach <= 0:
        return set(cells)
    grown = set(cells)
    for cell in cells:
        for dx in range(-reach, reach + 1):
            for dy in range(-reach, reach + 1):
                if max(abs(dx), abs(dy)) <= reach:
                    grown.add((cell[0] + dx, cell[1] + dy))
    return grown


def trigger_cells(encounter, caster, spec: ZoneSpec, placed: dict, *, later: bool) -> set[Square]:
    if spec.bolt and not later:
        return bolt_cells((int(placed.get("x", 0)), int(placed.get("y", 0))), spec.bolt)
    cells = cells_of(spec, placed)
    if later and spec.side:
        cells = side_band(cells, foe_points(encounter, caster), spec.side)
    if spec.band and not later:
        cells = _grow(cells, spec.band)
    return cells


def in_zone(encounter, caster, spec: ZoneSpec, placed: dict, anchor: Square, span: int, *, later: bool) -> bool:
    if spec.near and spec.shape == "token":
        gap = space_gap((int(placed["x"]), int(placed["y"])), spec.span, anchor, span)
        if later or "end" in spec.when or "enter" in spec.when:
            return gap * 5 <= spec.near
    return _body_hits(anchor, span, trigger_cells(encounter, caster, spec, placed, later=later))


def turn_mark(encounter) -> list[int]:
    return [int(encounter.round), int(encounter.turn_index)]


def already_saved(placed: dict, encounter, creature_id: str) -> bool:
    saved = placed.get("saved")
    if not isinstance(saved, dict) or list(saved.get("turn") or []) != turn_mark(encounter):
        return False
    return str(creature_id) in {str(item) for item in saved.get("ids") or []}


def mark_saved(placed: dict, encounter, creature_ids) -> None:
    saved = placed.get("saved")
    if not isinstance(saved, dict) or list(saved.get("turn") or []) != turn_mark(encounter):
        saved = {"turn": turn_mark(encounter), "ids": []}
    ids = [str(item) for item in saved.get("ids") or []]
    for creature_id in creature_ids:
        text = str(creature_id)
        if text not in ids:
            ids.append(text)
    saved["ids"] = ids
    placed["saved"] = saved


def drift_delta(caster_at: Square, zone_at: Square) -> tuple[int, int]:
    """Ten feet directly away from the caster. East when the caster is on the zone."""
    dx = int(zone_at[0]) - int(caster_at[0])
    dy = int(zone_at[1]) - int(caster_at[1])
    if dx == 0 and dy == 0:
        return (2, 0)
    step_x = 0 if dx == 0 else (1 if dx > 0 else -1)
    step_y = 0 if dy == 0 else (1 if dy > 0 else -1)
    if step_x and step_y:
        return (2 * step_x, 2 * step_y)
    if step_x:
        return (2 * step_x, 0)
    return (0, 2 * step_y)


def follow_delta(origin: Square, dest: Square, move_feet: int) -> tuple[int, int] | None:
    path = step_path(origin, dest)
    cap = max(0, int(move_feet) // 5)
    path = path[:cap]
    if not path:
        return None
    last = path[-1]
    return last[0] - int(origin[0]), last[1] - int(origin[1])


def footprint_free(encounter, anchor: Square, span: int) -> bool:
    body = footprint(anchor, span)
    width = int(encounter.width)
    height = int(encounter.height)
    if any(not in_bounds(cell, width, height) for cell in body):
        return False
    for creature in encounter.combatants:
        if creature.hp > 0 and creature.squares() & body:
            return False
    return True
