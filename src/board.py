"""Battle-mat payload and the Streamlit component that draws it.

The component owns hover. Python sends the legal squares and the template
for each one, and the page only hears about a committed click.
"""

from __future__ import annotations

import base64
import re
from pathlib import Path

from src.actions import EMANATION_INDEXES
from src.conditions import condition_labels
from src.zones import each_zone, paint_cells, read_zone
from src.grid import area_squares, footprint
from src.paths import IMPORTS
from src.targeting import (
    actions_for,
    board_marks,
    chosen_slot,
    facing_for_square,
    offer_weapons,
    preview_squares,
)

LABEL_BAND = 32
_TOKEN_INDEX = re.compile(r"^[a-z0-9-]+$")
_PORTRAIT_NAME = re.compile(r"^\d+\.portrait$")
_CLASS_ICONS = {
    "barbarian",
    "bard",
    "cleric",
    "druid",
    "fighter",
    "monk",
    "paladin",
    "ranger",
    "rogue",
    "sorcerer",
    "warlock",
    "wizard",
}

BOARD_DIR = Path(__file__).resolve().parents[1] / "board"
TOKENS_DIR = BOARD_DIR / "tokens"
_component = None


def battle_payload(encounter, *, mode, actor, actions, abilities, armed, slot, selected_id) -> dict:
    """JSON the board draws. Coordinates are [x, y] with north at the top."""
    if mode == "combat" and actor is not None:
        abilities = offer_weapons(encounter, actor, list(abilities))
    wheel = [
        ability
        for ability in abilities
        if mode == "combat" and ability.budget in {"action", "bonus"} and ability.kind != "end"
    ]
    if mode != "combat" or actor is None or armed is None:
        armed = None
        slot = None
    elif armed.budget not in {"action", "bonus"} or armed.kind == "end":
        armed = None
        slot = None
    else:
        slot = chosen_slot(armed, slot)

    if mode == "combat" and actor is not None:
        move, aim, targets, preview = board_marks(encounter, actor, actions, armed, slot, None)
        hovers = _hovers(encounter, actor, actions, armed, slot)
        if not hovers:
            hovers = _move_hovers(actor, move)
    else:
        move, aim, targets, preview = set(), set(), set(), set()
        hovers = {}

    target_squares = _on_board(encounter, targets)
    acting_id = actor.id if actor is not None else None
    return {
        "mode": mode,
        "width": int(encounter.width),
        "height": int(encounter.height),
        "actorId": acting_id,
        "selectedId": selected_id,
        "armed": _armed_payload(armed, slot),
        "abilities": [_ability_payload(ability) for ability in wheel],
        "tokens": [
            *_weapon_tokens(encounter),
            *[
                _token(creature, acting_id, selected_id, target_squares)
                for creature in encounter.combatants
            ],
        ],
        "move": _pairs(encounter, move),
        "aim": _pairs(encounter, aim),
        "targets": _pairs(encounter, targets),
        "preview": _pairs(encounter, preview),
        "aura": _pairs(encounter, _aura_cells(encounter)),
        "hovers": hovers,
        "hint": _hint(mode, armed, targets=targets, aim=aim, names=_names_on(encounter, targets), actor=actor),
    }


def render_board(spec: dict, *, key: str):
    """Draw the mat. Returns the last click, which Streamlit repeats until it changes."""
    # Streamlit sizes the iframe from this arg before the component answers.
    # A missing handshake used to leave the mat at height 0.
    return _declare()(spec=spec, height=_frame_height(spec), default=None, key=key)


def _frame_height(spec: dict) -> int:
    # The mat then shrinks each square to the window and reports its real height.
    # A 20-row grid at 64 pixels would open already scrolled.
    rows = int(spec.get("height") or 1)
    return min(rows * 64 + 96 + LABEL_BAND, 720)


def _declare():
    global _component
    if _component is None:
        import streamlit.components.v1 as components

        _component = components.declare_component("battle_board", path=str(BOARD_DIR))
    return _component


def _hovers(encounter, actor, actions, ability, slot) -> dict[str, list[list[int]]]:
    if ability is None:
        return {}
    rows = actions_for(actions, ability, slot)
    if ability.needs == "square":
        found = {}
        for action in rows:
            if action.aim is None:
                continue
            found[_coord_key(action.aim)] = _pairs(encounter, preview_squares(encounter, actor, action))
        return found
    if ability.needs != "facing":
        return {}
    origin = tuple(actor.position)
    by_facing = {}
    for action in rows:
        if action.facing is None:
            continue
        by_facing[tuple(action.facing)] = _pairs(encounter, preview_squares(encounter, actor, action))
    found = {}
    for y in range(encounter.height):
        for x in range(encounter.width):
            facing = facing_for_square(origin, (x, y))
            if facing is None or facing not in by_facing:
                continue
            found[f"{x},{y}"] = by_facing[facing]
    return found


def _move_hovers(actor, squares) -> dict[str, list[list[int]]]:
    if actor is None or actor.span <= 1:
        return {}
    found = {}
    for anchor in squares:
        body = footprint(anchor, actor.span)
        found[f"{int(anchor[0])},{int(anchor[1])}"] = [[x, y] for x, y in sorted(body)]
    return found


def _armed_payload(ability, slot) -> dict | None:
    if ability is None:
        return None
    return {
        "key": ability.key,
        "label": ability.label,
        "budget": ability.budget,
        "needs": ability.needs,
        "kind": ability.kind,
        "slot": None if slot is None else int(slot),
        "slots": [int(level) for level in ability.slots],
    }


def _ability_payload(ability) -> dict:
    payload = {
        "key": ability.key,
        "label": ability.label,
        "budget": ability.budget,
        "kind": ability.kind,
        "needs": ability.needs,
        "slots": [int(level) for level in ability.slots],
    }
    if ability.kind == "attack" and ability.budget == "action":
        payload["group"] = "Attack"
    return payload


def _weapon_tokens(encounter) -> list[dict]:
    """A Spiritual Weapon is drawn, but it is not a creature and has no hit points."""
    tokens = []
    for creature in encounter.combatants:
        if creature.hp <= 0 or creature.concentrating_on != "spiritual-weapon":
            continue
        placed = creature.traits.get("spiritual_weapon")
        if not isinstance(placed, dict):
            continue
        tokens.append(_marker(
            creature, "Spiritual Weapon", "SW", int(placed.get("x", 0)), int(placed.get("y", 0)), 1, "spiritual-weapon",
        ))
    tokens.extend(_zone_markers(encounter))
    return tokens


def _marker(creature, name: str, initials: str, x: int, y: int, span: int, stem: str | None = None) -> dict:
    return {
        "id": f"{stem or name}-{creature.id}",
        "name": name,
        "initials": initials,
        "side": creature.side,
        "x": x,
        "y": y,
        "span": span,
        "hp": 1,
        "maxHp": 1,
        "ac": 0,
        "defeated": False,
        "acting": False,
        "selected": False,
        "target": False,
        "marker": True,
    }


def _zone_markers(encounter) -> list[dict]:
    markers = []
    for creature in encounter.combatants:
        found = read_zone(creature)
        if found is None:
            continue
        spec, placed = found
        if spec.shape != "token":
            continue
        spell = creature.spell_named(spec.index)
        name = spell.name if spell is not None else spec.index
        markers.append(
            _marker(
                creature,
                name,
                _initials(name),
                int(placed.get("x", 0)),
                int(placed.get("y", 0)),
                spec.span,
                spec.index,
            )
        )
    return markers


def _token(creature, acting_id, selected_id, target_squares) -> dict:
    square = (int(creature.position[0]), int(creature.position[1]))
    payload = {
        "id": creature.id,
        "name": creature.name,
        "initials": _initials(creature.name),
        "side": creature.side,
        "x": square[0],
        "y": square[1],
        "span": int(creature.span),
        "hp": int(creature.hp),
        "maxHp": int(creature.max_hp),
        "ac": int(creature.ac),
        "defeated": creature.hp <= 0,
        "acting": creature.id == acting_id,
        "selected": creature.id == selected_id,
        "target": creature.hp > 0 and bool(creature.squares() & target_squares),
    }
    labels = condition_labels(creature)
    if labels:
        payload["conditions"] = labels
    image = token_image(creature)
    if image:
        payload["image"] = image
    return payload


def class_icon_name(class_name: str | None) -> str | None:
    """The first class on a sheet, when this app has an icon for it."""
    first = str(class_name or "").split("/")[0].strip().lower()
    if first not in _CLASS_ICONS:
        return None
    return f"class-{first}.png"


def token_image(creature, *, imports_dir: Path | None = None, tokens_dir: Path | None = None) -> str | None:
    """A portrait data URL, or a path the component can load next to index.html."""
    folder = imports_dir or IMPORTS
    icons = tokens_dir or TOKENS_DIR
    portrait = _portrait_data(creature, folder)
    if portrait:
        return portrait
    index = str(getattr(creature, "creature_index", "") or "")
    if _TOKEN_INDEX.fullmatch(index):
        relative = f"tokens/{index}.png"
        if (icons / f"{index}.png").is_file():
            return relative
    if getattr(creature, "side", "") == "party":
        icon = class_icon_name((getattr(creature, "traits", None) or {}).get("class_name"))
        if icon and (icons / icon).is_file():
            return f"tokens/{icon}"
    return None


def _portrait_data(creature, folder: Path) -> str | None:
    name = str((getattr(creature, "traits", None) or {}).get("portrait") or "")
    if not _PORTRAIT_NAME.fullmatch(name):
        return None
    path = folder / name
    if not path.is_file():
        return None
    try:
        data = path.read_bytes()
    except OSError:
        return None
    kind = _media_type(data)
    if kind is None or len(data) > 3_000_000:
        return None
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{kind};base64,{encoded}"


def _media_type(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _initials(name: str) -> str:
    letters = []
    for word in str(name).split():
        for char in word:
            if char.isalnum():
                letters.append(char.upper())
                break
        if len(letters) == 3:
            break
    return "".join(letters) or "?"


def _aura_cells(encounter) -> set[tuple[int, int]]:
    """Emanations follow the caster. A placed zone stays where it was put."""
    cells: set[tuple[int, int]] = set()
    for creature in encounter.combatants:
        if creature.hp <= 0 or creature.concentrating_on not in EMANATION_INDEXES:
            continue
        spell = creature.spell_named(creature.concentrating_on)
        if spell is None or not spell.area_type:
            continue
        cells.update(area_squares(tuple(creature.position), spell.area_type, int(spell.area_size or 0), None))
    for creature, spec, placed in each_zone(encounter):
        cells.update(paint_cells(encounter, creature, spec, placed))
    return cells


def _on_board(encounter, cells) -> set[tuple[int, int]]:
    return {
        (int(x), int(y))
        for x, y in cells
        if 0 <= int(x) < encounter.width and 0 <= int(y) < encounter.height
    }


def _pairs(encounter, cells) -> list[list[int]]:
    return [[x, y] for x, y in sorted(_on_board(encounter, cells))]


def _coord_key(square) -> str:
    return f"{int(square[0])},{int(square[1])}"


def _names_on(encounter, cells) -> list[str]:
    wanted = _on_board(encounter, cells)
    names = []
    for creature in encounter.combatants:
        if creature.hp > 0 and creature.squares() & wanted:
            names.append(creature.name)
    return names


def _hint(mode: str, armed, targets=None, aim=None, names=None, actor=None) -> str:
    if mode == "place":
        return "Click a token to select it. Click an empty square to move it."
    if armed is None:
        if actor is not None and actor.span > 1:
            return (
                "Green squares are where this creature's corner can move. "
                "Point at one to see the whole body. Open Action or Bonus beside the current token."
            )
        return "Green squares are movement. Open Action or Bonus beside the current token."
    if armed.needs in {"creature", "square", "facing"} and not targets and not aim:
        return "Nobody is in reach. Move closer."
    if armed.needs == "creature":
        if names:
            return "Click a gold-ringed creature: " + ", ".join(names) + "."
        return "Click a creature with a gold ring."
    if armed.needs == "square":
        return "Point at a gold square. The amber template follows the pointer. Click to cast."
    if armed.needs == "facing":
        return "Point in a direction. Click to cast."
    if armed.needs == "confirm":
        return "The amber squares are the area. Click one to cast."
    return "Click the ability to resolve it."
