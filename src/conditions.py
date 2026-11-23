"""Conditions, the spells that set them, and monster riders parsed from SRD text.

This module does not import the encounter, the rules, or the bestiary. Callers
pass creatures and roll the dice. Marks live on ``traits["marks"]``. The
``conditions`` dict is the active set, rebuilt by ``sync``.
"""

from __future__ import annotations

import itertools
import re

from src.grid import feet_between, space_gap, step_path
from src.zones import anchor_cost, difficult_costs

_IDS = itertools.count(1)

CONDITIONS = (
    "blinded",
    "charmed",
    "deafened",
    "exhaustion",
    "frightened",
    "grappled",
    "incapacitated",
    "invisible",
    "paralyzed",
    "petrified",
    "poisoned",
    "prone",
    "restrained",
    "stunned",
    "unconscious",
)
INCAPACITATING = {"incapacitated", "paralyzed", "petrified", "stunned", "unconscious"}
SPEED_LOCK = {"grappled", "restrained", "paralyzed", "petrified", "unconscious"}
SIZE_RANK = {"tiny": 0, "small": 1, "medium": 2, "large": 3, "huge": 4, "gargantuan": 5}
EVIL_TYPES = {"aberration", "celestial", "elemental", "fey", "fiend", "undead"}
OUTSIDERS = {"aberration", "celestial", "elemental", "fey", "fiend"}
BANISH_TYPES = {"celestial", "elemental", "fey", "fiend"}
ACTION_DENY = {"paralyzed", "stunned", "unconscious", "incapacitated", "petrified"}

# mode cast is a condition spell. mode rider adds a condition to a hit or a failed save.
# mode heal restores hit points and then clears the named conditions.
AFFLICTIONS: dict[str, dict] = {
    "hold-person": {"mode": "cast", "condition": "paralyzed", "save": "wisdom", "until": "concentration", "repeat": True, "sight": True, "humanoid": True, "extra": True, "concentration": True, "rounds": 10},
    "hold-monster": {"mode": "cast", "condition": "paralyzed", "save": "wisdom", "until": "concentration", "repeat": True, "sight": True, "extra": True, "concentration": True, "rounds": 10},
    "blindness-deafness": {"mode": "cast", "save": "constitution", "until": "rounds", "repeat": True, "sight": True, "extra": True, "rounds": 10, "choices": ("blinded", "deafened")},
    "color-spray": {"mode": "cast", "condition": "blinded", "save": "constitution", "until": "source-next-end", "area": "cone", "area_feet": 15},
    "hideous-laughter": {"mode": "cast", "special": "laughter", "save": "wisdom", "sight": True, "concentration": True, "rounds": 10},
    "sleep": {"mode": "cast", "special": "sleep", "save": "wisdom", "area": "sphere", "area_feet": 5, "concentration": True, "rounds": 10},
    "hypnotic-pattern": {"mode": "cast", "special": "hypnotic", "save": "wisdom", "area": "cube", "area_feet": 30, "sight": True, "concentration": True, "rounds": 10},
    "charm-person": {"mode": "cast", "condition": "charmed", "save": "wisdom", "until": "fight", "sight": True, "humanoid": True, "extra": True, "charm_damage": True, "fight_advantage": True},
    "charm-monster": {"mode": "cast", "condition": "charmed", "save": "wisdom", "until": "fight", "sight": True, "extra": True, "charm_damage": True, "fight_advantage": True},
    "banishment": {"mode": "cast", "special": "banish", "save": "charisma", "sight": True, "concentration": True, "rounds": 10, "extra": True},
    "power-word-stun": {"mode": "cast", "special": "power-stun", "sight": True},
    "flesh-to-stone": {"mode": "cast", "special": "flesh", "save": "constitution", "sight": True, "concentration": True, "rounds": 10},
    "irresistible-dance": {"mode": "cast", "special": "dance", "save": "wisdom", "sight": True, "concentration": True, "rounds": 10},
    "eyebite": {"mode": "cast", "special": "eyebite", "save": "wisdom", "sight": True, "concentration": True, "rounds": 10, "choices": ("asleep", "panicked", "sickened")},
    "command": {"mode": "cast", "special": "command", "save": "wisdom", "sight": True, "extra": True, "choices": ("approach", "drop", "flee", "grovel", "halt")},
    "divine-word": {"mode": "cast", "special": "divine", "save": "charisma", "each": True},
    "fear": {"mode": "cast", "special": "fear", "save": "wisdom", "area": "cone", "area_feet": 30, "concentration": True, "rounds": 10},
    "ensnaring-strike": {"mode": "cast", "special": "ensnare", "save": "strength", "concentration": True, "rounds": 10, "after_hit": True},
    "invisibility": {"mode": "cast", "condition": "invisible", "until": "fight", "ally": True, "extra": True, "concentration": True, "ends_on_attack": True},
    "greater-invisibility": {"mode": "cast", "condition": "invisible", "until": "concentration", "ally": True, "concentration": True, "rounds": 10},
    "mislead": {"mode": "cast", "condition": "invisible", "until": "fight", "self_cast": True, "concentration": True, "ends_on_attack": True},
    "see-invisibility": {"mode": "cast", "special": "see", "self_cast": True},
    "heroism": {"mode": "cast", "special": "heroism", "ally": True, "concentration": True, "rounds": 10, "extra": True},
    "protection-from-evil-and-good": {"mode": "cast", "special": "protection", "ally": True, "concentration": True},
    "freedom-of-movement": {"mode": "cast", "special": "freedom", "ally": True},
    "calm-emotions": {"mode": "cast", "special": "calm", "save": "charisma", "area": "sphere", "area_feet": 20, "concentration": True, "rounds": 10, "humanoid": True},
    "lesser-restoration": {"mode": "cast", "special": "lesser", "ally": True, "choices": ("blinded", "deafened", "paralyzed", "poisoned")},
    "greater-restoration": {"mode": "cast", "special": "greater", "ally": True, "choices": ("exhaustion", "charmed", "petrified")},
    "heal": {"mode": "heal", "healing": {str(level): str(70 + 10 * (level - 6)) for level in range(6, 10)}, "clears": ("blinded", "deafened", "poisoned")},
    "mass-heal": {"mode": "cast", "special": "mass-heal", "each": True, "ally": True},
    "power-word-heal": {"mode": "cast", "special": "power-heal", "sight": True, "ally": True},
    "protection-from-poison": {"mode": "cast", "special": "poison-ward", "ally": True},
    "ray-of-sickness": {"mode": "rider", "on": "hit", "condition": "poisoned", "until": "source-next-end"},
    "sunbeam": {"mode": "rider", "on": "fail", "condition": "blinded", "until": "start", "area": "line", "area_feet": 60},
    "sunburst": {"mode": "rider", "on": "fail", "condition": "blinded", "until": "rounds", "rounds": 10, "repeat": True, "save": "constitution", "area": "sphere", "area_feet": 60},
    "weird": {"mode": "rider", "on": "fail", "condition": "frightened", "until": "concentration", "repeat": True, "save": "wisdom", "weird": True, "area": "sphere", "area_feet": 30, "concentration": True},
    "contagion": {"mode": "rider", "on": "fail", "condition": "poisoned", "until": "fight", "save": "constitution", "damage": "11d8", "damage_type": "necrotic"},
    "prismatic-spray": {"mode": "rider", "on": "fail", "special": "prismatic", "area": "cone", "area_feet": 60},
    "starry-wisp": {"mode": "rider", "on": "hit", "clear_invisible": True, "until": "source-next-end"},
    "shining-smite": {"mode": "cast", "special": "smite", "after_hit": True, "clear_invisible": True, "concentration": True, "until": "concentration"},
    "faerie-fire": {"mode": "rider", "on": "fail", "clear_invisible": True},
}


def affliction(index: str | None) -> dict | None:
    if not index:
        return None
    return AFFLICTIONS.get(index)


def holds_concentration(spell) -> bool:
    """True when this cast should keep the caster's concentration."""
    if spell is None:
        return False
    from src.zones import zone_spec

    spec = zone_spec(getattr(spell, "index", None))
    if spec is not None and spec.linger:
        return False
    if getattr(spell, "ongoing", False) or getattr(spell, "index", "") == "spiritual-weapon":
        return True
    if spec is not None:
        return True
    found = affliction(getattr(spell, "index", None))
    return bool(found and found.get("concentration"))


def marks_of(creature) -> list[dict]:
    rows = creature.traits.get("marks")
    if not isinstance(rows, list):
        return []
    return rows


def _store(creature, rows: list[dict]) -> None:
    creature.traits["marks"] = rows


def exhaustion_level(creature) -> int:
    return int(creature.conditions.get("exhaustion") or 0)


def is_incapacitated(creature) -> bool:
    if creature.hp <= 0:
        return True
    if creature.traits.get("banished") or creature.traits.get("removed"):
        return True
    return any(name in creature.conditions for name in INCAPACITATING)


def speed_locked(creature) -> bool:
    if creature.hp <= 0 or creature.traits.get("banished") or creature.traits.get("removed"):
        return True
    if any(name in creature.conditions for name in SPEED_LOCK):
        return True
    return bool(creature.traits.get("dance") or creature.traits.get("speed_zero"))


def base_speed(creature) -> int:
    speed = int(creature.speed)
    level = exhaustion_level(creature)
    if level >= 6:
        return 0
    speed -= 5 * level
    if not creature.traits.get("freedom"):
        speed -= int(creature.traits.get("slow") or 0)
    if any(mark.get("half_speed") for mark in marks_of(creature)):
        speed //= 2
    return max(0, speed)


def stand_cost(creature) -> int:
    return base_speed(creature) // 2


def can_stand(creature) -> bool:
    if "prone" not in creature.conditions or is_incapacitated(creature):
        return False
    if any(mark.get("condition") == "prone" and mark.get("no_stand") for mark in marks_of(creature)):
        return False
    if speed_locked(creature) or base_speed(creature) <= 0:
        return False
    left = creature.movement_left
    budget = base_speed(creature) if left is None else int(left)
    return budget >= stand_cost(creature)


def size_rank(name: str | None) -> int:
    return SIZE_RANK.get(str(name or "medium").strip().lower(), 2)


def creature_types(creature) -> set[str]:
    text = str(getattr(creature, "creature_type", "") or "").lower()
    return {part.strip() for part in re.split(r"[\s,]+", text) if part.strip()}


def is_humanoid(creature) -> bool:
    kinds = creature_types(creature)
    if not kinds:
        return True
    return "humanoid" in kinds


def is_elf(creature) -> bool:
    species = str(creature.traits.get("species") or "").lower()
    return "elf" in species


def immune_to(target, condition: str, source=None) -> bool:
    name = condition.lower()
    owned = {str(item).lower() for item in target.traits.get("condition_immunities") or []}
    if name in owned or name == "exhaustion" and "exhaustion" in owned:
        return True
    if name == "poisoned" and "petrified" in target.conditions:
        return True
    if name == "frightened" and target.traits.get("heroism"):
        return True
    if name in {"paralyzed", "restrained", "grappled"} and target.traits.get("freedom"):
        return True
    if name in {"charmed", "frightened"} and target.traits.get("calm"):
        return True
    if name in {"charmed", "frightened"} and target.traits.get("ward_evil") and source is not None:
        if creature_types(source) & EVIL_TYPES:
            return True
    if name == "invisible" and target.traits.get("no_invisible"):
        return True
    return False


def sync(creature) -> None:
    """Rebuild the active condition set from marks, plus 0 hit points."""
    active: dict[str, int] = {}
    dance = False
    speed_zero = False
    lock_action = False
    heroism_temp = 0
    for mark in marks_of(creature):
        condition = str(mark.get("condition") or "")
        if condition == "exhaustion":
            active["exhaustion"] = max(active.get("exhaustion", 0), int(mark.get("level") or 1))
        elif condition:
            active[condition] = 99
        if condition in INCAPACITATING:
            active["incapacitated"] = 99
        if condition == "unconscious":
            active["prone"] = 99
            active["incapacitated"] = 99
        if mark.get("dance"):
            dance = True
        if mark.get("speed_zero"):
            speed_zero = True
        if mark.get("lock_action"):
            lock_action = True
        if mark.get("heroism"):
            heroism_temp = max(heroism_temp, int(mark.get("temp") or 0))
    if creature.hp <= 0:
        active["unconscious"] = 99
        active["incapacitated"] = 99
        active["prone"] = 99
    if any(mark.get("calm") for mark in marks_of(creature)):
        active.pop("charmed", None)
        active.pop("frightened", None)
    if creature.traits.get("outline_source") or "outlined" in active:
        active["outlined"] = 99
    level = exhaustion_level(creature)
    if "exhaustion" not in active and level and not marks_of(creature):
        pass
    creature.conditions = active
    creature.traits["dance"] = dance
    creature.traits["speed_zero"] = speed_zero
    creature.traits["lock_action"] = lock_action
    creature.traits["heroism"] = heroism_temp > 0
    creature.traits["ward_evil"] = any(mark.get("ward") for mark in marks_of(creature))
    creature.traits["freedom"] = any(mark.get("freedom") for mark in marks_of(creature))
    creature.traits["calm"] = any(mark.get("calm") for mark in marks_of(creature))
    creature.traits["see_invisible"] = any(mark.get("see") for mark in marks_of(creature))
    creature.traits["poison_ward"] = any(mark.get("poison_ward") for mark in marks_of(creature))
    creature.traits["no_invisible"] = any(mark.get("no_invisible") for mark in marks_of(creature))
    if heroism_temp:
        creature.traits["heroism_temp"] = heroism_temp
    else:
        creature.traits.pop("heroism_temp", None)
    if exhaustion_level(creature) >= 6 and creature.hp > 0:
        creature.hp = 0
        active["unconscious"] = 99
        active["incapacitated"] = 99
        creature.conditions = active


def add_mark(target, source=None, **fields) -> str:
    """Apply one mark. An immunity is logged and does not change the target."""
    condition = str(fields.get("condition") or "")
    if condition and immune_to(target, condition, source):
        return f"{target.name} is immune to {condition}."
    if condition == "exhaustion":
        level = min(6, exhaustion_level(target) + int(fields.get("level") or 1))
        fields["level"] = level
    mark = {
        "id": next(_IDS),
        "condition": condition,
        "source": getattr(source, "id", "") or str(fields.get("source_id") or ""),
        "source_side": getattr(source, "side", "") or str(fields.get("source_side") or ""),
        "source_name": getattr(source, "name", "") or str(fields.get("source_name") or ""),
        "spell": str(fields.get("spell") or ""),
        "until": str(fields.get("until") or "fight"),
        "rounds": int(fields.get("rounds") or 0),
        "save": str(fields.get("save") or ""),
        "dc": int(fields.get("dc") or 0),
        "no_stand": bool(fields.get("no_stand")),
        "ends_on_damage": bool(fields.get("ends_on_damage")),
        "damage_save": bool(fields.get("damage_save")),
        "ends_on_attack": bool(fields.get("ends_on_attack")),
        "charm_damage": bool(fields.get("charm_damage")),
        "lock_action": bool(fields.get("lock_action")),
        "speed_zero": bool(fields.get("speed_zero")),
        "dance": bool(fields.get("dance")),
        "repeat": bool(fields.get("repeat")),
        "stage": int(fields.get("stage") or 0),
        "order": str(fields.get("order") or ""),
        "repeat_order": bool(fields.get("repeat_order")),
        "shake": bool(fields.get("shake")),
        "banish": bool(fields.get("banish")),
        "heroism": bool(fields.get("heroism")),
        "temp": int(fields.get("temp") or 0),
        "successes": int(fields.get("successes") or 0),
        "failures": int(fields.get("failures") or 0),
        "tied": str(fields.get("tied") or ""),
        "grapple": bool(fields.get("grapple")),
        "escape_dc": int(fields.get("escape_dc") or 0),
        "reach": int(fields.get("reach") or 0),
        "level": int(fields.get("level") or 0),
        "tick": str(fields.get("tick") or ""),
        "weird": bool(fields.get("weird")),
        "ward": bool(fields.get("ward")),
        "freedom": bool(fields.get("freedom")),
        "calm": bool(fields.get("calm")),
        "see": bool(fields.get("see")),
        "poison_ward": bool(fields.get("poison_ward")),
        "no_invisible": bool(fields.get("no_invisible")),
        "half_speed": bool(fields.get("half_speed")),
        "next_attack": bool(fields.get("next_attack")),
        "no_opportunity": bool(fields.get("no_opportunity")),
        "sleep_stage": int(fields.get("sleep_stage") or 0),
        "second": str(fields.get("second") or ""),
        "track": str(fields.get("track") or ""),
        "seen": False,
        "armed": False,
    }
    rows = marks_of(target)
    rows.append(mark)
    _store(target, rows)
    if condition in INCAPACITATING:
        target.concentrating_on = None
        target.ongoing_slot = None
        target.traits.pop("zone", None)
        target.traits.pop("spiritual_weapon", None)
    sync(target)
    if not condition:
        return ""
    return f"{target.name} is {condition}."


def drop_marks(target, predicate) -> list[str]:
    kept = []
    lines = []
    for mark in marks_of(target):
        if predicate(mark):
            condition = str(mark.get("condition") or "")
            if condition:
                lines.append(f"{target.name} is no longer {condition}.")
            continue
        kept.append(mark)
    _store(target, kept)
    sync(target)
    return lines


def clear_condition(target, condition: str) -> list[str]:
    if condition == "exhaustion":
        return drop_marks(target, lambda mark: mark.get("condition") == "exhaustion")
    return drop_marks(target, lambda mark: mark.get("condition") == condition)


def stand(target) -> list[str]:
    return drop_marks(target, lambda mark: mark.get("condition") == "prone" and not mark.get("no_stand"))


def break_on_attack(creature) -> list[str]:
    lines = drop_marks(creature, lambda mark: mark.get("ends_on_attack"))
    return lines


def note_damage(target, source, amount: int) -> list[str]:
    """End marks that damage ends. A charm ends when the charmer's side deals the damage."""
    if amount <= 0:
        return []
    lines = []
    if source is not None:
        lines.extend(break_on_attack(source))
    kept = []
    for mark in marks_of(target):
        ended = False
        if mark.get("ends_on_damage"):
            ended = True
        elif mark.get("charm_damage") and source is not None and source.side == mark.get("source_side"):
            ended = True
        elif mark.get("damage_save"):
            mark["pending_save"] = True
            mark["save_advantage"] = True
        if ended:
            condition = str(mark.get("condition") or "the effect")
            lines.append(f"{target.name} is no longer {condition}.")
            continue
        kept.append(mark)
    _store(target, kept)
    sync(target)
    return lines


def can_see(attacker, target) -> bool:
    if target.traits.get("banished") or target.traits.get("removed"):
        return False
    hidden = "invisible" in target.conditions and "outlined" not in target.conditions
    if not hidden:
        return True
    if attacker.traits.get("see_invisible"):
        return True
    gap = attacker.gap_to(target) * 5
    for key in ("blindsight", "truesight", "tremorsense"):
        if int(attacker.traits.get(key) or 0) >= gap and gap > 0:
            return True
    return False


def charmed_by(actor, other) -> bool:
    for mark in marks_of(actor):
        if mark.get("condition") == "charmed" and mark.get("source") == other.id:
            return True
    return False


def frightened_sources(encounter, actor) -> list:
    ids = {mark.get("source") for mark in marks_of(actor) if mark.get("condition") == "frightened" and mark.get("source")}
    found = []
    for creature in encounter.combatants:
        if creature.id in ids and creature.hp > 0:
            found.append(creature)
    return found


def closer_to_source(actor, square, source) -> bool:
    from src.actions import space_between

    return space_between(square, actor, source) < actor.gap_to(source)


def condition_reasons(attacker, target) -> tuple[list[str], list[str]]:
    """Advantage reasons, then disadvantage reasons, from conditions."""
    advantage: list[str] = []
    disadvantage: list[str] = []
    for name in ("poisoned", "restrained", "frightened", "blinded", "prone"):
        if name in attacker.conditions:
            disadvantage.append(f"the attacker is {name}")
    if attacker.traits.get("inside_blind"):
        disadvantage.append("the attacker is in magical darkness")
    if attacker.traits.get("dance"):
        disadvantage.append("the attacker is dancing")
    for name in ("restrained", "unconscious", "stunned", "outlined", "paralyzed", "petrified", "blinded"):
        if name in target.conditions:
            advantage.append(f"the target is {name}")
    if target.traits.get("inside_blind"):
        advantage.append("the target is in magical darkness")
    if target.traits.get("dance"):
        advantage.append("the target is dancing")
    if "prone" in target.conditions:
        if attacker.gap_to(target) <= 1:
            advantage.append("the target is prone and within 5 feet")
        else:
            disadvantage.append("the target is prone and more than 5 feet away")
    if "invisible" in attacker.conditions and "outlined" not in attacker.conditions and not can_see(target, attacker):
        advantage.append("the attacker is invisible and unseen")
    if "invisible" in target.conditions and "outlined" not in target.conditions and not can_see(attacker, target):
        disadvantage.append("the target is invisible and unseen")
    if target.traits.get("ward_evil") and creature_types(attacker) & EVIL_TYPES:
        disadvantage.append("the target is warded against this creature type")
    return advantage, disadvantage


def condition_edges(attacker, target) -> tuple[bool, bool]:
    """Advantage, then disadvantage, from conditions. Feats are applied by the caller."""
    advantage, disadvantage = condition_reasons(attacker, target)
    return bool(advantage), bool(disadvantage)


def save_mode(target, ability: str) -> int | None:
    """None means the Strength or Dexterity save fails without a roll."""
    full = ability.lower()
    if full in {"strength", "dexterity", "str", "dex"}:
        full = {"str": "strength", "dex": "dexterity"}.get(full, full)
    if full in {"strength", "dexterity"} and any(name in target.conditions for name in ("paralyzed", "stunned", "unconscious", "petrified")):
        return None
    mode = 0
    if full == "dexterity" and (target.dodging or target.traits.get("danger_sense")):
        mode += 1
    if full == "dexterity" and ("restrained" in target.conditions or target.traits.get("dance")):
        mode -= 1
    return max(-1, min(1, mode))


def attack_penalty(creature) -> int:
    return 2 * exhaustion_level(creature)


def save_penalty(creature) -> int:
    return 2 * exhaustion_level(creature)


def within_five(attacker, target) -> bool:
    return attacker.gap_to(target) <= 1


def auto_crit(attacker, target) -> bool:
    if not within_five(attacker, target):
        return False
    return any(name in target.conditions for name in ("unconscious", "paralyzed"))


def movement_cost(encounter, actor, origin, dest) -> int:
    """Feet charged along the path. A prone creature pays 10 feet for each new square."""
    prone = "prone" in actor.conditions
    costs = {} if actor.traits.get("freedom") else difficult_costs(encounter)
    if not costs and not prone:
        return feet_between(origin, dest)
    total = 0
    for square in step_path(origin, dest):
        step = 1 if actor.traits.get("freedom") or not costs else anchor_cost(square, actor.span, costs)
        if prone:
            step += 1
        total += step * 5
    return total


def step_cost(encounter, actor):
    """A weighted step, or None when every square costs 5 feet."""
    prone = "prone" in actor.conditions
    costs = {} if actor.traits.get("freedom") else difficult_costs(encounter)
    if not costs and not prone:
        return None

    def step(anchor, span=actor.span, table=costs, crawling=prone, free=bool(actor.traits.get("freedom"))):
        amount = 1 if free or not table else anchor_cost(anchor, span, table)
        if crawling:
            amount += 1
        return amount

    return step


def charged_toward(actor, target, feet: int) -> bool:
    """True when the last steps of this turn closed on the target in a straight line."""
    path = list(actor.traits.get("move_path") or [])
    steps = max(1, int(feet) // 5)
    if len(path) <= steps:
        return False
    if int(actor.traits.get("moved_feet") or 0) < int(feet):
        return False
    previous = path[-(steps + 1)]
    span = actor.span
    last_gap = space_gap(previous, span, tuple(target.position), target.span)
    for square in path[-steps:]:
        gap = space_gap(square, span, tuple(target.position), target.span)
        if gap > last_gap:
            return False
        last_gap = gap
    start_gap = space_gap(path[-(steps + 1)], span, tuple(target.position), target.span)
    return actor.gap_to(target) < start_gap


def condition_labels(creature) -> list[str]:
    names = []
    level = exhaustion_level(creature)
    if level:
        names.append(f"Exhaustion {level}")
    for name in CONDITIONS:
        if name == "exhaustion":
            continue
        if name in creature.conditions:
            names.append(name.title())
    if "outlined" in creature.conditions:
        names.append("Outlined")
    if creature.traits.get("inside_deaf") and "Deafened" not in names:
        names.append("Deafened")
    if creature.traits.get("inside_blind") and "Blinded" not in names:
        names.append("Blinded")
    if creature.traits.get("banished"):
        names.append("Banished")
    return names


def pending_order(creature) -> dict | None:
    for mark in marks_of(creature):
        if mark.get("order") and not mark.get("order_used"):
            return mark
    return None


def mark_by_id(creature, mark_id) -> dict | None:
    for mark in marks_of(creature):
        if mark.get("id") == mark_id:
            return mark
    return None


def _source_alive(encounter, mark) -> bool:
    source_id = mark.get("source")
    if not source_id:
        return True
    try:
        owner = encounter.by_id(source_id)
    except KeyError:
        return False
    if owner.hp <= 0:
        return False
    if mark.get("tied") == "concentration" or mark.get("until") == "concentration":
        return owner.concentrating_on == mark.get("spell")
    return True


def reap(encounter, *, moment: str = "") -> list[str]:
    """Drop marks whose source, concentration, or timing has ended."""
    lines = []
    current = None
    try:
        current = encounter.current()
    except Exception:
        current = None
    for creature in encounter.combatants:
        kept = []
        for mark in marks_of(creature):
            if mark.get("until") == "start" and moment == "start" and current is not None and mark.get("source") == current.id:
                lines.append(f"{creature.name} is no longer {mark.get('condition') or 'affected'}.")
                continue
            if mark.get("until") == "own" and moment == "start" and current is not None and creature.id == current.id:
                lines.append(f"{creature.name} is no longer {mark.get('condition') or 'affected'}.")
                continue
            if mark.get("condition") == "frightened" and not _source_alive(encounter, mark):
                lines.append(f"{creature.name} is no longer frightened.")
                continue
            if mark.get("tied") == "concentration" or mark.get("until") == "concentration":
                if not _source_alive(encounter, mark):
                    lines.append(f"{creature.name} is no longer {mark.get('condition') or 'affected'}.")
                    continue
            if mark.get("grapple"):
                owner = None
                try:
                    owner = encounter.by_id(mark.get("source"))
                except KeyError:
                    owner = None
                if owner is None or is_incapacitated(owner):
                    lines.append(f"{creature.name} is no longer grappled.")
                    continue
                reach = int(mark.get("reach") or 5)
                if owner.gap_to(creature) * 5 > reach:
                    lines.append(f"{creature.name} is no longer grappled.")
                    continue
            kept.append(mark)
        if not any(mark.get("grapple") for mark in kept):
            still = []
            for mark in kept:
                if mark.get("tied") == "grapple":
                    lines.append(f"{creature.name} is no longer {mark.get('condition') or 'restrained'}.")
                    continue
                still.append(mark)
            kept = still
        _store(creature, kept)
        if creature.traits.get("banished") and not any(mark.get("banish") for mark in kept):
            creature.traits.pop("banished", None)
            lines.append(f"{creature.name} returns.")
        sync(creature)
    return lines


def normalize_text(text: str) -> str:
    cleaned = (text or "").replace("\u2019", "'").replace("\u2018", "'").replace("\u2014", "-")
    return re.sub(r"\s+", " ", cleaned).strip()


def parse_rider(desc: str) -> dict:
    """Read the repeating condition sentences on a monster action. Pack Tactics is not a rider."""
    text = normalize_text(desc)
    low = text.lower()
    rider: dict = {}
    size = re.search(
        r"if the target is a (tiny|small|medium|large|huge|gargantuan) or smaller creature",
        low,
    )
    grapple = re.search(r"grappled condition \(escape dc (\d+)\)", low)
    prone = "prone condition" in low
    charge = re.search(r"moved (\d+)\+? feet straight toward", low)
    if size and grapple:
        rider["grapple_size"] = size.group(1)
        rider["escape_dc"] = int(grapple.group(1))
        if "restrained" in low and "grapple ends" in low:
            rider["restrain_while_grappled"] = True
    elif size and charge and prone:
        rider["prone_size"] = size.group(1)
        rider["charge_feet"] = int(charge.group(1))
    elif size and prone:
        rider["prone_size"] = size.group(1)
    if "undead or elf" in low:
        rider["skip_types"] = ["undead"]
        rider["skip_species"] = ["elf"]
    ability = re.search(
        r"(strength|dexterity|constitution|intelligence|wisdom|charisma) saving throw:\s*dc (\d+)",
        low,
    )
    fail = re.search(
        r"failure:\s*(?:the target )?(?:has the (\w+) condition|is (poisoned|paralyzed|restrained|stunned|blinded|frightened|charmed|prone|unconscious|incapacitated|petrified))",
        low,
    )
    first = re.search(r"first failure:.*?(\w+) condition", low)
    second = re.search(r"second failure:.*?(\w+) condition", low)
    if first and ability:
        rider["save_ability"] = ability.group(1)
        rider["save_dc"] = int(ability.group(2))
        rider["first"] = first.group(1)
        rider["second"] = second.group(1) if second else ""
    elif fail and ability and "saving throw" in low:
        condition = fail.group(1) or fail.group(2)
        rider["save_ability"] = ability.group(1)
        rider["save_dc"] = int(ability.group(2))
        rider["save_condition"] = condition
        if "until the start" in low:
            rider["save_until"] = "start"
        else:
            rider["save_until"] = "end"
    if "save_condition" not in rider and "first" not in rider:
        hit = re.search(
            r"the target has the (\w+) condition until the (end|start) of (?:its|the \w+'s) next turn",
            low,
        )
        if hit and hit.group(1) != "prone":
            rider["hit_condition"] = hit.group(1)
            rider["hit_until"] = "start" if hit.group(2) == "start" else "end"
    if "becomes invisible" in low or ("invisible condition" in low and "bonus" in low):
        rider["become"] = "invisible"
    if re.search(r"\bcan see\b", low):
        rider["needs_sight"] = True
    return rider


def parse_special(desc: str) -> dict:
    """Auras, Pack Tactics, and a trait that starts the creature invisible."""
    low = normalize_text(desc).lower()
    found: dict = {}
    if "advantage on attack rolls" in low and "within 5 feet" in low and "incapacitated" in low:
        found["pack_tactics"] = True
        return found
    if re.search(r"\bhas the invisible condition\b", low) and "can't benefit" not in low and "until" not in low:
        found["starts_invisible"] = True
    if "starts its turn" not in low and "start its turn" not in low:
        return found
    feet = re.search(r"within (\d+) feet", low)
    save = re.search(
        r"(strength|dexterity|constitution|intelligence|wisdom|charisma) saving throw:\s*dc (\d+)",
        low,
    )
    cond = re.search(r"failure:.*?(\w+) condition", low)
    if feet and save and cond:
        found["aura"] = {
            "feet": int(feet.group(1)),
            "save": save.group(1),
            "dc": int(save.group(2)),
            "condition": cond.group(1),
            "until": "start" if "until the start" in low else "end",
        }
    return found


def rider_skipped(target, rider: dict) -> bool:
    kinds = creature_types(target)
    if any(kind in kinds for kind in rider.get("skip_types") or []):
        return True
    species = str(target.traits.get("species") or "").lower()
    return any(name in species for name in rider.get("skip_species") or [])


def hit_rider_marks(attacker, target, rider: dict, *, toward: bool) -> list[str]:
    """Marks a hit adds. The caller has already decided the attack hit."""
    if not rider or rider_skipped(target, rider):
        return []
    lines = []
    rank = size_rank(target.size)
    prone_size = rider.get("prone_size")
    if prone_size and rank <= size_rank(prone_size):
        if rider.get("charge_feet"):
            if toward:
                lines.append(add_mark(target, attacker, condition="prone", until="stand"))
        else:
            lines.append(add_mark(target, attacker, condition="prone", until="stand"))
    grapple_size = rider.get("grapple_size")
    if grapple_size and rank <= size_rank(grapple_size):
        lines.append(
            add_mark(
                target,
                attacker,
                condition="grappled",
                until="fight",
                grapple=True,
                escape_dc=int(rider.get("escape_dc") or 10),
                reach=int(rider.get("reach") or getattr(attacker, "reach", 5) or 5),
            )
        )
        if rider.get("restrain_while_grappled"):
            lines.append(add_mark(target, attacker, condition="restrained", until="fight", tied="grapple"))
    hit_condition = rider.get("hit_condition")
    if hit_condition:
        until = "start" if rider.get("hit_until") == "start" else "end"
        lines.append(add_mark(target, attacker, condition=hit_condition, until=until, spell=rider.get("name") or ""))
    return [line for line in lines if line]


def spell_targets_extra(slot: int, level: int) -> int:
    return 1 + max(0, int(slot) - int(level))


def deny_weight(condition: str) -> float:
    if condition in ACTION_DENY:
        return 8.0
    if condition:
        return 3.0
    return 0.0
