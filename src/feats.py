"""Player's Handbook feats this fight can roll.

Epic boons are named here and left as notes. The SRD API does not publish the
rest of the Player's Handbook, so the names live in this table. Nothing here
imports the encounter, the sheet loader, or reactions.
"""

from __future__ import annotations

import random
import re

from src.bestiary import ability_mod
from src.dice import RollResult, parse_dice, roll_dice

_SPACE = re.compile(r"\s+")

EPIC_BOONS = frozenset({
    "boon of combat prowess",
    "boon of dimensional travel",
    "boon of energy resistance",
    "boon of fate",
    "boon of fortitude",
    "boon of irresistible offense",
    "boon of recovery",
    "boon of skill",
    "boon of speed",
    "boon of spell recall",
    "boon of the night spirit",
    "boon of truesight",
})

LISTED = frozenset({
    "actor",
    "athlete",
    "chef",
    "crafter",
    "durable",
    "grappler",
    "inspiring leader",
    "keen mind",
    "mounted combatant",
    "musician",
    "observant",
    "ritual caster",
    "sharpshooter",
    "skill expert",
    "skilled",
    "skulker",
    "telepathic",
    "blind fighting",
})

QUIET = frozenset({
    "ability score improvement",
    "tough",
    "resilient",
    "lightly armored",
    "moderately armored",
    "heavily armored",
    "martial weapon training",
    "great weapon master",
    "alert",
    "savage attacker",
    "lucky",
    "healer",
    "tavern brawler",
    "unarmed fighting",
    "archery",
    "defense",
    "dueling",
    "great weapon fighting",
    "two weapon fighting",
    "crossbow expert",
    "thrown weapon fighting",
    "interception",
    "protection",
    "charger",
    "crusher",
    "piercer",
    "slasher",
    "defensive duelist",
    "dual wielder",
    "heavy armor master",
    "medium armor master",
    "mage slayer",
    "poisoner",
    "polearm master",
    "sentinel",
    "shield master",
    "spell sniper",
    "telekinetic",
    "war caster",
    "speedy",
})

STYLES = frozenset({
    "archery",
    "defense",
    "dueling",
    "great weapon fighting",
    "two weapon fighting",
    "thrown weapon fighting",
    "interception",
    "protection",
    "blind fighting",
    "unarmed fighting",
})

DAMAGE_TYPES = frozenset({
    "acid", "cold", "fire", "force", "lightning", "necrotic", "poison",
    "psychic", "radiant", "thunder",
})

_POLE_NAMES = frozenset({"quarterstaff", "spear"})


def feat_key(name: str) -> str:
    text = name.lower().replace("-", " ").replace("'", "").replace("’", "").replace("‘", "")
    return _SPACE.sub(" ", text).strip()


def is_epic_boon(name: str) -> bool:
    return feat_key(name) in EPIC_BOONS


def is_quiet(name: str) -> bool:
    return feat_key(name) in QUIET or feat_key(name) in EPIC_BOONS


def gwf_weapon(option, shield: bool) -> bool:
    if option is None or not option.reach_feet:
        return False
    if option.two_handed:
        return True
    return bool(option.versatile) and not shield


def pole_weapon(option) -> bool:
    if option is None or not option.reach_feet:
        return False
    if option.name.lower() in _POLE_NAMES:
        return True
    return bool(option.heavy) and int(option.reach_feet) >= 10


def melee_reach(creature) -> int:
    reaches = [int(attack.reach_feet) for attack in creature.attacks if attack.reach_feet]
    return max(reaches) if reaches else 5


def _bumped(rolled: RollResult, rolls: tuple[int, ...]) -> RollResult:
    return RollResult(
        expression=rolled.expression,
        rolls=rolls,
        modifier=rolled.modifier,
        total=sum(rolls) + rolled.modifier,
        critical=rolled.critical,
    )


def _gwf(rolled: RollResult) -> tuple[RollResult, bool]:
    if not rolled.rolls:
        return rolled, False
    faces = tuple(3 if face in {1, 2} else face for face in rolled.rolls)
    if faces == rolled.rolls:
        return rolled, False
    return _bumped(rolled, faces), True


def _reroll_ones(rolled: RollResult, rng: random.Random) -> RollResult:
    if not rolled.rolls or 1 not in rolled.rolls:
        return rolled
    count, sides, _modifier = parse_dice(rolled.expression)
    if sides < 2:
        return rolled
    del count
    faces = tuple(rng.randint(1, sides) if face == 1 else face for face in rolled.rolls)
    return _bumped(rolled, faces)


def roll_weapon(
    expression: str,
    rng: random.Random,
    *,
    critical: bool,
    attacker,
    option,
) -> tuple[RollResult, list[str]]:
    """Weapon damage, including Great Weapon Fighting and Savage Attacker."""
    notes: list[str] = []
    shield = bool(attacker.traits.get("has_shield"))
    fighting = bool(attacker.traits.get("great_weapon_fighting")) and gwf_weapon(option, shield)

    def once() -> RollResult:
        rolled = roll_dice(expression, rng, critical=critical)
        if fighting:
            rolled, changed = _gwf(rolled)
            if changed and "Great Weapon Fighting treats 1 or 2 as 3." not in notes:
                notes.append("Great Weapon Fighting treats 1 or 2 as 3.")
        if option is not None and option.name == "Unarmed Strike" and attacker.traits.get("unarmed_reroll"):
            rolled = _reroll_ones(rolled, rng)
        return rolled

    first = once()
    if attacker.traits.get("savage_attacker") and not attacker.traits.get("savage_used") and first.rolls:
        second = once()
        attacker.traits["savage_used"] = True
        kept, other = (first, second) if sum(first.rolls) >= sum(second.rolls) else (second, first)
        notes.append(
            f"Savage Attacker keeps {sum(kept.rolls)} (other roll {sum(other.rolls)})."
        )
        return kept, notes
    return first, notes


def strip_modifier(expression: str, modifier: int) -> str:
    """Drop a positive ability modifier from a stored weapon expression."""
    if modifier <= 0:
        return expression
    count, sides, flat = parse_dice(expression)
    if flat < modifier:
        return expression
    kept = flat - modifier
    if sides == 0:
        return str(kept)
    dice = f"{count}d{sides}" if count else f"d{sides}"
    if count == 0:
        return str(kept)
    return dice if kept == 0 else f"{dice}{kept:+d}"


def extra_damage(actor, option, damage: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """The Light extra attack omits a positive ability modifier unless a feat keeps it."""
    if actor is None or option is None or not actor.traits.get("strip_mod"):
        return damage
    modifier = ability_mod(actor.abilities.get(option.ability or "strength", 10))
    if modifier <= 0:
        return damage
    if actor.traits.get("two_weapon_fighting"):
        return damage
    if actor.traits.get("crossbow_expert") and option.light and option.crossbow:
        return damage
    return [(kind, strip_modifier(expr, modifier)) for kind, expr in damage]


def spell_faces(attacker, damage_type: str, rolled: RollResult) -> RollResult:
    chosen = str((attacker.traits.get("elemental_adept") if attacker is not None else "") or "")
    if not chosen or chosen != damage_type or not rolled.rolls:
        return rolled
    faces = tuple(2 if face == 1 else face for face in rolled.rolls)
    if faces == rolled.rolls:
        return rolled
    return _bumped(rolled, faces)


def ignores_resistance(source, damage_type: str, *, spell: bool) -> bool:
    if source is None:
        return False
    if damage_type == "poison" and source.traits.get("poisoner"):
        return True
    chosen = str(source.traits.get("elemental_adept") or "")
    return bool(spell and chosen and chosen == damage_type)


def reroll_ones(rolled: RollResult, rng: random.Random) -> RollResult:
    """Healer and unarmed strikes: a 1 is rolled again and the new face stands."""
    return _reroll_ones(rolled, rng)


def reroll_one_low(rolled: RollResult, rng: random.Random) -> tuple[RollResult, bool]:
    """Piercer rerolls the first damage die that shows 1 or 2 and keeps the new face."""
    picked = None
    for index, face in enumerate(rolled.rolls):
        if face in {1, 2}:
            picked = index
            break
    if picked is None:
        return rolled, False
    _count, sides, _modifier = parse_dice(rolled.expression)
    if sides < 2:
        return rolled, False
    faces = list(rolled.rolls)
    faces[picked] = rng.randint(1, sides)
    return _bumped(rolled, tuple(faces)), True


def armor_reduction(target, damage_type: str, dealt: int) -> int:
    if dealt <= 0 or not target.traits.get("heavy_armor_master"):
        return 0
    if target.traits.get("armor") != "heavy":
        return 0
    if damage_type not in {"bludgeoning", "piercing", "slashing"}:
        return 0
    return min(dealt, int(target.traits.get("proficiency") or 2))
