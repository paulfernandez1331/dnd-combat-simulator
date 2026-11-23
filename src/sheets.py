"""Readable stat blocks for the creatures already in an encounter.

The panel is for inspection. It does not add a button for text the engine
cannot roll.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.bestiary import ABILITIES, ability_mod


@dataclass
class Sheet:
    title: str
    kind_label: str
    url: str | None
    lines: list[str]


def sheet_for(creature, catalog=None) -> Sheet:
    record = None
    if catalog is not None and creature.creature_index:
        record = catalog.monsters.get(creature.creature_index)
    kind = "SRD stat block" if record is not None else "Character sheet"
    lines = [_summary(creature, record)]
    lines.append(_abilities(creature))
    if creature.save_bonuses:
        lines.append("Saves: " + ", ".join(
            f"{name[:3].upper()} {bonus:+d}" for name, bonus in creature.save_bonuses.items()
        ) + ".")
    defenses = _defenses(creature)
    if defenses:
        lines.append(defenses)
    attacks = [_attack_line(attack) for attack in creature.attacks]
    if attacks:
        lines.append("Attacks: " + "; ".join(attacks) + ".")
    if creature.multiattack:
        parts = [f"{name} ×{count}" if count else name for name, count in creature.multiattack]
        lines.append("Multiattack: " + ", ".join(parts) + ".")
    if creature.spells:
        lines.append("Spells the fight can cast: " + ", ".join(spell.name for spell in creature.spells) + ".")
    if creature.uses_max:
        uses = [f"{name} {creature.uses_left.get(name, 0)}/{maximum}" for name, maximum in creature.uses_max.items()]
        lines.append("Uses: " + ", ".join(uses) + ".")
    for note in creature.traits.get("sheet_notes") or []:
        lines.append(str(note))
    unresolved = _unresolved(creature, record)
    if unresolved:
        lines.append("On the character, not a button: " + "; ".join(unresolved) + ".")
    url = creature.traits.get("ddb_url") or None
    return Sheet(title=creature.name, kind_label=kind, url=url, lines=lines)


def _summary(creature, record) -> str:
    bits = [
        creature.size,
        f"AC {creature.ac}",
        f"HP {creature.hp}/{creature.max_hp}",
        f"speed {creature.speed} ft",
    ]
    kind = creature.creature_type or (record.creature_type if record else "")
    if kind:
        bits.append(kind)
    if creature.challenge_rating is not None:
        bits.append(f"CR {creature.challenge_rating}")
    class_name = creature.traits.get("class_name")
    if class_name:
        subclass = creature.traits.get("subclass")
        level = creature.traits.get("level")
        label = str(class_name)
        if subclass:
            label += f" ({subclass})"
        if level:
            label += f" {level}"
        bits.append(label)
    species = creature.traits.get("species")
    if species:
        bits.append(str(species))
    return ", ".join(bits) + "."


def _abilities(creature) -> str:
    parts = []
    for name in ABILITIES:
        score = int(creature.abilities.get(name, 10))
        parts.append(f"{name[:3].upper()} {score} ({ability_mod(score):+d})")
    return "Abilities: " + ", ".join(parts) + "."


def _defenses(creature) -> str:
    chunks = []
    if creature.resistances:
        chunks.append("resists " + ", ".join(creature.resistances))
    if creature.immunities:
        chunks.append("immune to " + ", ".join(creature.immunities))
    if creature.vulnerabilities:
        chunks.append("vulnerable to " + ", ".join(creature.vulnerabilities))
    if not chunks:
        return ""
    return "Defenses: " + "; ".join(chunks) + "."


def _attack_line(attack) -> str:
    bits = [attack.name]
    if attack.attack_bonus is not None:
        bits.append(f"{attack.attack_bonus:+d}")
    damage = []
    for part in attack.damage or []:
        expr = part.get("expr") or part.get("dice") or ""
        kind = part.get("type") or ""
        if expr or kind:
            damage.append(f"{expr} {kind}".strip())
    if damage:
        bits.append(", ".join(damage))
    if attack.reach_feet:
        bits.append(f"reach {attack.reach_feet} ft")
    if attack.range_feet:
        bits.append(f"range {attack.range_feet} ft")
    return " ".join(bits)


def _unresolved(creature, record) -> list[str]:
    names: list[str] = []
    for name in creature.traits.get("excluded_features") or []:
        _add(names, str(name))
    for name in creature.traits.get("unresolved") or []:
        _add(names, str(name))
    if record is not None:
        for name in record.excluded_actions or []:
            _add(names, str(name))
    for spell in creature.spells:
        if spell.exclude_reason:
            _add(names, f"{spell.name}: {spell.exclude_reason}")
    return names


def _add(names: list[str], text: str) -> None:
    cleaned = " ".join(text.split())
    if cleaned and cleaned not in names:
        names.append(cleaned)
