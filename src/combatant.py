"""Creatures on the grid. Records are copied off the API so a fight can rerun offline."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.bestiary import (
    ABILITIES,
    AttackOption,
    MonsterRecord,
    ReactionRecord,
    SpellRecord,
    ability_mod,
)
from src.grid import footprint, space_gap, span_for


@dataclass
class Combatant:
    id: str
    name: str
    side: str
    ac: int
    max_hp: int
    hp: int
    speed: int
    abilities: dict[str, int]
    save_bonuses: dict[str, int]
    position: tuple[int, int]
    attacks: list[AttackOption] = field(default_factory=list)
    multiattack: list[tuple[str, int]] = field(default_factory=list)
    spells: list[SpellRecord] = field(default_factory=list)
    spell_attack_bonus: int = 0
    spell_save_dc: int = 0
    spellcasting_mod: int = 0
    caster_level: int = 1
    slot_max: dict[int, int] = field(default_factory=dict)
    slots: dict[int, int] = field(default_factory=dict)
    resistances: list[str] = field(default_factory=list)
    immunities: list[str] = field(default_factory=list)
    vulnerabilities: list[str] = field(default_factory=list)
    conditions: dict[str, int] = field(default_factory=dict)
    concentrating_on: str | None = None
    ongoing_slot: int | None = None
    dodging: bool = False
    recharging: list[str] = field(default_factory=list)
    uses_left: dict[str, int] = field(default_factory=dict)
    uses_max: dict[str, int] = field(default_factory=dict)
    challenge_rating: float | None = None
    creature_index: str | None = None
    movement_left: int | None = None
    action_available: bool = True
    bonus_available: bool = True
    attacks_per_action: int = 1
    attacks_made: int = 0
    reaction_available: bool = True
    temp_hp: int = 0
    crit_min: int = 20
    creature_type: str = ""
    size: str = "Medium"
    traits: dict = field(default_factory=dict)
    reactions: list[ReactionRecord] = field(default_factory=list)

    @property
    def span(self) -> int:
        return span_for(self.size)

    def squares(self) -> set[tuple[int, int]]:
        return footprint(tuple(self.position), self.span)

    def gap_to(self, other: Combatant) -> int:
        return space_gap(tuple(self.position), self.span, tuple(other.position), other.span)

    def gap_to_square(self, square: tuple[int, int]) -> int:
        return space_gap(tuple(self.position), self.span, (int(square[0]), int(square[1])), 1)

    def in_cells(self, cells) -> bool:
        return bool(self.squares() & set(cells))

    def attack_named(self, name: str | None) -> AttackOption | None:
        for attack in self.attacks:
            if attack.name == name:
                return attack
        return None

    def spell_named(self, index: str | None) -> SpellRecord | None:
        for spell in self.spells:
            if spell.index == index:
                return spell
        return None

    def restore_body(self) -> None:
        self.hp = self.max_hp
        self.slots = {int(level): count for level, count in self.slot_max.items()}
        self.uses_left = dict(self.uses_max)
        self.conditions = {}
        self.concentrating_on = None
        self.ongoing_slot = None
        self.dodging = False
        self.recharging = []
        self.movement_left = None
        self.action_available = True
        self.bonus_available = True
        self.attacks_made = 0
        self.reaction_available = True
        self.temp_hp = 0
        for kind in list(self.traits.get("rage_added") or []):
            if kind in self.resistances:
                self.resistances.remove(kind)
        for key in (
            "rage", "rage_added", "reckless", "hidden", "smite_target", "light_used", "nick_used",
            "sneak_used", "colossus_used", "sap", "vex_target", "slow", "slow_from", "mark_id",
            "mark_die", "mark_type", "steady", "twin", "sacred", "fright_seen", "cleave_used",
            "poison_next", "stun_next", "surge_blocks_magic", "banked_action", "surged", "shift_feet", "inspiration",
            "leveled_cast", "stunned_by", "innate", "fury_used", "hew_ready", "hew_strike",
            "shield_ac", "duelist_ac", "savage_used", "charger_used", "piercer_used",
            "slasher_used", "crusher_used", "tavern_push_used", "shield_bash_used",
            "pole_ready", "moved_feet", "on_attack_action", "strip_mod", "reaction_strike",
            "disengaged", "flurry_left", "monk_attacked", "flurry_swing", "stunning_used",
            "swing_budget", "swing_tag", "metabolism_closed",
            "protected_by", "crusher_from", "slasher_disadvantage",
            "spiritual_weapon", "zone", "outlined_ids", "outline_source",
            "restrained_by", "frightened_by", "marks", "lingers", "banished",
            "removed", "dance", "speed_zero", "lock_action", "heroism",
            "heroism_temp", "inside_blind", "inside_deaf", "inside_silence",
            "thunder_immune", "see_invisible", "ward_evil", "freedom", "calm",
            "poison_ward", "no_invisible", "move_path", "last_hit",
        ):
            self.traits.pop(key, None)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "side": self.side,
            "ac": self.ac,
            "max_hp": self.max_hp,
            "hp": self.hp,
            "speed": self.speed,
            "abilities": dict(self.abilities),
            "save_bonuses": dict(self.save_bonuses),
            "position": [self.position[0], self.position[1]],
            "attacks": [attack.to_dict() for attack in self.attacks],
            "multiattack": [list(part) for part in self.multiattack],
            "spells": [spell.to_dict() for spell in self.spells],
            "spell_attack_bonus": self.spell_attack_bonus,
            "spell_save_dc": self.spell_save_dc,
            "spellcasting_mod": self.spellcasting_mod,
            "caster_level": self.caster_level,
            "slot_max": {str(level): count for level, count in self.slot_max.items()},
            "slots": {str(level): count for level, count in self.slots.items()},
            "resistances": list(self.resistances),
            "immunities": list(self.immunities),
            "vulnerabilities": list(self.vulnerabilities),
            "conditions": dict(self.conditions),
            "concentrating_on": self.concentrating_on,
            "ongoing_slot": self.ongoing_slot,
            "dodging": self.dodging,
            "recharging": list(self.recharging),
            "uses_left": dict(self.uses_left),
            "uses_max": dict(self.uses_max),
            "challenge_rating": self.challenge_rating,
            "creature_index": self.creature_index,
            "movement_left": self.movement_left,
            "action_available": self.action_available,
            "bonus_available": self.bonus_available,
            "attacks_per_action": self.attacks_per_action,
            "attacks_made": self.attacks_made,
            "reaction_available": self.reaction_available,
            "temp_hp": self.temp_hp,
            "crit_min": self.crit_min,
            "creature_type": self.creature_type,
            "size": self.size,
            "traits": dict(self.traits),
            "reactions": [item.to_dict() for item in self.reactions],
        }

    @classmethod
    def from_dict(cls, data: dict) -> Combatant:
        slots = {int(level): count for level, count in (data.get("slots") or {}).items()}
        slot_max = {int(level): count for level, count in (data.get("slot_max") or {}).items()}
        return cls(
            id=data["id"],
            name=data["name"],
            side=data["side"],
            ac=int(data["ac"]),
            max_hp=int(data["max_hp"]),
            hp=int(data["hp"]),
            speed=int(data["speed"]),
            abilities={key: int(value) for key, value in data["abilities"].items()},
            save_bonuses={key: int(value) for key, value in data["save_bonuses"].items()},
            position=(int(data["position"][0]), int(data["position"][1])),
            attacks=[AttackOption.from_dict(item) for item in data.get("attacks") or []],
            multiattack=[(str(name), int(count)) for name, count in data.get("multiattack") or []],
            spells=[SpellRecord.from_dict(item) for item in data.get("spells") or []],
            spell_attack_bonus=int(data.get("spell_attack_bonus") or 0),
            spell_save_dc=int(data.get("spell_save_dc") or 0),
            spellcasting_mod=int(data.get("spellcasting_mod") or 0),
            caster_level=int(data.get("caster_level") or 1),
            slot_max=slot_max,
            slots=slots or dict(slot_max),
            resistances=list(data.get("resistances") or []),
            immunities=list(data.get("immunities") or []),
            vulnerabilities=list(data.get("vulnerabilities") or []),
            conditions={key: int(value) for key, value in (data.get("conditions") or {}).items()},
            concentrating_on=data.get("concentrating_on"),
            ongoing_slot=data.get("ongoing_slot"),
            dodging=bool(data.get("dodging")),
            recharging=list(data.get("recharging") or []),
            uses_left={key: int(value) for key, value in (data.get("uses_left") or {}).items()},
            uses_max={key: int(value) for key, value in (data.get("uses_max") or {}).items()},
            challenge_rating=data.get("challenge_rating"),
            creature_index=data.get("creature_index"),
            movement_left=None if data.get("movement_left") is None else int(data["movement_left"]),
            action_available=bool(data.get("action_available", True)),
            bonus_available=bool(data.get("bonus_available", True)),
            attacks_per_action=int(data.get("attacks_per_action") or 1),
            attacks_made=int(data.get("attacks_made") or 0),
            reaction_available=bool(data.get("reaction_available", True)),
            temp_hp=int(data.get("temp_hp") or 0),
            crit_min=int(data.get("crit_min") or 20),
            creature_type=str(data.get("creature_type") or ""),
            size=str(data.get("size") or "Medium"),
            traits=dict(data.get("traits") or {}),
            reactions=[ReactionRecord.from_dict(item) for item in data.get("reactions") or []],
        )


def blank_abilities(scores: dict[str, int] | None = None) -> dict[str, int]:
    base = {ability: 10 for ability in ABILITIES}
    for key, value in (scores or {}).items():
        base[key] = int(value)
    return base


def save_bonuses_for(abilities: dict[str, int], proficient: list[str], proficiency: int) -> dict[str, int]:
    bonuses = {ability: ability_mod(score) for ability, score in abilities.items()}
    for ability in proficient:
        bonuses[ability] = bonuses.get(ability, 0) + proficiency
    return bonuses


def from_monster(record: MonsterRecord, *, creature_id: str, name: str, side: str, position: tuple[int, int]) -> Combatant:
    uses = {attack.name: attack.uses for attack in record.attacks if attack.uses is not None}
    for reaction in record.reactions:
        if reaction.pool and reaction.uses is not None:
            uses.setdefault(reaction.pool, int(reaction.uses))
    unresolved = []
    for reaction in record.reactions:
        if reaction.effect == "unresolved" and reaction.name not in unresolved:
            unresolved.append(reaction.name)
    traits = {}
    if unresolved:
        traits["excluded_features"] = unresolved
    if record.condition_immunities:
        traits["condition_immunities"] = list(record.condition_immunities)
    for key in ("blindsight", "truesight", "tremorsense"):
        if record.senses.get(key):
            traits[key] = int(record.senses[key])
    if record.auras:
        traits["auras"] = list(record.auras)
    if record.starts_invisible:
        traits["starts_invisible"] = True
    if record.pack_tactics:
        traits["pack_tactics"] = True
    return Combatant(
        id=creature_id,
        name=name,
        side=side,
        ac=record.ac,
        max_hp=record.hp,
        hp=record.hp,
        speed=record.speed,
        abilities=dict(record.abilities),
        save_bonuses=dict(record.save_bonuses),
        position=position,
        attacks=list(record.attacks),
        multiattack=list(record.multiattack),
        resistances=list(record.resistances),
        immunities=list(record.immunities),
        vulnerabilities=list(record.vulnerabilities),
        uses_left=dict(uses),
        uses_max=dict(uses),
        challenge_rating=record.challenge_rating,
        creature_index=record.index,
        creature_type=record.creature_type or "",
        size=record.size,
        spell_save_dc=int(record.spell_save_dc or 0),
        reactions=list(record.reactions),
        traits=traits,
    )
