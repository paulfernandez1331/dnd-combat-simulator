"""Attack rolls, saving throws, damage, and healing.

A natural 20 hits and doubles the dice, not the modifier. A natural 1 misses.
Successful saves deal half damage only when the record says so. Resistance
and vulnerability are applied after that, rounded down.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from src.bestiary import ABILITIES, ABILITY_FROM_SHORT, ability_mod
from src.combatant import Combatant
from src.conditions import (
    attack_penalty,
    auto_crit,
    condition_reasons,
    marks_of,
    note_damage,
    save_mode,
    save_penalty,
    sync,
)
from src.dice import RollResult, parse_dice, roll_dice
from src.feats import armor_reduction, extra_damage, ignores_resistance, reroll_one_low, roll_weapon, spell_faces


@dataclass
class Effect:
    damage: int = 0
    healing: int = 0
    lines: list[str] | None = None
    hit: bool = False

    def __post_init__(self) -> None:
        if self.lines is None:
            self.lines = []


def ability_name(value: str) -> str:
    lowered = value.lower()
    if lowered in ABILITIES:
        return lowered
    return ABILITY_FROM_SHORT[lowered[:3]]


def d20(rng: random.Random, mode: int) -> tuple[int, tuple[int, ...]]:
    if mode > 0:
        rolls = (rng.randint(1, 20), rng.randint(1, 20))
        return max(rolls), rolls
    if mode < 0:
        rolls = (rng.randint(1, 20), rng.randint(1, 20))
        return min(rolls), rolls
    roll = rng.randint(1, 20)
    return roll, (roll,)


def attack_reasons(attacker: Combatant, target: Combatant, *, melee: bool = False, spell: bool = False) -> tuple[list[str], list[str]]:
    """Advantage reasons, then disadvantage reasons. The two parameters stay for callers."""
    del melee, spell
    advantage, disadvantage = condition_reasons(attacker, target)
    if target.dodging:
        disadvantage.append("the target is dodging")
    if attacker.traits.get("sap"):
        disadvantage.append("the attacker is sapped")
    if attacker.traits.get("reckless") or target.traits.get("reckless"):
        advantage.append("Reckless Attack")
    if attacker.traits.get("vex_target") == target.id:
        advantage.append("Vex")
    if attacker.traits.get("pack_now"):
        advantage.append("Pack Tactics")
    if attacker.traits.get("slasher_disadvantage"):
        disadvantage.append("Slasher")
    if target.traits.get("crusher_from"):
        advantage.append("Crusher")
    if target.traits.get("protection_now"):
        disadvantage.append("Protection")
    if any(mark.get("next_attack") for mark in marks_of(target)):
        advantage.append("Stunning Strike")
    if attacker.traits.get("reaction_strike") and target.traits.get("speedy"):
        disadvantage.append("the target is speedy")
    return advantage, disadvantage


def attack_reason_text(attacker: Combatant, target: Combatant, *, melee: bool = False, spell: bool = False) -> str:
    """The sources that remain after advantage and disadvantage cancel."""
    advantage, disadvantage = attack_reasons(attacker, target, melee=melee, spell=spell)
    if advantage and disadvantage:
        return ""
    return "; ".join(advantage or disadvantage)


def attack_mode(attacker: Combatant, target: Combatant, *, melee: bool = False, spell: bool = False) -> int:
    """Advantage and disadvantage cancel, no matter how many sources stack."""
    advantage, disadvantage = attack_reasons(attacker, target, melee=melee, spell=spell)
    if advantage and disadvantage:
        return 0
    if advantage:
        return 1
    if disadvantage:
        return -1
    return 0


def _typed_amount(amount: int, damage_type: str, target: Combatant, *, ignore_resistance: bool = False) -> int:
    if damage_type in target.immunities:
        return 0
    if damage_type == "thunder" and target.traits.get("thunder_immune"):
        return 0
    if damage_type == "poison" and ("petrified" in target.conditions):
        return 0
    resisted = damage_type in target.resistances and not ignore_resistance
    vulnerable = damage_type in target.vulnerabilities
    if target.traits.get("poison_ward") and damage_type == "poison":
        resisted = True
    if "petrified" in target.conditions and damage_type != "poison":
        resisted = True
    if resisted and vulnerable:
        return amount
    if resisted:
        return amount // 2
    if vulnerable:
        return amount * 2
    return amount


def _drop_concentration(target: Combatant) -> str | None:
    if not target.concentrating_on:
        return None
    lost = target.concentrating_on
    target.concentrating_on = None
    target.ongoing_slot = None
    target.traits.pop("spiritual_weapon", None)
    target.traits.pop("zone", None)
    target.traits.pop("outlined_ids", None)
    if lost in {"hex", "hunters-mark"}:
        target.traits.pop("mark_id", None)
        target.traits.pop("mark_die", None)
        target.traits.pop("mark_type", None)
    return lost


def _end_rage(target: Combatant) -> None:
    if not target.traits.get("rage"):
        return
    for kind in list(target.traits.get("rage_added") or []):
        if kind in target.resistances:
            target.resistances.remove(kind)
    target.traits["rage"] = False
    target.traits["rage_added"] = []


def apply_damage(
    target: Combatant,
    parts: list[tuple[str, int]],
    rng: random.Random,
    *,
    source: Combatant | None = None,
    attack_roll: bool = False,
    echo: bool = False,
    spell: bool = False,
) -> Effect:
    effect = Effect()
    totals: dict[str, int] = {}
    raw_totals: dict[str, int] = {}
    order: list[str] = []
    reduced = 0
    for damage_type, amount in parts:
        amount = max(0, amount)
        ignore = ignores_resistance(source, damage_type, spell=spell)
        dealt = _typed_amount(amount, damage_type, target, ignore_resistance=ignore)
        cut = armor_reduction(target, damage_type, dealt)
        if cut:
            dealt -= cut
            reduced += cut
        effect.damage += dealt
        plain = (
            cut > 0
            and damage_type not in target.resistances
            and damage_type not in target.vulnerabilities
            and damage_type not in target.immunities
        )
        if damage_type not in totals:
            order.append(damage_type)
            totals[damage_type] = 0
            raw_totals[damage_type] = 0
        totals[damage_type] += dealt
        raw_totals[damage_type] += dealt if plain else amount
    absorbed = min(target.temp_hp, effect.damage)
    if absorbed:
        target.temp_hp -= absorbed
        effect.lines.append(f"{target.name} loses {absorbed} temporary hit points ({target.temp_hp} left).")
    target.hp = max(0, target.hp - (effect.damage - absorbed))
    phrases = [_damage_phrase(kind, raw_totals[kind], totals[kind]) for kind in order if raw_totals[kind] or totals[kind]]
    if phrases:
        taken = _and(phrases)
        if target.hp > 0:
            effect.lines.append(f"{target.name} takes {taken} and has {target.hp} hit points.")
        else:
            effect.lines.append(f"{target.name} takes {taken}.")
    if reduced:
        effect.lines.append(f"Heavy Armor Master reduces the damage by {reduced}.")
    if target.hp == 0:
        target.conditions["unconscious"] = 99
        target.conditions["incapacitated"] = 99
        target.conditions["prone"] = 99
        _end_rage(target)
        lost = _drop_concentration(target)
        effect.lines.append(f"{target.name} drops to 0 hit points.")
        if lost:
            effect.lines.append(f"{target.name} loses concentration on {lost}.")
        return effect
    if effect.damage > 0:
        effect.lines.extend(note_damage(target, source, effect.damage))
    if effect.damage > 0 and target.concentrating_on:
        dc = max(10, effect.damage // 2)
        bonus = target.save_bonuses.get("constitution", ability_mod(target.abilities.get("constitution", 10)))
        mode = 0
        if target.traits.get("war_caster"):
            mode += 1
        if source is not None and source.traits.get("mage_slayer"):
            mode -= 1
        roll, shown = d20(rng, mode)
        penalty = save_penalty(target)
        total = roll + bonus - penalty
        if mode == 0 and not penalty:
            roll_text = f"{shown[0]}+{bonus}={total}"
        elif mode == 0:
            roll_text = f"{shown[0]}+{bonus}-{penalty}={total}"
        else:
            word = "advantage" if mode > 0 else "disadvantage"
            faces = " and ".join(str(item) for item in shown)
            extra = f"-{penalty}" if penalty else ""
            roll_text = f"{word} {faces}, keeps {roll}, +{bonus}{extra}={total}"
        spell_name = target.concentrating_on
        if total < dc:
            _drop_concentration(target)
            effect.lines.append(
                f"{target.name} fails the concentration save ({roll_text} vs DC {dc}) and loses {spell_name}."
            )
        else:
            effect.lines.append(
                f"{target.name} holds concentration ({roll_text} vs DC {dc})."
            )
    return effect


def apply_heal(target: Combatant, amount: int) -> Effect:
    gained = max(0, min(amount, target.max_hp - target.hp))
    target.hp += gained
    if target.hp > 0:
        from src.conditions import marks_of

        if not any(mark.get("condition") == "unconscious" for mark in marks_of(target)):
            target.conditions.pop("unconscious", None)
            target.conditions.pop("incapacitated", None)
        sync(target)
    return Effect(healing=gained, lines=[f"{target.name} regains {gained} hit points (now {target.hp})."])


def _and(items: list[str]) -> str:
    if len(items) <= 1:
        return items[0] if items else ""
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + ", and " + items[-1]


def _damage_phrase(kind: str, raw: int, dealt: int) -> str:
    if dealt == 0 and raw > 0:
        return f"0 {kind} (immune, from {raw})"
    if dealt < raw:
        return f"{dealt} {kind} (resisted, from {raw})"
    if dealt > raw:
        return f"{dealt} {kind} (vulnerable, from {raw})"
    return f"{dealt} {kind}"


def _proficiency(actor: Combatant) -> int:
    return int(actor.traits.get("proficiency") or 3)


def armor_class(target: Combatant) -> int:
    """Armor class, including a Shield spell that is already up."""
    return int(target.ac) + int(target.traits.get("shield_ac") or 0)


def opposed_ac(target: Combatant, *, melee: bool, ac_bonus: int = 0) -> int:
    """AC for this attack. Defensive Duelist adds only against a melee attack."""
    ac = armor_class(target) + int(ac_bonus)
    if melee:
        ac += int(target.traits.get("duelist_ac") or 0)
    return ac


def _attack_line(
    mode: int,
    shown: tuple[int, ...],
    roll: int,
    bonus: int,
    total: int,
    ac: int,
    penalty: int = 0,
    reason: str = "",
) -> str:
    if mode > 0:
        kept = f"advantage {' and '.join(str(item) for item in shown)}, keeps {roll}"
    elif mode < 0:
        kept = f"disadvantage {' and '.join(str(item) for item in shown)}, keeps {roll}"
    else:
        kept = str(roll)
    if reason and mode != 0:
        kept = f"{kept} ({reason})"
    if penalty:
        return f"Attack  {kept} + {bonus} - {penalty} = {total} vs AC {ac}."
    return f"Attack  {kept} + {bonus} = {total} vs AC {ac}."


def _weapon_clause(rolled, damage_type: str) -> str:
    faces = " + ".join(str(face) for face in rolled.rolls)
    if rolled.critical and rolled.rolls:
        flat = f", modifier {rolled.modifier:+d} not doubled" if rolled.modifier else ""
        return f"{rolled.expression} critical: dice {faces}{flat}, total {rolled.total} {damage_type}."
    if rolled.rolls and rolled.modifier:
        return f"{rolled.expression}: dice {faces}, modifier {rolled.modifier:+d}, total {rolled.total} {damage_type}."
    if rolled.rolls:
        return f"{rolled.expression}: dice {faces}, total {rolled.total} {damage_type}."
    return f"{rolled.total} {damage_type}."


def _fury_note(die: str, rolled, flat: int, kind: str) -> str:
    faces = " + ".join(str(face) for face in rolled.rolls) if rolled.rolls else "0"
    total = rolled.total + flat
    if rolled.critical:
        flat_text = f", modifier +{flat} not doubled" if flat else ""
        return f"Divine Fury critical: {die} dice {faces}{flat_text}, total {total} {kind}."
    if flat:
        return f"Divine Fury: {die} dice {faces}, modifier +{flat}, total {total} {kind}."
    return f"Divine Fury: {die} dice {faces}, total {total} {kind}."


def _block(rows: list[str]) -> list[str]:
    return ["\n".join(row for row in rows if row)]


def _consume_roll_flags(attacker: Combatant, target: Combatant) -> None:
    attacker.traits.pop("sap", None)
    attacker.traits.pop("hidden", None)
    attacker.traits.pop("steady", None)
    if attacker.traits.get("vex_target") == target.id:
        attacker.traits.pop("vex_target", None)
    for mark in marks_of(target):
        if mark.get("next_attack"):
            mark["next_attack"] = False


def _rider_parts(attacker, target, option, critical, rng):
    parts: list[tuple[str, int]] = []
    notes: list[str] = []
    dtype = "slashing"
    if option is not None and option.damage:
        dtype = str(option.damage[0].get("type") or dtype)
    if attacker.traits.get("rage") and option is not None and option.ability == "strength":
        bonus = int(attacker.traits.get("rage_bonus") or 0)
        if bonus:
            parts.append((dtype, bonus))
            notes.append(f"Rage +{bonus} {dtype}." + (" Flat bonus, not doubled." if critical else ""))
    if (
        attacker.traits.get("gwm")
        and option is not None
        and option.heavy
        and not attacker.traits.get("hew_strike")
    ):
        bonus = _proficiency(attacker)
        if bonus:
            parts.append((dtype, bonus))
            notes.append(f"Great Weapon Master +{bonus} {dtype}." + (" Flat bonus, not doubled." if critical else ""))
    fury = attacker.traits.get("divine_fury")
    if fury and attacker.traits.get("rage") and option is not None and not attacker.traits.get("fury_used"):
        attacker.traits["fury_used"] = True
        die = str(fury.get("die") or "1d6")
        flat = int(fury.get("flat") or 0)
        kind = str(fury.get("type") or "radiant")
        rolled = roll_dice(die, rng, critical=critical)
        total = rolled.total + flat
        parts.append((kind, total))
        notes.append(_fury_note(die, rolled, flat, kind))
    return parts, notes


def resolve_attack(
    attacker: Combatant,
    target: Combatant,
    *,
    bonus: int,
    damage: list[tuple[str, str]],
    rng: random.Random,
    melee: bool,
    label: str,
    option=None,
    allies=None,
    spell: bool = False,
    fixed_roll: tuple[int, tuple[int, ...]] | None = None,
    penalty: int = 0,
    ac_bonus: int = 0,
    before_damage=None,
    mode_override: int | None = None,
    reason: str | None = None,
) -> Effect:
    if mode_override is None:
        mode = attack_mode(attacker, target, melee=melee, spell=spell)
        if reason is None:
            reason = attack_reason_text(attacker, target, melee=melee, spell=spell)
    else:
        mode = int(mode_override)
        if reason is None:
            reason = ""
    if fixed_roll is None:
        roll, shown = d20(rng, mode)
    else:
        roll, shown = fixed_roll
    total = roll + bonus - int(penalty)
    crit_min = int(getattr(attacker, "crit_min", 20) or 20)
    critical = roll >= crit_min and roll != 1
    ac = opposed_ac(target, melee=melee, ac_bonus=int(ac_bonus))
    hit = roll != 1 and (critical or total >= ac)
    _consume_roll_flags(attacker, target)
    math_line = _attack_line(mode, shown, roll, bonus, total, ac, int(penalty), reason or "")
    if not hit:
        rows = [f"Miss  {attacker.name} uses {label} on {target.name}.", math_line]
        damage_done = 0
        if option is not None and option.mastery == "graze":
            ability = option.ability or "strength"
            grazed = max(0, ability_mod(attacker.abilities.get(ability, 10)))
            if grazed:
                dtype = option.damage[0]["type"] if option.damage else "slashing"
                chip = apply_damage(target, [(dtype, grazed)], rng, source=attacker)
                damage_done = chip.damage
                rows.append(f"Graze deals {grazed} {dtype}.")
                rows.extend(chip.lines or [])
        return Effect(lines=_block(rows), damage=damage_done)
    if auto_crit(attacker, target):
        critical = True
    parts: list[tuple[str, int]] = []
    dice_bits = []
    weapon_notes: list[str] = []
    expressions = extra_damage(attacker, option, list(damage)) if not spell else list(damage)
    piercing_done = False
    for damage_type, expression in expressions:
        if spell:
            rolled = roll_dice(expression, rng, critical=critical)
            adjusted = spell_faces(attacker, damage_type, rolled)
            if adjusted.rolls != rolled.rolls:
                weapon_notes.append(f"Elemental Adept turns a 1 into a 2 on {damage_type}.")
            rolled = adjusted
        else:
            rolled, notes = roll_weapon(expression, rng, critical=critical, attacker=attacker, option=option)
            weapon_notes.extend(notes)
            if attacker.traits.get("piercer") and damage_type == "piercing" and not piercing_done:
                piercing_done = True
                if not attacker.traits.get("piercer_used"):
                    rolled, changed = reroll_one_low(rolled, rng)
                    if changed:
                        attacker.traits["piercer_used"] = True
                        weapon_notes.append("Piercer rerolls one damage die.")
                if critical:
                    _count, sides, _flat = parse_dice(expression)
                    if sides:
                        face = rng.randint(1, sides)
                        rolled = RollResult(
                            expression=rolled.expression,
                            rolls=rolled.rolls + (face,),
                            modifier=rolled.modifier,
                            total=rolled.total + face,
                            critical=rolled.critical,
                        )
                        weapon_notes.append(f"Piercer adds a {face}.")
        parts.append((damage_type, rolled.total))
        dice_bits.append(_weapon_clause(rolled, damage_type))
    dtype = expressions[0][0] if expressions else "slashing"
    if (
        not spell
        and melee
        and attacker.traits.get("dueling")
        and attacker.traits.get("one_melee")
        and option is not None
        and not option.two_handed
        and option.name != "Pole strike"
    ):
        parts.append((dtype, 2))
        weapon_notes.append("Dueling +2.")
    if (
        not spell
        and not melee
        and attacker.traits.get("thrown_weapon_fighting")
        and option is not None
        and option.thrown
    ):
        parts.append((dtype, 2))
        weapon_notes.append("Thrown Weapon Fighting +2.")
    if (
        not spell
        and melee
        and attacker.traits.get("charger")
        and not attacker.traits.get("charger_used")
        and int(attacker.traits.get("moved_feet") or 0) >= 10
    ):
        attacker.traits["charger_used"] = True
        bonus_roll = roll_dice("1d8", rng)
        face = bonus_roll.rolls[0] if bonus_roll.rolls else 0
        parts.append((dtype, bonus_roll.total))
        weapon_notes.append(f"Charger 1d8: dice {face}, total {bonus_roll.total}.")
    dice_bits.extend(weapon_notes)
    riders, rider_notes = _rider_parts(attacker, target, option, critical, rng)
    if before_damage is not None:
        held = before_damage(
            parts, riders, dice_bits, rider_notes, math_line, "Critical hit" if critical else "Hit", critical
        )
        if held == "pause":
            return Effect(hit=True)
    parts.extend(riders)
    return complete_hit(
        attacker,
        target,
        parts=parts,
        dice_bits=dice_bits,
        rider_notes=rider_notes,
        math_line=math_line,
        title="Critical hit" if critical else "Hit",
        critical=critical,
        melee=melee,
        option=option,
        rng=rng,
        label=label,
        spell=spell,
    )


def complete_hit(
    attacker: Combatant,
    target: Combatant,
    *,
    parts: list[tuple[str, int]],
    dice_bits: list[str],
    rider_notes: list[str],
    math_line: str,
    title: str,
    critical: bool,
    melee: bool,
    option,
    rng: random.Random,
    label: str,
    spell: bool = False,
) -> Effect:
    effect = apply_damage(target, parts, rng, source=attacker, attack_roll=True, spell=spell)
    effect.hit = True
    if (
        option is not None
        and melee
        and attacker.traits.get("hew")
        and not attacker.traits.get("hew_strike")
        and (critical or target.hp <= 0)
    ):
        attacker.traits["hew_ready"] = option.name
    if option is not None and option.mastery == "sap":
        target.traits["sap"] = True
        rider_notes.append(f"{target.name} is sapped.")
    if option is not None and option.mastery == "slow":
        target.traits["slow"] = 10
        target.traits["slow_from"] = attacker.id
        rider_notes.append(f"{target.name} is slowed by 10 feet.")
    if option is not None and option.mastery == "vex":
        attacker.traits["vex_target"] = target.id
        rider_notes.append(f"{attacker.name} has advantage on the next attack against {target.name}.")
    if option is not None and option.damage and not spell:
        kind = str(option.damage[0].get("type") or "")
        if attacker.traits.get("slasher") and kind == "slashing":
            if not attacker.traits.get("slasher_used"):
                attacker.traits["slasher_used"] = True
                target.traits["slow"] = 10
                target.traits["slow_from"] = attacker.id
                rider_notes.append(f"{target.name} is slowed by 10 feet.")
            if critical:
                target.traits["slasher_disadvantage"] = attacker.id
                rider_notes.append(f"{target.name} has disadvantage on attack rolls.")
        if attacker.traits.get("crusher") and kind == "bludgeoning" and critical:
            target.traits["crusher_from"] = attacker.id
            rider_notes.append(f"Attacks against {target.name} have advantage until {attacker.name}'s next turn.")
    rows = [f"{title}  {attacker.name} uses {label} on {target.name}."]
    if critical and "Dice are doubled" not in math_line:
        math_line += " Dice are doubled. Flat bonuses are not."
    rows.append(math_line)
    rows.extend(dice_bits)
    rows.extend(note for note in rider_notes if note)
    rows.extend(effect.lines or [])
    effect.lines = _block(rows)
    return effect


def resolve_save(
    source: Combatant,
    target: Combatant,
    *,
    dc: int,
    ability: str,
    success: str,
    damage: list[tuple[str, str]],
    rng: random.Random,
    label: str,
    advantage: bool = False,
    echo: bool = False,
    fixed_roll: tuple[int, tuple[int, ...]] | None = None,
    forced_success: bool = False,
    negate_damage: bool = False,
) -> Effect:
    full = ability_name(ability)
    mode = save_mode(target, full)
    auto_failed = mode is None
    if auto_failed:
        mode = 0
    elif advantage:
        mode = min(1, mode + 1)
    if fixed_roll is None and not auto_failed:
        roll, shown = d20(rng, mode)
    elif fixed_roll is None:
        roll, shown = 0, (0,)
    else:
        roll, shown = fixed_roll
    bonus = target.save_bonuses.get(full, ability_mod(target.abilities.get(full, 10)))
    penalty = save_penalty(target)
    total = 0 if auto_failed else roll + bonus - penalty
    saved = (not auto_failed) and (bool(forced_success) or total >= dc)
    parts: list[tuple[str, int]] = []
    dice_bits = []
    # A success that negates the damage does not roll those dice.
    damage = [] if saved and success != "half" else damage
    for damage_type, expression in damage:
        rolled = spell_faces(source, damage_type, roll_dice(expression, rng, critical=False))
        amount = rolled.total
        if negate_damage or (saved and success != "half"):
            amount = 0
        elif saved and success == "half":
            amount //= 2
        parts.append((damage_type, amount))
        clause = _weapon_clause(rolled, damage_type)
        if negate_damage:
            clause = clause[:-1] + ", reduced to 0."
        elif saved and success == "half":
            clause = clause[:-1] + f", halved to {amount}."
        elif saved:
            clause = clause[:-1] + ", no damage on a successful save."
        dice_bits.append(clause)
    effect = apply_damage(target, parts, rng, source=source, echo=echo, spell=True) if any(amount > 0 for _, amount in parts) else Effect()
    if auto_failed:
        roll_line = f"Roll  {target.name} automatically fails the {full.title()} save."
    elif penalty:
        roll_line = f"Roll  {roll} + {bonus} - {penalty} = {total}."
    elif len(shown) > 1:
        roll_line = f"Roll  {' and '.join(str(item) for item in shown)}, keeps {roll}, +{bonus} = {total}."
    else:
        roll_line = f"Roll  {roll} + {bonus} = {total}."
    lead = "Saved" if saved else "Failed save"
    extra = []
    if forced_success:
        extra.append(f"{target.name} uses Guarded Mind and succeeds instead.")
    if negate_damage:
        extra.append(f"{target.name} interposes a shield. The damage is 0.")
    effect.lines = _block([
        f"{lead}  {target.name} vs DC {dc} {full.title()} ({source.name}'s {label}).",
        roll_line,
        *dice_bits,
        *extra,
        *(effect.lines or []),
    ])
    return effect


def resolve_auto(
    source: Combatant,
    target: Combatant,
    *,
    damage: list[tuple[str, str]],
    rng: random.Random,
    label: str,
) -> Effect:
    parts = []
    dice_bits = []
    for damage_type, expression in damage:
        rolled = spell_faces(source, damage_type, roll_dice(expression, rng))
        parts.append((damage_type, rolled.total))
        dice_bits.append(_weapon_clause(rolled, damage_type))
    effect = apply_damage(target, parts, rng, source=source, spell=True)
    effect.lines = _block([
        f"Hit  {source.name}'s {label} hits {target.name} automatically.",
        *dice_bits,
        *(effect.lines or []),
    ])
    return effect


def combine(effects: list[Effect]) -> Effect:
    total = Effect(lines=[])
    for effect in effects:
        total.damage += effect.damage
        total.healing += effect.healing
        total.hit = total.hit or effect.hit
        total.lines.extend(effect.lines or [])
    return total
