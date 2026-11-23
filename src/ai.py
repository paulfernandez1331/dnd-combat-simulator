"""Enemy action choice. Batch fights and live fights share this path."""

from __future__ import annotations

import json
import os
import re
from typing import Literal

from pydantic import BaseModel, create_model
from toon_format import encode

from contextlib import contextmanager

from src.actions import (
    EMANATION_INDEXES,
    _afflict_pool,
    _reachable,
    adjacent_foe,
    _token_target,
    hostile_zones,
    movement_budget,
    path_enters,
    space_between,
    spell_parts,
    spell_range_feet,
    targets_for,
    turn_speed,
    using_melee,
    weapon_reaches,
)
from src.bestiary import spell_heal_expression
from src.conditions import (
    ACTION_DENY,
    BANISH_TYPES,
    EVIL_TYPES,
    affliction,
    condition_labels,
    creature_types,
    exhaustion_level,
    holds_concentration,
    immune_to,
    is_elf,
    is_humanoid,
    is_incapacitated,
    marks_of,
    rider_skipped,
    size_rank,
)
from src.dice import average, critical_average, parse_dice, substitute_mod
from src.rules import _typed_amount, attack_mode, attack_penalty, auto_crit, save_mode, save_penalty
from src.zones import effect_parts, zone_spec
from src.grid import chebyshev, footprint, space_gap, step_path


ACTION_SYSTEM = (
    "The user message is TOON. "
    "expected is pressure removed, in hit points. "
    "A hit's number is its damage plus the chance it drops the target. "
    "A condition's number is the attacks it prevents and the extra damage allies then land, times the save chance. "
    "Compare those numbers directly. "
    "Paralyzed, stunned, unconscious, petrified, and a forced order remove the target's action. "
    "Charm, frighten, prone, blind, poison, deafen, and grapple leave the action in place. "
    "Finish a creature this action can drop. "
    "Lock a healthy creature whose coming attacks outweigh the hit. "
    "Blinded beats Deafened. "
    "Leave a slept or hypnotized creature alone while another enemy is still fighting. "
    "Keep concentration on a lock that still prevents more than the replacement spell. "
    "When you are prone and Stand is listed, stand. The attack is the next choice. "
    "When an order is the only action, take it. "
    "fresh, opportunity, and tick are movement numbers. "
    "Choose exactly one listed action id. You may pick a different id than pick."
)

DECISION_MODEL = "grok-4.20-non-reasoning"
REPORT_MODEL = "grok-4.7"

REPORT_SYSTEM = (
    "You write an encounter report for a data-science class. "
    "Use only numbers present in the user message. "
    "Do not invent a statistic or a winner group that is absent. "
    "Write four or five short paragraphs about how these fights tended to go. "
    "Open with the shape of the encounter: who won, how long it lasted, and who was still standing. "
    "Use the middle for why: pace, first turn, first knockout, concentration, how living turns were spent, "
    "and whether the dice ran hot or cold against the fair marks of 10.5 and 50. "
    "Close with whether the encounter looks easy, even, or deadly, and what would have had to differ for the other result. "
    "Cite a number only when it carries the point. Do not restate the brief and do not open with a list of means."
)


class LLMError(RuntimeError):
    """The model cannot be called."""


class LLMClient:
    """OpenAI-compatible responses client. Action calls and the report use different models."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        *,
        report: bool = False,
    ):
        key = api_key or os.getenv("XAI_API_KEY")
        if not key:
            raise LLMError("Set XAI_API_KEY in .env. Heuristic enemy control still works without a key.")
        from openai import OpenAI

        self.report = report
        if model:
            self.model = model
        elif report:
            self.model = os.getenv("LLM_REPORT_MODEL", REPORT_MODEL)
        else:
            self.model = os.getenv("LLM_MODEL", DECISION_MODEL)
        self.client = OpenAI(
            api_key=key,
            base_url=base_url or os.getenv("LLM_BASE_URL", "https://api.x.ai/v1"),
            timeout=120.0,
        )

    def complete(self, system: str, user: str, action_ids: list[int] | None = None) -> str:
        payload = response_kwargs(
            self.model,
            system,
            user,
            action_ids=action_ids,
            reasoning=self.report and action_ids is None,
        )
        if action_ids is None:
            response = self.client.responses.create(**payload)
            text = getattr(response, "output_text", None)
            if text:
                return text
            message = next(item for item in response.output if getattr(item, "type", None) == "message")
            return next(part.text for part in message.content if getattr(part, "type", None) == "output_text")
        response = self.client.responses.parse(**payload)
        parsed = response.output_parsed
        if parsed is None:
            raise LLMError("The model did not return a choice.")
        action_id = parsed.action_id
        if hasattr(action_id, "value"):
            action_id = action_id.value
        return json.dumps({"action_id": int(action_id), "reason": str(parsed.reason or "")})


def choice_model(action_ids: list[int]) -> type[BaseModel]:
    """Schema whose action_id enum is exactly the ids in this decision."""
    ids = tuple(dict.fromkeys(int(item) for item in action_ids))
    if not ids:
        raise LLMError("There is no legal action id.")
    kind = Literal[*ids]
    return create_model("ActionChoice", action_id=(kind, ...), reason=(str, ...))


def response_kwargs(
    model: str,
    system: str,
    user: str,
    *,
    action_ids: list[int] | None = None,
    reasoning: bool = False,
) -> dict:
    """Request fields for one completion. Decision calls omit reasoning.

    The report asks for medium effort so it can connect the brief instead of
    paraphrasing it. Action calls leave this flag false.
    """
    payload = {
        "model": model,
        "input": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if action_ids is not None:
        payload["text_format"] = choice_model(action_ids)
    if reasoning:
        payload["reasoning"] = {"effort": "medium"}
    return payload


def _resist(damage_type: str, target) -> float:
    if damage_type in target.immunities:
        return 0.0
    if damage_type in target.resistances:
        return 0.5
    if damage_type in target.vulnerabilities:
        return 2.0
    return 1.0


def _hit_probs(bonus: int, armor_class: int, *, mode: int = 0, penalty: int = 0) -> tuple[float, float]:
    def hits_on(roll: int) -> tuple[int, int]:
        if roll == 1:
            return 0, 0
        if roll == 20 or roll + bonus - penalty >= armor_class:
            return 1, 1 if roll == 20 else 0
        return 0, 0

    if mode == 0:
        hits = 0
        crits = 0
        for roll in range(1, 21):
            hit, crit = hits_on(roll)
            hits += hit
            crits += crit
        return hits / 20, crits / 20
    hits = 0
    crits = 0
    for left in range(1, 21):
        for right in range(1, 21):
            hit, crit = hits_on(max(left, right) if mode > 0 else min(left, right))
            hits += hit
            crits += crit
    return hits / 400, crits / 400


def _fail_chance(bonus: int, dc: int, *, advantage: bool = False, disadvantage: bool = False, penalty: int = 0) -> float:
    if advantage and disadvantage:
        advantage = False
        disadvantage = False

    def fails(roll: int) -> bool:
        return roll + bonus - penalty < dc

    if not advantage and not disadvantage:
        return sum(1 for roll in range(1, 21) if fails(roll)) / 20
    kept = max if advantage else min
    return sum(1 for left in range(1, 21) for right in range(1, 21) if fails(kept(left, right))) / 400


def _parts_from_option(option) -> list[tuple[str, str]]:
    return [(part["type"], part["expr"]) for part in option.damage]


def _expected_parts(
    parts,
    target,
    *,
    bonus=None,
    dc=None,
    save_bonus=0,
    success=None,
    mode: int = 0,
    penalty: int = 0,
    always_crit: bool = False,
) -> float:
    total = 0.0
    for damage_type, expression in parts:
        multiplier = _resist(damage_type, target)
        if bonus is not None:
            hit_chance, crit_chance = _hit_probs(bonus, target.ac, mode=mode, penalty=penalty)
            if always_crit:
                total += multiplier * hit_chance * critical_average(expression)
            else:
                total += multiplier * (
                    (hit_chance - crit_chance) * average(expression) + crit_chance * critical_average(expression)
                )
        elif dc is not None:
            fail = _fail_chance(save_bonus, dc, penalty=penalty)
            typical = average(expression)
            if success == "half":
                total += multiplier * typical * (0.5 + 0.5 * fail)
            else:
                total += multiplier * typical * fail
        else:
            total += multiplier * average(expression)
    return total


def _save_bonus(target, ability: str | None) -> int:
    from src.rules import ability_name

    if not ability:
        return 0
    full = ability_name(ability)
    return int(target.save_bonuses.get(full, 0))


def _spiritual_weapon_action(action) -> bool:
    if action.kind == "feature" and action.attack_name == "Spiritual Weapon":
        return True
    return action.kind == "spell" and action.spell_index == "spiritual-weapon"


def _zone_repeat(action) -> bool:
    return action.kind == "feature" and action.tag in {"zone", "bolt"}


def _offense(action) -> bool:
    return action.kind in {"attack", "spell"} or _spiritual_weapon_action(action) or _zone_repeat(action)


def _spiritual_weapon_score(encounter, actor, action) -> float:
    if action.aim is None:
        return 0.0
    foe = adjacent_foe(encounter, actor, action.aim)
    spell = actor.spell_named("spiritual-weapon")
    if foe is None or spell is None:
        return 0.0
    if action.kind == "spell":
        slot = int(action.slot_level or spell.level)
    else:
        placed = actor.traits.get("spiritual_weapon") or {}
        slot = int(placed.get("slot") or actor.ongoing_slot or spell.level)
    return _expected_parts(spell_parts(actor, spell, slot), foe, bonus=actor.spell_attack_bonus)


def _zone_dice(actor, spec, slot: int, phase: str) -> list[tuple[str, str]]:
    spell = actor.spell_named(spec.index)
    level = spell.level if spell is not None else 0
    parts = []
    for damage_type, expression in effect_parts(spec, slot, level, phase):
        if "MOD" in expression.upper():
            expression = substitute_mod(expression, actor.spellcasting_mod)
        parts.append((damage_type, expression))
    return parts


def _zone_score(encounter, actor, action) -> float:
    """A placed zone is worth the rolls it keeps making, not one blast."""
    placed = actor.traits.get("zone") if isinstance(actor.traits.get("zone"), dict) else {}
    index = action.spell_index or placed.get("index")
    spec = zone_spec(str(index or ""))
    if spec is None:
        return 0.0
    if action.kind == "spell":
        slot = int(action.slot_level or 0)
    else:
        slot = int(placed.get("slot") or actor.ongoing_slot or 0)
    parts = _zone_dice(actor, spec, slot, "main")
    if spec.attack:
        if action.aim is None:
            return 0.0
        foe = _token_target(encounter, actor, action.aim, spec)
        if foe is None:
            return 0.0
        return _expected_parts(parts, foe, bonus=actor.spell_attack_bonus)
    if spec.shape == "token" and parts and action.aim is not None:
        foe = _token_target(encounter, actor, action.aim, spec)
        if foe is None:
            return 0.0
        return _expected_parts(
            parts,
            foe,
            dc=actor.spell_save_dc,
            save_bonus=_save_bonus(foe, spec.save),
            success=spec.success,
        )
    if action.tag == "bolt" and action.aim is not None and parts:
        from src.zones import bolt_cells

        cells = bolt_cells((int(action.aim[0]), int(action.aim[1])), spec.bolt)
        total = 0.0
        for target in encounter.combatants:
            if target.hp <= 0 or target.side == actor.side or not target.in_cells(cells):
                continue
            total += _expected_parts(
                parts,
                target,
                dc=actor.spell_save_dc,
                save_bonus=_save_bonus(target, spec.save),
                success=spec.success,
            )
        return total
    targets = targets_for(encounter, actor, action) if action.kind == "spell" else []
    if spec.allows("squares") and parts:
        total = 0.0
        for target in targets:
            sign = 1.0 if target.side != actor.side else -1.5
            total += sign * 2 * _expected_parts(parts, target)
        return total
    if parts and (spec.allows("cast") or action.tag == "bolt"):
        total = 0.0
        for target in targets:
            sign = 1.0 if target.side != actor.side else -1.5
            total += sign * _expected_parts(
                parts,
                target,
                dc=actor.spell_save_dc,
                save_bonus=_save_bonus(target, spec.save),
                success=spec.success,
            )
        return total
    if spec.condition and spec.allows("cast"):
        return 4.0 * sum(1 for target in targets if target.side != actor.side)
    return 0.0


CONCENTRATION_TURNS = 3
_ACCURACY = {"paralyzed", "unconscious", "restrained", "stunned", "petrified", "blinded", "prone", "outlined"}
_OWN_DISADVANTAGE = {"poisoned", "restrained", "frightened", "blinded", "prone"}
_ONE_TURN = {"end", "start", "source-next-end"}


@contextmanager
def _masked(creature, **changes):
    """Overlay conditions for a score, then put the previous set back."""
    previous = dict(creature.conditions)
    updated = dict(previous)
    for key, value in changes.items():
        if value is None:
            updated.pop(key, None)
        else:
            updated[key] = value
    creature.conditions = updated
    try:
        yield
    finally:
        creature.conditions = previous


def _added(name: str) -> dict:
    rows = {name: 99}
    if name in ACTION_DENY:
        rows["incapacitated"] = 99
    if name == "unconscious":
        rows["prone"] = 99
    return rows


def _horizon(encounter) -> int:
    cap = int(getattr(encounter, "round_cap", 15) or 15)
    return max(1, cap - int(getattr(encounter, "round", 1) or 1) + 1)


def _lasting(encounter, *, repeat_success, rounds, concentration: bool, one_turn: bool) -> float:
    """How many turns a lock should be worth. Concentration stops at three."""
    horizon = float(_horizon(encounter))
    if one_turn:
        return 1.0
    if repeat_success is None or repeat_success <= 0:
        length = horizon
    else:
        length = 1.0 / repeat_success
    cap = horizon
    if rounds is not None and int(rounds) > 0:
        cap = min(cap, float(rounds))
    if concentration:
        cap = min(cap, float(CONCENTRATION_TURNS))
    return max(1.0, min(length, cap))


def _save_fail(target, ability, dc: int, *, advantage: bool = False) -> float:
    if not ability:
        return 1.0
    mode = save_mode(target, str(ability))
    if mode is None:
        return 1.0
    adv = bool(advantage or mode > 0)
    disadv = mode < 0
    if advantage and mode < 0:
        adv = False
        disadv = False
    elif advantage:
        adv = True
        disadv = False
    return _fail_chance(
        _save_bonus(target, str(ability)),
        int(dc or 0),
        advantage=adv,
        disadvantage=disadv,
        penalty=save_penalty(target),
    )


def _effect(name: str, turns: float, *, ends_on_damage: bool = False, charm: bool = False, fear: bool = False) -> dict:
    deny = name in ACTION_DENY
    return {
        "name": name,
        "deny": deny,
        "accuracy": name in _ACCURACY,
        "disadvantage": name in _OWN_DISADVANTAGE and not deny,
        "ends_on_damage": ends_on_damage,
        "charm": charm or name == "charmed",
        "fear_dash": fear,
        "grapple": name == "grappled",
        "turns": float(turns),
        "token": 0.25 if name == "deafened" else None,
        "no_ally": False,
        "bonus_damage": 0.0,
    }


def _die_sums(count: int, sides: int, modifier: int) -> dict[int, float]:
    if sides == 0 or count <= 0:
        return {int(modifier): 1.0}
    dist = {0: 1.0}
    face = 1.0 / sides
    for _ in range(count):
        nxt: dict[int, float] = {}
        for total, prob in dist.items():
            share = prob * face
            for roll in range(1, sides + 1):
                nxt[total + roll] = nxt.get(total + roll, 0.0) + share
        dist = nxt
    if modifier:
        return {total + modifier: prob for total, prob in dist.items()}
    return dist


def _typed_distribution(expression: str, damage_type: str, target, *, critical: bool = False, halved: bool = False) -> dict[int, float]:
    count, sides, modifier = parse_dice(expression)
    if critical and sides:
        count *= 2
    out: dict[int, float] = {}
    for total, prob in _die_sums(count, sides, modifier).items():
        amount = max(0, total)
        if halved:
            amount //= 2
        dealt = _typed_amount(amount, damage_type, target)
        out[dealt] = out.get(dealt, 0.0) + prob
    return out


def _convolve(left: dict[int, float], right: dict[int, float]) -> dict[int, float]:
    nxt: dict[int, float] = {}
    for left_total, left_prob in left.items():
        for right_total, right_prob in right.items():
            key = left_total + right_total
            nxt[key] = nxt.get(key, 0.0) + left_prob * right_prob
    return nxt


def _max_typed(parts, target, *, critical: bool = False, halved: bool = False) -> int:
    total = 0
    for damage_type, expression in parts:
        count, sides, modifier = parse_dice(expression)
        if critical and sides:
            count *= 2
        raw = count * sides + modifier if sides else modifier
        amount = max(0, raw)
        if halved:
            amount //= 2
        total += _typed_amount(amount, damage_type, target)
    return total


def _p_at_least(parts, target, hp: int, *, critical: bool = False, halved: bool = False) -> float:
    if hp <= 0:
        return 1.0
    if not parts or _max_typed(parts, target, critical=critical, halved=halved) < hp:
        return 0.0
    dist = {0: 1.0}
    for damage_type, expression in parts:
        dist = _convolve(dist, _typed_distribution(expression, damage_type, target, critical=critical, halved=halved))
    return sum(prob for total, prob in dist.items() if total >= hp)


def _p_kill_attack(parts, target, *, hit: float, crit: float, always_crit: bool) -> float:
    if not parts or target.hp <= 0:
        return 0.0
    hp = int(target.hp)
    if always_crit:
        if _max_typed(parts, target, critical=True) < hp:
            return 0.0
        return hit * _p_at_least(parts, target, hp, critical=True)
    chance = 0.0
    if hit > crit and _max_typed(parts, target) >= hp:
        chance += (hit - crit) * _p_at_least(parts, target, hp)
    if crit > 0 and _max_typed(parts, target, critical=True) >= hp:
        chance += crit * _p_at_least(parts, target, hp, critical=True)
    return chance


def _p_kill_save(parts, target, *, fail: float, success) -> float:
    if not parts or target.hp <= 0:
        return 0.0
    hp = int(target.hp)
    chance = 0.0
    if fail > 0 and _max_typed(parts, target) >= hp:
        chance += fail * _p_at_least(parts, target, hp)
    if success == "half" and fail < 1 and _max_typed(parts, target, halved=True) >= hp:
        chance += (1.0 - fail) * _p_at_least(parts, target, hp, halved=True)
    return chance


def _foes_of(encounter, creature):
    rows = []
    for other in encounter.combatants:
        if other.id == creature.id or other.hp <= 0 or other.side == creature.side:
            continue
        if other.traits.get("banished") or other.traits.get("removed"):
            continue
        rows.append(other)
    return rows


def _another_threat(encounter, target) -> bool:
    for other in encounter.combatants:
        if other.id == target.id or other.hp <= 0 or other.side != target.side:
            continue
        if other.traits.get("banished") or other.traits.get("removed") or is_incapacitated(other):
            continue
        return True
    return False


def _has_farther(encounter, actor, source) -> bool:
    if source is None:
        return False
    budget = movement_budget(actor) + turn_speed(actor)
    if budget < 5:
        return False
    current = actor.gap_to(source)
    for square in _reachable(encounter, actor, budget):
        if space_between(square, actor, source) > current:
            return True
    return False


def _melee_swing(attacker, target, option) -> bool:
    return bool(getattr(option, "reach_feet", None)) and using_melee(attacker, target, option)


def _strike(attacker, target, option) -> float:
    melee = _melee_swing(attacker, target, option)
    return _expected_parts(
        _parts_from_option(option),
        target,
        bonus=int(option.attack_bonus or 0),
        mode=attack_mode(attacker, target, melee=melee),
        penalty=attack_penalty(attacker),
        always_crit=bool(auto_crit(attacker, target) and melee),
    )


def _save_damage(parts, target, *, dc, ability, success) -> float:
    if ability and save_mode(target, str(ability)) is None:
        return _expected_parts(parts, target)
    return _expected_parts(
        parts,
        target,
        dc=dc,
        save_bonus=_save_bonus(target, ability),
        success=success,
        penalty=save_penalty(target),
    )


def _could_reach(actor, target, option) -> bool:
    if weapon_reaches(actor, target, option):
        return True
    gap = actor.gap_to(target)
    if gap <= 0:
        return False
    feet = gap * 5
    speed = max(0, int(actor.speed or 0))
    reach = int(option.reach_feet or 0)
    ranged = int(option.range_feet or 0)
    if reach and feet <= reach + speed:
        return True
    if ranged and feet <= ranged + speed:
        return True
    return not reach and not ranged and feet <= 5 + speed


def _spell_in_reach(actor, target, spell) -> bool:
    gap = actor.gap_to(target)
    speed = max(0, int(actor.speed or 0))
    feet = max(0, gap) * 5
    if spell.area_type and spell.range_mode == "self":
        return feet <= int(spell.area_size or 0) + speed
    if spell.range_mode == "self":
        return target.id == actor.id
    if spell.range_mode == "touch":
        return gap > 0 and feet <= 5 + speed
    reach = spell_range_feet(actor, spell) + int(spell.area_size or 0)
    return gap > 0 and feet <= reach + speed


def _spell_amount(actor, target, spell, parts) -> float:
    if spell.kind == "attack":
        melee = spell.attack_type == "melee" and actor.gap_to(target) <= 1
        return _expected_parts(
            parts,
            target,
            bonus=actor.spell_attack_bonus,
            mode=attack_mode(actor, target, melee=melee, spell=True),
            penalty=attack_penalty(actor),
            always_crit=bool(auto_crit(actor, target) and melee),
        )
    if spell.dc_ability or spell.kind == "save":
        return _save_damage(parts, target, dc=actor.spell_save_dc, ability=spell.dc_ability, success=spell.dc_success)
    return _expected_parts(parts, target)


def _threat_against(actor, foe) -> float:
    """Damage this creature could land on ``foe`` this turn, even while locked."""
    if actor.multiattack:
        total = 0.0
        reached = False
        for name, count in actor.multiattack:
            option = actor.attack_named(name)
            if option is None or option.resolution != "attack" or not _option_ready(actor, option):
                continue
            if not _could_reach(actor, foe, option):
                continue
            reached = True
            total += int(count) * _strike(actor, foe, option)
        if reached:
            return total
    best = 0.0
    for option in actor.attacks:
        if option.activation != "action" or not _option_ready(actor, option) or not _could_reach(actor, foe, option):
            continue
        if option.resolution == "attack":
            best = max(best, _strike(actor, foe, option))
        elif option.resolution == "save":
            best = max(best, _save_damage(
                _parts_from_option(option),
                foe,
                dc=option.dc_value or 10,
                ability=option.dc_ability,
                success=option.dc_success,
            ))
    for spell in getattr(actor, "spells", ()) or ():
        if spell.activation != "action" or spell.kind in {"heal", "excluded", "afflict"}:
            continue
        if not _spell_in_reach(actor, foe, spell):
            continue
        slot = _lowest_slot(actor, spell)
        if slot is None and actor.concentrating_on == spell.index and actor.ongoing_slot:
            slot = int(actor.ongoing_slot)
        if slot is None:
            continue
        parts = spell_parts(actor, spell, slot)
        if parts:
            best = max(best, _spell_amount(actor, foe, spell, parts))
    return best


def _outgoing(encounter, creature) -> float:
    best = 0.0
    for foe in _foes_of(encounter, creature):
        best = max(best, _threat_against(creature, foe))
    return best


def _outgoing_toward(encounter, creature, victim) -> float:
    if victim is None or victim.hp <= 0 or victim.traits.get("banished") or victim.traits.get("removed"):
        return 0.0
    return _threat_against(creature, victim)


def _outgoing_under(encounter, creature, name: str) -> float:
    with _masked(creature, **_added(name)):
        return _outgoing(encounter, creature)


def _weapons_against(attacker, target) -> float:
    """Weapon damage from where the attacker is standing. No kill term."""
    if attacker.multiattack:
        total = 0.0
        reached = False
        for name, count in attacker.multiattack:
            option = attacker.attack_named(name)
            if option is None or option.resolution != "attack" or not _option_ready(attacker, option):
                continue
            if not weapon_reaches(attacker, target, option):
                continue
            reached = True
            total += int(count) * _strike(attacker, target, option)
        if reached:
            return total
    best = 0.0
    for option in attacker.attacks:
        if option.activation != "action" or option.resolution != "attack" or not _option_ready(attacker, option):
            continue
        if weapon_reaches(attacker, target, option):
            best = max(best, _strike(attacker, target, option))
    return best


def _petrified_factor(attacker, target) -> float:
    if "petrified" not in target.conditions:
        return 1.0
    kinds = [part.get("type") for option in attacker.attacks if option.resolution == "attack" for part in option.damage]
    if not kinds:
        return 0.5
    if all(kind == "poison" or kind in target.immunities for kind in kinds):
        return 0.0
    if all(kind in target.resistances or kind in target.immunities or kind == "poison" for kind in kinds):
        return 1.0
    return 0.5


def _swing(attacker, target) -> float:
    amount = _weapons_against(attacker, target)
    if amount <= 0:
        return 0.0
    return amount * _petrified_factor(attacker, target)


def _ally_delta(encounter, actor, target, name: str) -> float:
    if not name:
        return 0.0
    total = 0.0
    for ally in encounter.combatants:
        if ally.hp <= 0 or ally.side != actor.side or ally.id == target.id or is_incapacitated(ally):
            continue
        before = _swing(ally, target)
        with _masked(target, **_added(name)):
            after = _swing(ally, target)
        total += after - before
    return total


def _control_amount(encounter, actor, target, effect: dict, chance: float, *, fresh: bool = True) -> float:
    if chance <= 0 or target.hp <= 0:
        return 0.0
    name = str(effect.get("name") or "")
    if name and immune_to(target, name, actor):
        return 0.0
    if fresh and effect.get("deny") and is_incapacitated(target):
        return 0.0
    turns = float(effect.get("turns") or 1)
    if effect.get("ends_on_damage") and not _another_threat(encounter, target):
        turns = min(turns, 1.0)
    if effect.get("token") is not None:
        return float(effect["token"]) * chance
    removed = 0.0
    pressure = _outgoing(encounter, target)
    if effect.get("charm"):
        removed += _outgoing_toward(encounter, target, actor) * min(turns, 1.0)
    elif effect.get("fear_dash"):
        if _has_farther(encounter, target, actor):
            removed += pressure * turns
        else:
            removed += max(0.0, pressure - _outgoing_under(encounter, target, "frightened")) * turns
    elif effect.get("deny"):
        removed += pressure * turns
    elif effect.get("disadvantage") and name:
        removed += max(0.0, pressure - _outgoing_under(encounter, target, name)) * turns
    if effect.get("accuracy") and not effect.get("no_ally"):
        removed += _ally_delta(encounter, actor, target, name) * turns
    if effect.get("grapple") and not effect.get("deny"):
        removed += 1.0
    if effect.get("bonus_damage"):
        removed += float(effect["bonus_damage"]) * turns
    return removed * chance


def _kill_value(encounter, target, chance: float) -> float:
    if chance <= 0 or target.hp <= 0 or target.side is None:
        return 0.0
    return chance * _outgoing(encounter, target) * _horizon(encounter)


def _chosen(encounter, actor, action, spell=None):
    targets = list(targets_for(encounter, actor, action))
    spec = affliction(getattr(spell, "index", None)) if spell is not None else None
    if spec:
        allowed = {creature.id for creature in _afflict_pool(encounter, actor, spell, spec)}
        targets = [creature for creature in targets if creature.id in allowed]
    return [creature for creature in targets if creature.hp > 0]


def _attack_faces(actor, target, option) -> tuple[float, float, bool]:
    melee = _melee_swing(actor, target, option)
    always = bool(auto_crit(actor, target) and melee)
    hit, crit = _hit_probs(
        int(option.attack_bonus or 0),
        target.ac,
        mode=attack_mode(actor, target, melee=melee),
        penalty=attack_penalty(actor),
    )
    return hit, crit, always


def _weapon_rider(encounter, actor, option, target, *, hit_chance: float | None = None) -> float:
    rider = getattr(option, "rider", None) or {}
    if not rider or rider_skipped(target, rider):
        return 0.0
    if rider.get("charge_feet") and not rider.get("grapple_size") and not rider.get("save_condition") and not rider.get("first"):
        return 0.0
    condition = str(rider.get("save_condition") or rider.get("hit_condition") or rider.get("first") or "")
    if not condition and rider.get("prone_size") and not rider.get("charge_feet"):
        condition = "prone"
    if not condition and rider.get("grapple_size"):
        condition = "restrained" if rider.get("restrain_while_grappled") else "grappled"
    if not condition or immune_to(target, condition, actor):
        return 0.0
    limit = rider.get("prone_size") if condition == "prone" else rider.get("grapple_size")
    if limit and size_rank(target.size) > size_rank(str(limit)):
        return 0.0
    if hit_chance is None:
        hit_chance, _crit, _always = _attack_faces(actor, target, option)
    if rider.get("save_ability") and rider.get("save_dc"):
        chance = hit_chance * _save_fail(target, rider.get("save_ability"), int(rider["save_dc"]))
    else:
        chance = hit_chance
    until = str(rider.get("save_until") or rider.get("hit_until") or "end")
    effect = _effect(condition, 1.0 if until in _ONE_TURN or until == "stand" or not rider.get("save_ability") else 1.0)
    return _control_amount(encounter, actor, target, effect, chance)


def _invisibility_gain(encounter, beneficiary, turns: float) -> float:
    offense_now = _outgoing(encounter, beneficiary)
    defense_now = sum(_swing(foe, beneficiary) for foe in _foes_of(encounter, beneficiary) if not is_incapacitated(foe))
    with _masked(beneficiary, **_added("invisible")):
        offense_then = _outgoing(encounter, beneficiary)
        defense_then = sum(_swing(foe, beneficiary) for foe in _foes_of(encounter, beneficiary) if not is_incapacitated(foe))
    gain = max(0.0, offense_then - offense_now) + max(0.0, defense_now - defense_then)
    return gain * float(turns)


def _value_of_mark(encounter, actor, target, mark: dict) -> float:
    """What an existing mark still prevents. ``fresh`` stays false so a lock is not scored as zero."""
    condition = str(mark.get("condition") or "")
    until = str(mark.get("until") or "")
    if mark.get("heroism"):
        turns = _lasting(encounter, repeat_success=None, rounds=10, concentration=True, one_turn=False)
        return float(max(0, int(mark.get("temp") or 0))) * turns
    if mark.get("speed_zero") and not condition and not mark.get("order"):
        return 1.0
    if condition == "invisible":
        turns = 1.0 if mark.get("ends_on_attack") else _lasting(
            encounter,
            repeat_success=None,
            rounds=10 if until == "concentration" else None,
            concentration=until == "concentration",
            one_turn=False,
        )
        return _invisibility_gain(encounter, target, turns)
    if mark.get("repeat_order"):
        turns = _lasting(encounter, repeat_success=None, rounds=10, concentration=True, one_turn=False)
        effect = _effect("frightened", turns, fear=True)
        return _control_amount(encounter, actor, target, effect, 1.0, fresh=False)
    if mark.get("banish"):
        turns = _lasting(encounter, repeat_success=None, rounds=10, concentration=True, one_turn=False)
        effect = _effect("incapacitated", turns)
        effect["no_ally"] = True
        return _control_amount(encounter, actor, target, effect, 1.0, fresh=False)
    if mark.get("order"):
        effect = _effect("", 1.0)
        effect["deny"] = True
        effect["name"] = ""
        return _control_amount(encounter, actor, target, effect, 1.0, fresh=False)
    if not condition:
        return 0.0
    one = until in _ONE_TURN or condition == "charmed"
    repeat_success = None
    if mark.get("repeat") and mark.get("save") and not one:
        repeat_success = 1.0 - _save_fail(target, str(mark.get("save") or ""), int(mark.get("dc") or 0))
    rounds = int(mark.get("rounds") or 0) or None
    if until not in {"rounds", "concentration"}:
        rounds = None
    turns = _lasting(
        encounter,
        repeat_success=repeat_success,
        rounds=rounds,
        concentration=until == "concentration",
        one_turn=one,
    )
    effect = _effect(
        condition,
        turns,
        ends_on_damage=bool(mark.get("ends_on_damage") or mark.get("damage_save")),
        charm=condition == "charmed",
    )
    return _control_amount(encounter, actor, target, effect, 1.0, fresh=False)


def _wake_penalty(encounter, actor, target) -> float:
    if target.side == actor.side or not _another_threat(encounter, target):
        return 0.0
    total = 0.0
    for mark in marks_of(target):
        ends = bool(mark.get("ends_on_damage") or mark.get("damage_save"))
        charm = bool(mark.get("charm_damage") and mark.get("source_side") == actor.side)
        if ends or charm:
            total += _value_of_mark(encounter, actor, target, mark)
    return total


def _concentration_cost(encounter, actor, spell) -> float:
    if spell is None or not holds_concentration(spell) or not actor.concentrating_on:
        return 0.0
    current = actor.concentrating_on
    total = 0.0
    for creature in encounter.combatants:
        best = 0.0
        for mark in marks_of(creature):
            if mark.get("source") != actor.id or mark.get("spell") != current:
                continue
            if mark.get("tied") != "concentration" and mark.get("until") != "concentration":
                continue
            best = max(best, _value_of_mark(encounter, actor, creature, mark))
        total += best
    return total


def _restoration_gain(encounter, target, condition: str) -> float:
    if condition == "exhaustion":
        if exhaustion_level(target) <= 0:
            return 0.0
    elif condition not in target.conditions:
        return 0.0
    horizon = float(_horizon(encounter))
    if condition in ACTION_DENY:
        return _outgoing(encounter, target) * horizon
    current = _outgoing(encounter, target)
    if condition == "exhaustion":
        level = max(0, exhaustion_level(target) - 1)
        changes = {"exhaustion": level or None}
    else:
        changes = {condition: None}
    with _masked(target, **changes):
        freed = _outgoing(encounter, target)
    return max(0.0, freed - current) * horizon


def _stand_value(encounter, actor) -> float:
    foes = _foes_of(encounter, actor)
    if not foes:
        return 0.0
    now = max(_weapons_against(actor, foe) for foe in foes)
    with _masked(actor, prone=None):
        later = max(_weapons_against(actor, foe) for foe in foes)
    return max(0.0, later - now)


def _escape_value(encounter, actor) -> float:
    current = _outgoing(encounter, actor)
    with _masked(actor, restrained=None, grappled=None):
        freed = _outgoing(encounter, actor)
    return max(1.0, freed - current)


def _shake_value(encounter, actor, action) -> float:
    if not action.target_id:
        return 0.0
    try:
        target = encounter.by_id(action.target_id)
    except KeyError:
        return 0.0
    value = 0.0
    for mark in marks_of(target):
        if mark.get("shake") or mark.get("ends_on_damage"):
            value += _value_of_mark(encounter, actor, target, mark)
    if target.side == actor.side:
        return value
    return -abs(value)


def _cast_condition(encounter, actor, target, spec: dict, condition: str, dc: int) -> float:
    if spec.get("humanoid") and not is_humanoid(target):
        return 0.0
    if not condition or immune_to(target, condition, actor):
        return 0.0
    advantage = bool(spec.get("fight_advantage")) or (condition == "poisoned" and bool(target.traits.get("poison_ward")))
    fail = _save_fail(target, spec.get("save"), dc, advantage=advantage) if spec.get("save") else 1.0
    one = str(spec.get("until") or "") in _ONE_TURN or condition == "charmed"
    repeat_success = None
    if spec.get("repeat") and spec.get("save") and not one:
        repeat_success = 1.0 - _save_fail(target, spec.get("save"), dc, advantage=advantage)
    concentration = bool(spec.get("concentration")) or spec.get("until") == "concentration"
    rounds = spec.get("rounds") if concentration or spec.get("until") == "rounds" else None
    turns = _lasting(encounter, repeat_success=repeat_success, rounds=rounds, concentration=concentration, one_turn=one)
    effect = _effect(condition, turns, ends_on_damage=bool(spec.get("ends_on_damage")), charm=condition == "charmed")
    return _control_amount(encounter, actor, target, effect, fail)


def _special_control(encounter, actor, action, spell, spec: dict, target, dc: int) -> float | None:
    """A spell whose duration is not the generic condition window. None means use that window."""
    name = str(spec.get("special") or "")
    if name == "sleep":
        if is_elf(target) or immune_to(target, "exhaustion"):
            return 0.0
        fail = _save_fail(target, "wisdom", dc)
        pressure = _outgoing(encounter, target)
        if not _another_threat(encounter, target):
            return fail * pressure
        window = _lasting(encounter, repeat_success=None, rounds=10, concentration=True, one_turn=False)
        return fail * (pressure + fail * pressure * window)
    if name == "hypnotic":
        fail = _save_fail(target, "wisdom", dc)
        turns = _lasting(encounter, repeat_success=None, rounds=10, concentration=True, one_turn=False)
        return _control_amount(encounter, actor, target, _effect("incapacitated", turns, ends_on_damage=True), fail)
    if name == "laughter":
        fail = _save_fail(target, "wisdom", dc)
        turns = _lasting(encounter, repeat_success=1.0 - fail, rounds=10, concentration=True, one_turn=False)
        return _control_amount(encounter, actor, target, _effect("incapacitated", turns, ends_on_damage=True), fail)
    if name == "banish":
        fail = _save_fail(target, "charisma", dc)
        turns = _lasting(encounter, repeat_success=None, rounds=10, concentration=True, one_turn=False)
        effect = _effect("incapacitated", turns)
        effect["no_ally"] = True
        return _control_amount(encounter, actor, target, effect, fail)
    if name == "power-stun":
        if target.hp > 150:
            return 1.0
        if immune_to(target, "stunned", actor):
            return 0.0
        fail = _save_fail(target, "constitution", dc)
        turns = _lasting(encounter, repeat_success=1.0 - fail, rounds=None, concentration=False, one_turn=False)
        return _control_amount(encounter, actor, target, _effect("stunned", turns), 1.0)
    if name == "flesh":
        if "construct" in creature_types(target):
            return 0.0
        fail = _save_fail(target, "constitution", dc)
        turns = _lasting(encounter, repeat_success=1.0 - fail, rounds=10, concentration=True, one_turn=False)
        return _control_amount(encounter, actor, target, _effect("restrained", turns), fail) + 4.0 * fail
    if name == "dance":
        fail = _save_fail(target, "wisdom", dc)
        window = _lasting(encounter, repeat_success=None, rounds=10, concentration=True, one_turn=False)
        pressure = _outgoing(encounter, target)
        return 0.35 * pressure * (fail * window + (1.0 - fail))
    if name == "fear":
        fail = _save_fail(target, "wisdom", dc)
        turns = _lasting(encounter, repeat_success=None, rounds=10, concentration=True, one_turn=False)
        return _control_amount(encounter, actor, target, _effect("frightened", turns, fear=True), fail)
    if name == "command":
        fail = _save_fail(target, "wisdom", dc)
        effect = _effect("", 1.0)
        effect["deny"] = True
        effect["name"] = ""
        return _control_amount(encounter, actor, target, effect, fail)
    if name == "divine":
        fail = _save_fail(target, "charisma", dc)
        if fail <= 0:
            return 0.0
        if creature_types(target) & BANISH_TYPES or target.hp <= 20:
            return fail * _outgoing(encounter, target) * _horizon(encounter)
        if target.hp <= 30:
            return _control_amount(encounter, actor, target, _effect("stunned", float(_horizon(encounter))), fail)
        if 41 <= target.hp <= 50:
            return fail * 0.25
        return 0.0
    if name == "eyebite":
        choice = action.attack_name or "asleep"
        fail = _save_fail(target, "wisdom", dc)
        turns = _lasting(encounter, repeat_success=None, rounds=10, concentration=True, one_turn=False)
        if choice == "asleep":
            effect = _effect("unconscious", turns, ends_on_damage=True)
        elif choice == "panicked":
            effect = _effect("frightened", turns, fear=True)
        else:
            effect = _effect("poisoned", turns)
        return _control_amount(encounter, actor, target, effect, fail)
    if name == "ensnare":
        advantage = size_rank(target.size) >= size_rank("large")
        fail = _save_fail(target, "strength", dc, advantage=advantage)
        turns = _lasting(encounter, repeat_success=None, rounds=10, concentration=True, one_turn=False)
        effect = _effect("restrained", turns)
        effect["bonus_damage"] = 3.5
        return _control_amount(encounter, actor, target, effect, fail)
    if name == "smite":
        turns = _lasting(encounter, repeat_success=None, rounds=10, concentration=True, one_turn=False)
        return _control_amount(encounter, actor, target, _effect("outlined", turns), 1.0)
    return None


def _spell_control(encounter, actor, action, spell) -> float:
    spec = affliction(spell.index) or {}
    if not spec or spec.get("mode") == "rider":
        return 0.0
    if spec.get("special") in {
        "see", "heroism", "protection", "freedom", "lesser", "greater",
        "mass-heal", "power-heal", "poison-ward", "calm",
    }:
        return 0.0
    if spec.get("condition") == "invisible":
        return 0.0
    total = 0.0
    dc = int(actor.spell_save_dc or 0)
    choices = set(spec.get("choices") or [])
    for target in _chosen(encounter, actor, action, spell):
        if target.side == actor.side:
            continue
        special = _special_control(encounter, actor, action, spell, spec, target, dc)
        if special is not None:
            total += special
            continue
        condition = str(spec.get("condition") or "")
        if action.attack_name and action.attack_name in choices:
            condition = str(action.attack_name)
        total += _cast_condition(encounter, actor, target, spec, condition, dc)
    return total


def _rider_spell_control(encounter, actor, action, spell) -> float:
    spec = affliction(spell.index) or {}
    if spec.get("mode") != "rider":
        return 0.0
    condition = str(spec.get("condition") or "")
    if not condition:
        return 0.0
    until = str(spec.get("until") or "")
    one = until in _ONE_TURN
    concentration = bool(spec.get("concentration")) or until == "concentration"
    total = 0.0
    dc = int(actor.spell_save_dc or 0)
    for target in _chosen(encounter, actor, action, spell):
        if target.side == actor.side or immune_to(target, condition, actor):
            continue
        advantage = condition == "poisoned" and bool(target.traits.get("poison_ward"))
        if spec.get("on") == "hit":
            melee = spell.attack_type == "melee" and actor.gap_to(target) <= 1
            hit, _crit = _hit_probs(
                int(actor.spell_attack_bonus or 0),
                target.ac,
                mode=attack_mode(actor, target, melee=melee, spell=True),
                penalty=attack_penalty(actor),
            )
            chance = hit
        else:
            chance = _save_fail(target, spec.get("save") or spell.dc_ability, dc, advantage=advantage)
        repeat_success = None
        if spec.get("repeat") and not one:
            repeat_success = 1.0 - _save_fail(target, spec.get("save") or spell.dc_ability, dc, advantage=advantage)
        rounds = spec.get("rounds") if concentration or until == "rounds" else None
        turns = _lasting(encounter, repeat_success=repeat_success, rounds=rounds, concentration=concentration, one_turn=one)
        total += _control_amount(encounter, actor, target, _effect(condition, turns), chance)
    return total


def _buff_portions(encounter, actor, action, spell) -> tuple[float, float] | None:
    spec = affliction(spell.index) or {}
    special = str(spec.get("special") or "")
    if spec.get("condition") == "invisible" or spell.index in {"invisibility", "greater-invisibility", "mislead"}:
        turns = 1.0 if spec.get("ends_on_attack") else _lasting(
            encounter, repeat_success=None, rounds=spec.get("rounds") or 10, concentration=True, one_turn=False,
        )
        targets = _chosen(encounter, actor, action, spell) or [actor]
        value = sum(_invisibility_gain(encounter, target, turns) for target in targets)
        return value - _concentration_cost(encounter, actor, spell), value
    if special == "see":
        gain = 0.0
        for foe in _foes_of(encounter, actor):
            if "invisible" not in foe.conditions:
                continue
            now = _swing(actor, foe)
            with _masked(foe, invisible=None):
                later = _swing(actor, foe)
            gain += max(0.0, later - now)
        return gain, gain
    if special == "heroism":
        targets = _chosen(encounter, actor, action, spell)
        turns = _lasting(encounter, repeat_success=None, rounds=10, concentration=True, one_turn=False)
        value = max(0, int(actor.spellcasting_mod or 0)) * turns * max(1, len(targets))
        if not targets:
            value = 0.0
        return value - _concentration_cost(encounter, actor, spell), value
    if special == "protection":
        value = 3.0 if any(creature_types(foe) & EVIL_TYPES for foe in _foes_of(encounter, actor)) else 0.0
        return value - _concentration_cost(encounter, actor, spell), value
    if special == "freedom":
        value = 0.0
        for target in _chosen(encounter, actor, action, spell):
            best = max((_restoration_gain(encounter, target, name) for name in ("paralyzed", "restrained", "grappled")), default=0.0)
            value += best if best > 0 else 1.0
        return value, value
    if special in {"lesser", "greater"}:
        choice = str(action.attack_name or "")
        value = sum(_restoration_gain(encounter, target, choice) for target in _chosen(encounter, actor, action, spell))
        return value, value
    if special == "mass-heal":
        missing = sum(max(0, ally.max_hp - ally.hp) for ally in encounter.combatants if ally.side == actor.side and ally.hp > 0)
        value = float(min(700, missing))
        return value, value
    if special == "power-heal":
        value = sum(max(0, target.max_hp - target.hp) for target in _chosen(encounter, actor, action, spell))
        return float(value), float(value)
    if special in {"poison-ward", "calm"}:
        if special == "calm":
            value = 0.0
            for ally in encounter.combatants:
                if ally.side != actor.side or ally.hp <= 0:
                    continue
                if "charmed" in ally.conditions or "frightened" in ally.conditions:
                    value += _outgoing(encounter, ally)
            return value - _concentration_cost(encounter, actor, spell), value
        return 0.0, 0.0
    return None


def _attack_portions(encounter, actor, action) -> tuple[float, float]:
    if action.attack_name == "Multiattack":
        try:
            target = encounter.by_id(action.target_id)
        except KeyError:
            return 0.0, 0.0
        damage = 0.0
        survival = 1.0
        rider = 0.0
        hostile = target.side != actor.side and target.hp > 0
        for name, count in actor.multiattack:
            option = actor.attack_named(name)
            if option is None or option.resolution != "attack" or not weapon_reaches(actor, target, option):
                continue
            damage += int(count) * _strike(actor, target, option)
            if hostile:
                hit, crit, always = _attack_faces(actor, target, option)
                chance = _p_kill_attack(_parts_from_option(option), target, hit=hit, crit=crit, always_crit=always)
                if int(count) > 1:
                    chance = 1.0 - (1.0 - chance) ** int(count)
                survival *= 1.0 - chance
                landed = 1.0 - (1.0 - hit) ** int(count)
                rider += _weapon_rider(encounter, actor, option, target, hit_chance=landed)
        kill = _kill_value(encounter, target, 1.0 - survival) if hostile else 0.0
        wake = _wake_penalty(encounter, actor, target) if hostile else 0.0
        return damage + kill + rider - wake, rider
    option = actor.attack_named(action.attack_name)
    if option is None:
        return 0.0, 0.0
    targets = [target for target in targets_for(encounter, actor, action) if target.hp > 0]
    if option.resolution == "save":
        damage = 0.0
        kill = 0.0
        wake = 0.0
        rider = 0.0
        for target in targets:
            if target.side == actor.side:
                continue
            parts = _parts_from_option(option)
            fail = _save_fail(target, option.dc_ability, int(option.dc_value or 10))
            damage += _save_damage(parts, target, dc=option.dc_value or 10, ability=option.dc_ability, success=option.dc_success)
            kill += _kill_value(encounter, target, _p_kill_save(parts, target, fail=fail, success=option.dc_success))
            wake += _wake_penalty(encounter, actor, target)
            rider += _weapon_rider(encounter, actor, option, target, hit_chance=1.0)
        return damage + kill + rider - wake, rider
    if not targets:
        return 0.0, 0.0
    target = targets[0]
    if option.resolution == "attack":
        damage = _strike(actor, target, option)
    else:
        damage = _expected_parts(_parts_from_option(option), target, bonus=option.attack_bonus or 0)
    hostile = target.side != actor.side
    kill = 0.0
    wake = 0.0
    rider = 0.0
    if hostile and option.resolution == "attack":
        parts = _parts_from_option(option)
        hit, crit, always = _attack_faces(actor, target, option)
        kill = _kill_value(encounter, target, _p_kill_attack(parts, target, hit=hit, crit=crit, always_crit=always))
        wake = _wake_penalty(encounter, actor, target)
        rider = _weapon_rider(encounter, actor, option, target)
    return damage + kill + rider - wake, rider


def _spell_portions(encounter, actor, action) -> tuple[float, float]:
    spell = actor.spell_named(action.spell_index)
    if spell is None or spell.kind == "heal":
        return 0.0, 0.0
    buff = _buff_portions(encounter, actor, action, spell)
    if buff is not None:
        return buff
    parts = spell_parts(actor, spell, int(action.slot_level or 0))
    damage = 0.0
    kill = 0.0
    wake = 0.0
    for target in _chosen(encounter, actor, action, spell):
        sign = 1.0 if target.side != actor.side else -1.5
        amount = _spell_amount(actor, target, spell, parts) if parts else 0.0
        damage += sign * amount
        if target.side == actor.side:
            continue
        if spell.kind == "attack" and parts:
            melee = spell.attack_type == "melee" and actor.gap_to(target) <= 1
            always = bool(auto_crit(actor, target) and melee)
            hit, crit = _hit_probs(
                int(actor.spell_attack_bonus or 0),
                target.ac,
                mode=attack_mode(actor, target, melee=melee, spell=True),
                penalty=attack_penalty(actor),
            )
            kill += _kill_value(encounter, target, _p_kill_attack(parts, target, hit=hit, crit=crit, always_crit=always))
        elif spell.kind == "save" and parts:
            fail = _save_fail(target, spell.dc_ability, int(actor.spell_save_dc or 0))
            kill += _kill_value(encounter, target, _p_kill_save(parts, target, fail=fail, success=spell.dc_success))
        elif parts:
            kill += _kill_value(encounter, target, _p_kill_save(parts, target, fail=1.0, success=None))
        wake += _wake_penalty(encounter, actor, target)
    control = _spell_control(encounter, actor, action, spell) + _rider_spell_control(encounter, actor, action, spell)
    cost = _concentration_cost(encounter, actor, spell)
    return damage + kill + control - wake - cost, control


def _portions(encounter, actor, action) -> tuple[float, float]:
    if action.kind == "stand":
        return _stand_value(encounter, actor), 0.0
    if action.kind == "order":
        return 0.0, 0.0
    if action.kind == "escape":
        return _escape_value(encounter, actor), 0.0
    if action.kind == "shake":
        value = _shake_value(encounter, actor, action)
        if value > 0:
            return value, value
        return value, 0.0
    if _zone_repeat(action) or (action.kind == "spell" and zone_spec(action.spell_index) is not None):
        return _zone_score(encounter, actor, action), 0.0
    if _spiritual_weapon_action(action):
        return _spiritual_weapon_score(encounter, actor, action), 0.0
    if action.kind == "attack":
        return _attack_portions(encounter, actor, action)
    if action.kind == "spell":
        return _spell_portions(encounter, actor, action)
    return 0.0, 0.0


def action_score(encounter, actor, action) -> float:
    total, _control = _portions(encounter, actor, action)
    return total


def condition_score(encounter, actor, action) -> float:
    _total, control = _portions(encounter, actor, action)
    return control


def _pressure_reason(encounter, actor, pool, chosen) -> str:
    total, control = _portions(encounter, actor, chosen)
    if control <= 0:
        return "Highest expected damage."
    bare = total - control
    others = 0.0
    for action in pool:
        if action.id == chosen.id:
            continue
        others = max(others, action_score(encounter, actor, action))
    if bare <= others:
        return "The lock removes more pressure than the hit."
    return "Highest expected damage."


def _best_offensive(encounter, actor, pool):
    """A heal for a badly wounded ally, otherwise the highest expected damage."""
    wounded: list[tuple[int, object]] = []
    for action in pool:
        if action.kind != "spell":
            continue
        spell = actor.spell_named(action.spell_index)
        if spell is None or spell.kind != "heal":
            continue
        for target in targets_for(encounter, actor, action):
            if target.max_hp and target.hp / target.max_hp <= 0.35:
                wounded.append((target.max_hp - target.hp, action))
    if wounded:
        return max(wounded, key=lambda item: item[0])[1]
    best = None
    best_score = 0.0
    for action in pool:
        if not _offense(action) and action.kind not in {"escape", "shake"}:
            continue
        score = action_score(encounter, actor, action)
        repeat = _spiritual_weapon_action(action) and action.kind == "feature"
        held = best is not None and _spiritual_weapon_action(best) and best.kind == "feature"
        if score > best_score or (score == best_score and score > 0 and repeat and not held):
            best = action
            best_score = score
    return best


def _tick_amount(actor, caster, spell) -> float:
    slot = int(caster.ongoing_slot or spell.level or 0)
    parts = spell_parts(caster, spell, slot)
    if not parts:
        return 0.0
    return _expected_parts(
        parts,
        actor,
        dc=int(caster.spell_save_dc or 0),
        save_bonus=_save_bonus(actor, spell.dc_ability),
        success=spell.dc_success,
    )


def _hazards(encounter, actor):
    """Unpaid emanation ticks. A caster this creature already saved against this turn is omitted."""
    rows = []
    for zone in hostile_zones(encounter, actor):
        if zone.saved:
            continue
        amount = _tick_amount(actor, zone.caster, zone.spell)
        if amount <= 0:
            continue
        rows.append((set(zone.cells), amount))
    return rows


def _covered(actor, square, hazards) -> bool:
    body = footprint(tuple(square), actor.span)
    return any(body & cells for cells, _amount in hazards)


def _square_tick(actor, origin, dest, hazards) -> float:
    """Damage the engine would roll for standing on ``dest`` or for the straight path into it."""
    total = 0.0
    start = tuple(origin)
    finish = tuple(dest)
    for cells, amount in hazards:
        if path_enters(actor, start, finish, cells):
            total += amount
            continue
        if (footprint(start, actor.span) & cells) and (footprint(finish, actor.span) & cells):
            total += amount
    return total


def _reaches_from(actor, square, foe, option) -> bool:
    distance = space_between(square, actor, foe)
    if distance <= 0:
        return False
    feet = distance * 5
    if option.reach_feet is not None and feet <= option.reach_feet:
        return True
    return option.range_feet is not None and feet <= option.range_feet


def _option_ready(actor, option) -> bool:
    if option.name in actor.recharging:
        return False
    if option.uses is not None and actor.uses_left.get(option.name, 0) <= 0:
        return False
    return True


def _lowest_slot(actor, spell):
    if spell.level == 0:
        return 0
    levels = [level for level, count in sorted(actor.slots.items()) if count > 0 and level >= spell.level]
    if not levels:
        free_name = (actor.traits.get("free_spells") or {}).get(spell.index)
        if free_name and int(actor.uses_left.get(free_name, 0)) > 0:
            return spell.level
        return None
    return levels[0]


def _spell_reaches(actor, square, foe, spell) -> bool:
    if spell.area_type or spell.range_mode == "self":
        return False
    distance = space_between(square, actor, foe)
    if spell.range_mode == "touch":
        return distance <= 1
    return distance > 0 and distance * 5 <= spell_range_feet(actor, spell)


def _weapon_damage(actor, square, foe) -> float:
    """One fresh action's best weapon or single-target spell against ``foe`` from ``square``."""
    best = 0.0
    if actor.multiattack:
        bundle = 0.0
        reached = False
        for name, count in actor.multiattack:
            option = actor.attack_named(name)
            if option is None or option.resolution != "attack" or not _option_ready(actor, option):
                continue
            if not _reaches_from(actor, square, foe, option):
                continue
            reached = True
            bundle += count * _expected_parts(_parts_from_option(option), foe, bonus=option.attack_bonus or 0)
        if reached:
            best = bundle
    for option in actor.attacks:
        if option.activation != "action" or not _option_ready(actor, option):
            continue
        if option.resolution == "attack" and _reaches_from(actor, square, foe, option):
            best = max(best, _expected_parts(_parts_from_option(option), foe, bonus=option.attack_bonus or 0))
        elif option.resolution == "save" and _reaches_from(actor, square, foe, option):
            best = max(best, _expected_parts(
                _parts_from_option(option),
                foe,
                dc=option.dc_value or 10,
                save_bonus=_save_bonus(foe, option.dc_ability),
                success=option.dc_success,
            ))
    for spell in actor.spells:
        if spell.activation != "action" or spell.kind in {"heal", "excluded"} or spell.index in EMANATION_INDEXES:
            continue
        slot = _lowest_slot(actor, spell)
        if slot is None or not _spell_reaches(actor, square, foe, spell):
            continue
        parts = spell_parts(actor, spell, slot)
        if spell.kind == "attack":
            amount = _expected_parts(parts, foe, bonus=actor.spell_attack_bonus)
        elif spell.kind == "save":
            amount = _expected_parts(
                parts,
                foe,
                dc=actor.spell_save_dc,
                save_bonus=_save_bonus(foe, spell.dc_ability),
                success=spell.dc_success,
            )
        else:
            amount = _expected_parts(parts, foe)
        best = max(best, amount)
    return best


def _best_fresh(actor, square, foes):
    best = 0.0
    chosen = None
    for foe in foes:
        damage = _weapon_damage(actor, square, foe)
        if damage > best:
            best = damage
            chosen = foe
    return best, chosen


def _landed(strike: float, tick: float, hp: int, pay_before: bool, cap: int) -> int:
    """How many attacks land before the ticks drop this creature.

    Entry damage is paid before the hit. End-of-turn damage is paid after it.
    """
    if strike <= 0 or tick <= 0:
        return 0
    if pay_before and tick >= hp:
        return 0
    remaining = hp - tick
    extra = 0 if remaining <= 0 else int(remaining // tick)
    return min(1 + extra, cap)


def _worth(strike: float, tick: float, hp: int, foe_hp: int, pay_before: bool, cap: int) -> bool:
    if tick <= 0:
        return True
    count = _landed(strike, tick, hp, pay_before, cap)
    if count <= 0:
        return False
    reward = count * strike
    return reward >= count * tick or (foe_hp > 0 and reward >= foe_hp)


def _nearest_move(actor, moves):
    origin = tuple(actor.position)
    return min(moves, key=lambda action: (chebyshev(tuple(action.destination), origin), tuple(action.destination)))


def _can_return(encounter, actor, square, foe) -> bool:
    """Next turn's full speed can reach a square that still hits ``foe``."""
    anchors = set(_reachable(encounter, actor, turn_speed(actor), origin=tuple(square)))
    anchors.add(tuple(square))
    return any(_weapon_damage(actor, anchor, foe) > 0 for anchor in anchors)


def _exits(actor, moves, hazards):
    origin = tuple(actor.position)
    found = []
    for action in moves:
        if action.destination is None:
            continue
        dest = tuple(action.destination)
        if _square_tick(actor, origin, dest, hazards) == 0 and not _covered(actor, dest, hazards):
            found.append(action)
    return found


def _attack_reach(creature) -> int:
    reaches = [
        int(option.reach_feet)
        for option in creature.attacks
        if option.resolution == "attack" and option.reach_feet
    ]
    return max(reaches) if reaches else 0


def _opportunity_weapon(reactor, gap: int):
    """Highest-bonus melee attack that still reaches at ``gap``."""
    best = None
    best_bonus = None
    for option in reactor.attacks:
        if option.resolution != "attack" or not option.reach_feet:
            continue
        if gap <= 0 or gap * 5 > int(option.reach_feet):
            continue
        bonus = int(option.attack_bonus or 0)
        if best is None or bonus > best_bonus:
            best = option
            best_bonus = bonus
    return best


def _can_swing(reactor) -> bool:
    if reactor.hp <= 0 or not reactor.reaction_available:
        return False
    return not any(condition in reactor.conditions for condition in ("unconscious", "stunned"))


def _opportunity_cost(encounter, actor, dest, *, disengaged: bool | None = None) -> float:
    """Expected damage from one melee attack each time a reach is left."""
    if disengaged is None:
        disengaged = bool(actor.traits.get("disengaged"))
    origin = tuple(actor.position)
    squares = step_path(origin, tuple(dest))
    if not squares:
        return 0.0
    spent: set[str] = set()
    total = 0.0
    previous = origin
    for square in squares:
        for reactor in encounter.combatants:
            if reactor.id == actor.id or reactor.side == actor.side or reactor.id in spent:
                continue
            if not _can_swing(reactor):
                continue
            if any(mark.get("no_opportunity") for mark in marks_of(reactor)):
                continue
            reach = _attack_reach(reactor)
            if reach <= 0:
                continue
            old = space_gap(tuple(reactor.position), reactor.span, previous, actor.span)
            new = space_gap(tuple(reactor.position), reactor.span, square, actor.span)
            if not (old > 0 and old * 5 <= reach and new * 5 > reach):
                continue
            spent.add(reactor.id)
            if disengaged and not reactor.traits.get("sentinel"):
                continue
            option = _opportunity_weapon(reactor, old)
            if option is None:
                continue
            total += _expected_parts(_parts_from_option(option), actor, bonus=int(option.attack_bonus or 0))
        previous = square
    return total


def _retreat(encounter, actor, moves, hazards, foes, *, disengaged: bool | None = None):
    origin = tuple(actor.position)
    if not _covered(actor, origin, hazards):
        return None
    tick = _square_tick(actor, origin, origin, hazards)
    exits = _exits(actor, moves, hazards)
    if tick <= 0 or not exits:
        return None
    strike, foe = _best_fresh(actor, origin, foes)
    if actor.action_available and strike > 0:
        return None
    # Leaving is worth it only when the tick avoided is greater than the swing.
    affordable = [
        action
        for action in exits
        if _opportunity_cost(encounter, actor, action.destination, disengaged=disengaged) < tick
    ]
    cap = int(encounter.round_cap or 15)
    commit = _worth(strike, tick, actor.hp, foe.hp if foe else 0, False, cap)
    if foe is not None and strike > 0:
        returning = [action for action in affordable if _can_return(encounter, actor, action.destination, foe)]
        if returning:
            return _nearest_move(actor, returning)
    if not commit and affordable:
        return _nearest_move(actor, affordable)
    return None


def _closer_step(encounter, actor, moves, foes):
    current = min(actor.gap_to(foe) for foe in foes)

    def closest(action) -> int:
        return min(space_between(action.destination, actor, foe) for foe in foes)

    closers = [action for action in moves if action.destination is not None and closest(action) < current]
    if not closers:
        return None
    here, _ignored = _best_fresh(actor, tuple(actor.position), foes)
    for action in sorted(closers, key=closest):
        cost = _opportunity_cost(encounter, actor, action.destination)
        if cost <= 0:
            return action
        there, _foe = _best_fresh(actor, action.destination, foes)
        if there - here > cost:
            return action
    return None


def _advance(encounter, actor, moves, hazards, foes):
    current = min(actor.gap_to(foe) for foe in foes)

    def closest(action) -> int:
        return min(space_between(action.destination, actor, foe) for foe in foes)

    closers = [action for action in moves if action.destination is not None and closest(action) < current]
    if not closers:
        return None
    origin = tuple(actor.position)
    already, _ignored = _best_fresh(actor, origin, foes)
    cap = int(encounter.round_cap or 15)
    for action in sorted(closers, key=closest):
        tick = _square_tick(actor, origin, action.destination, hazards)
        cost = _opportunity_cost(encounter, actor, action.destination)
        there, foe = _best_fresh(actor, action.destination, foes)
        if cost > 0 and there - already <= cost:
            continue
        if tick <= 0:
            return action
        # A step that does not improve the hit is not worth a tick.
        if there > already and _worth(there, tick, actor.hp, foe.hp if foe else 0, True, cap):
            return action
    return None


def _needs_dash(encounter, actor, hazards) -> bool:
    if not actor.action_available:
        return False
    origin = tuple(actor.position)
    tick = _square_tick(actor, origin, origin, hazards)
    if tick < actor.hp or not _covered(actor, origin, hazards):
        return False

    def clears(squares) -> bool:
        return any(
            _square_tick(actor, origin, square, hazards) == 0 and not _covered(actor, square, hazards)
            for square in squares
        )

    if clears(_reachable(encounter, actor, movement_budget(actor))):
        return False
    gained = turn_speed(actor)
    if actor.traits.get("charger"):
        gained += 10
    return clears(_reachable(encounter, actor, movement_budget(actor) + gained))


def _dash_leave_provokes(encounter, actor, hazards) -> bool:
    origin = tuple(actor.position)
    gained = turn_speed(actor)
    if actor.traits.get("charger"):
        gained += 10
    for square in _reachable(encounter, actor, movement_budget(actor) + gained):
        if _square_tick(actor, origin, square, hazards) != 0 or _covered(actor, square, hazards):
            continue
        if _opportunity_cost(encounter, actor, square) > 0:
            return True
    return False


def heuristic_choice(encounter, actor):
    """One choice. The caller repeats until End turn: action, else move, else dodge, else bonus."""
    action, _reason = heuristic_advice(encounter, actor)
    return action


def heuristic_advice(encounter, actor):
    """The heuristic's action and the branch that chose it. The action matches ``heuristic_choice``."""
    actions = encounter.legal_actions(actor, automated=True)
    profile = actor.traits.get("monk") if isinstance(actor.traits.get("monk"), dict) else {}
    metabolism = str((profile.get("metabolism") or {}).get("name") or "")
    if metabolism:
        found = next((item for item in actions if item.attack_name == metabolism), None)
        if found is not None:
            return found, "Uncanny Metabolism restores Focus and hit points."
    if not actor.traits.get("rage"):
        rage = next((item for item in actions if item.attack_name == "Rage"), None)
        if rage is not None:
            return rage, "Rage is available."
    if not actor.traits.get("reckless"):
        reckless = next((item for item in actions if item.attack_name == "Reckless Attack"), None)
        if reckless is not None:
            return reckless, "Reckless Attack is available."
    surge = next((item for item in actions if item.attack_name == "Action Surge"), None)
    if surge is not None and any(item.budget == "action" and item.kind == "attack" for item in actions):
        return surge, "Action Surge grants another action."
    hazards = _hazards(encounter, actor)
    if hazards and _needs_dash(encounter, actor, hazards):
        if _dash_leave_provokes(encounter, actor, hazards):
            focused = next((item for item in actions if item.attack_name == "Focused Step of the Wind"), None)
            if focused is not None:
                return focused, "Focused Step of the Wind is Disengage and Dash."
        dash = next((item for item in actions if item.kind == "dash" and item.budget == "bonus"), None)
        if dash is None:
            dash = next((item for item in actions if item.kind == "dash"), None)
        if dash is not None:
            return dash, "Dash clears a lethal area that a normal move does not."
    stand = next((item for item in actions if item.kind == "stand"), None)
    if stand is not None:
        return stand, "Stand before attacking."
    order = next((item for item in actions if item.kind == "order"), None)
    if order is not None:
        return order, "The order is the only action."
    action_pool = [
        item for item in actions if item.budget == "action" and (_offense(item) or item.kind in {"escape", "shake"})
    ]
    chosen = _best_offensive(encounter, actor, action_pool)
    if chosen is not None:
        return chosen, _pressure_reason(encounter, actor, action_pool, chosen)
    foes = [creature for creature in encounter.combatants if creature.hp > 0 and creature.side != actor.side]
    moves = [item for item in actions if item.kind == "move" and item.destination]
    if foes and moves:
        if hazards:
            retreat = _retreat(encounter, actor, moves, hazards, foes)
            if retreat is not None:
                bonus_pool = [item for item in actions if item.budget == "bonus" and _offense(item)]
                bonus = _best_offensive(encounter, actor, bonus_pool)
                if bonus is not None:
                    return bonus, "Bonus attack before leaving the area."
                return retreat, "Leave the area; the tick costs more than the opportunity attack."
            escaped = _retreat(encounter, actor, moves, hazards, foes, disengaged=True)
            if escaped is not None and not actor.traits.get("disengaged"):
                bonus_disengage = next(
                    (item for item in actions if item.kind == "disengage" and item.budget == "bonus"),
                    None,
                )
                if bonus_disengage is not None:
                    return bonus_disengage, "Bonus Disengage makes leaving safe."
                origin = tuple(actor.position)
                tick = _square_tick(actor, origin, origin, hazards)
                strike, _foe = _best_fresh(actor, origin, foes)
                raw = _opportunity_cost(encounter, actor, escaped.destination, disengaged=False)
                if actor.action_available and strike <= 0 and tick >= actor.hp and raw >= actor.hp:
                    action_disengage = next(
                        (item for item in actions if item.kind == "disengage" and item.budget == "action"),
                        None,
                    )
                    if action_disengage is not None:
                        return action_disengage, "Disengage makes leaving a lethal area safe."
            advance = _advance(encounter, actor, moves, hazards, foes)
            if advance is not None:
                return advance, "Step closer; the new attack is worth the tick and the opportunity attack."
        else:
            step = _closer_step(encounter, actor, moves, foes)
            if step is not None:
                return step, "Step closer; the new attack is worth more than the opportunity attack."
    dodge = next((item for item in actions if item.kind == "dodge"), None)
    if dodge is not None:
        threatened = any(
            creature.hp > 0 and creature.side != actor.side and actor.gap_to(creature) <= 1
            for creature in encounter.combatants
        )
        if threatened:
            focused = next((item for item in actions if item.attack_name == "Focused Patient Defense"), None)
            if focused is not None:
                return focused, "Focused Patient Defense is Disengage and Dodge."
        return dodge, "Dodge."
    if actor.max_hp and actor.hp / actor.max_hp <= 0.35:
        wind = next((item for item in actions if item.attack_name == "Second Wind"), None)
        if wind is not None:
            return wind, "Heal; hit points are at or below 35%."
        for heal in actor.traits.get("bonus_heals") or []:
            name = str(heal.get("name") or "")
            found = next((item for item in actions if name and item.attack_name == name), None)
            if found is not None:
                return found, "Heal; hit points are at or below 35%."
    flurry = next((item for item in actions if item.attack_name == "Flurry of Blows"), None)
    if flurry is not None:
        return flurry, "Flurry of Blows spends 1 Focus for two unarmed strikes."
    bonus_pool = [item for item in actions if item.budget == "bonus" and _offense(item)]
    chosen = _best_offensive(encounter, actor, bonus_pool)
    if chosen is not None:
        return chosen, "Highest expected bonus damage."
    end = next((item for item in actions if item.kind == "end"), None)
    if end is not None:
        return end, "End turn."
    return actions[0], "End turn."


def _round_number(value: float) -> float:
    return round(float(value), 1)


def _healing_amount(actor, action) -> float | None:
    if action.kind != "spell":
        return None
    spell = actor.spell_named(action.spell_index)
    if spell is None or spell.kind != "heal":
        return None
    expression = spell_heal_expression(
        spell,
        int(action.slot_level or spell.level or 0),
        int(actor.spellcasting_mod or 0),
    )
    if expression is None:
        return 0.0
    return average(expression)


def _action_row(encounter, actor, action) -> dict:
    row = {
        "id": action.id,
        "kind": action.kind,
        "label": action.label,
        "expected": None,
        "fresh": None,
        "opportunity": None,
        "tick": None,
        "healing": None,
    }
    healing = _healing_amount(actor, action)
    if healing is not None:
        row["healing"] = _round_number(healing)
        return row
    if action.kind == "move" and action.destination is not None:
        foes = [creature for creature in encounter.combatants if creature.hp > 0 and creature.side != actor.side]
        fresh, _foe = _best_fresh(actor, action.destination, foes)
        row["fresh"] = _round_number(fresh)
        row["opportunity"] = _round_number(_opportunity_cost(encounter, actor, action.destination))
        row["tick"] = _round_number(_square_tick(actor, tuple(actor.position), action.destination, _hazards(encounter, actor)))
        return row
    if _offense(action) or action.kind in {"stand", "escape", "shake", "order"}:
        row["expected"] = _round_number(action_score(encounter, actor, action))
        return row
    row["expected"] = 0
    return row


def _decision_menu(encounter, actor, actions, pick):
    selected = []
    seen = set()

    def add(action) -> None:
        if action.id in seen:
            return
        seen.add(action.id)
        selected.append(action)

    add(pick)
    for action in actions:
        if _healing_amount(actor, action) is not None or _offense(action) or action.kind in {"stand", "escape", "shake", "order"}:
            add(action)
    for action in actions:
        if action.kind == "move" and action.destination is not None:
            add(action)
    for action in actions:
        if action.kind == "end":
            add(action)
    return selected


def _needs_model(pick, menu) -> bool:
    if pick.attack_name in {
        "Rage",
        "Reckless Attack",
        "Action Surge",
        "Flurry of Blows",
        "Patient Defense",
        "Step of the Wind",
        "Focused Patient Defense",
        "Focused Step of the Wind",
        "Uncanny Metabolism",
    }:
        return False
    return any(action.id != pick.id and action.kind != "end" for action in menu)


def decision_document(encounter, actor) -> dict:
    """Facts for one model call. Every action row has the same columns."""
    actions = encounter.legal_actions(actor, automated=True)
    pick, reason = heuristic_advice(encounter, actor)
    menu = _decision_menu(encounter, actor, actions, pick)
    actor_row = {
        "name": actor.name,
        "hp": actor.hp,
        "max_hp": actor.max_hp,
        "ac": actor.ac,
        "x": int(actor.position[0]),
        "y": int(actor.position[1]),
        "movement": actor.movement_left if actor.movement_left is not None else actor.speed,
        "action_ready": bool(actor.action_available),
        "bonus_ready": bool(actor.bonus_available),
        "conditions": ", ".join(condition_labels(actor)),
        "concentrating": actor.concentrating_on or "",
    }
    placed = actor.traits.get("spiritual_weapon")
    if actor.concentrating_on == "spiritual-weapon" and isinstance(placed, dict):
        actor_row["weapon_x"] = int(placed.get("x", 0))
        actor_row["weapon_y"] = int(placed.get("y", 0))
    living = [creature for creature in encounter.combatants if creature.hp > 0]
    return {
        "actor": actor_row,
        "combatants": [
            {
                "name": creature.name,
                "side": creature.side,
                "hp": creature.hp,
                "max_hp": creature.max_hp,
                "ac": creature.ac,
                "x": int(creature.position[0]),
                "y": int(creature.position[1]),
                "conditions": ", ".join(condition_labels(creature)),
                "concentrating": creature.concentrating_on or "",
            }
            for creature in living
        ],
        "actions": [_action_row(encounter, actor, action) for action in menu],
        "pick": pick.id,
        "reason": reason,
        "_menu": menu,
        "_call": _needs_model(pick, menu),
    }


def render_prompt(encounter, actor, actions) -> str:
    """TOON document for the model. ``actions`` is accepted so older callers keep working."""
    del actions
    document = decision_document(encounter, actor)
    document.pop("_menu")
    document.pop("_call")
    return encode(document, delimiter="\t")


def parse_choice(text: str, count: int) -> tuple[int, str] | None:
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    action_id = payload.get("action_id")
    if isinstance(action_id, bool) or not isinstance(action_id, int):
        return None
    if not 0 <= action_id < count:
        return None
    return action_id, str(payload.get("reason") or "")[:240]


TURN_CHOICE_CAP = 6


def play_turn(encounter, policy: str, client: LLMClient | None = None) -> int:
    """Resolve the current creature's turn. Returns how many choices were made.

    The model still sees one action id at a time, because a move changes the
    legal list. Six choices is the cap; anything left is then ended. A reaction
    from the other side pauses the turn. Call this again after the player answers.
    Creatures on this turn's side answer their own reactions with the heuristic.
    """
    if encounter.pending:
        return 0
    actor_id = encounter.current().id
    encounter.auto_side = encounter.current().side
    encounter.auto_policy = policy
    calls = 0
    while calls < TURN_CHOICE_CAP and not encounter.over and not encounter.pending:
        encounter.begin_turn()
        if encounter.over or encounter.pending or encounter.current().id != actor_id:
            break
        actor = encounter.current()
        action, source, reason = choose_action(encounter, actor, policy, client if policy == "llm" else None)
        calls += 1
        encounter.apply(action, source, reason)
        if encounter.pending or encounter.over or action.kind == "end" or encounter.current().id != actor_id:
            break
    if (
        not encounter.over
        and not encounter.pending
        and encounter.started
        and encounter.current().id == actor_id
    ):
        ending = next((item for item in encounter.legal_actions(automated=False) if item.kind == "end"), None)
        if ending is not None:
            encounter.apply(ending, "heuristic", "")
    if not encounter.pending:
        encounter.auto_side = None
        encounter.auto_policy = None
    return calls


def choose_action(encounter, actor, policy: str, client: LLMClient | None = None):
    """Return ``(action, source, reason)``. Source is heuristic, llm, or fallback."""
    if policy != "llm":
        return heuristic_choice(encounter, actor), "heuristic", ""
    document = decision_document(encounter, actor)
    menu = document.pop("_menu")
    call_model = document.pop("_call")
    pick = next(action for action in menu if action.id == document["pick"])
    if not call_model:
        return pick, "heuristic", str(document["reason"])
    if client is None:
        client = LLMClient()
    prompt = encode(document, delimiter="\t")
    allowed = [action.id for action in menu]
    last_error = ""
    for _ in range(2):
        try:
            text = client.complete(ACTION_SYSTEM, prompt, allowed)
        except LLMError:
            raise
        except Exception as exc:
            last_error = str(exc)[:200]
            continue
        parsed = parse_choice(text, max(allowed) + 1)
        if parsed and parsed[0] in allowed:
            action_id, reason = parsed
            found = next(action for action in menu if action.id == action_id)
            return found, "llm", reason
        last_error = "The model did not return a legal action id."
    return pick, "fallback", last_error


def write_report(summary_text: str, client: LLMClient | None = None) -> str:
    if client is None:
        client = LLMClient(report=True)
    return client.complete(REPORT_SYSTEM, summary_text)
