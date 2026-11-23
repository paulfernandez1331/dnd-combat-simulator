"""Reactions loaded from a stat block or a character sheet.

A reaction is offered after the triggering roll and before the damage or the
spell is applied. The heuristic spends one only when the numbers already on
the table say it can change the result.
"""

from __future__ import annotations

import re

from src.actions import _slot_levels, spell_range_feet
from src.bestiary import ReactionRecord, ability_mod
from src.dice import roll_dice
from src.feats import pole_weapon
from src.rules import armor_class, d20, opposed_ac

_DIE = re.compile(r"d(\d+)", re.I)
_POOL = "Bardic Inspiration"


def die_size(expression: str) -> int:
    match = _DIE.search(expression or "")
    return int(match.group(1)) if match else 0


def lowest_slot(creature, minimum: int) -> int | None:
    levels = []
    for level, count in creature.slots.items():
        try:
            number = int(level)
            left = int(count)
        except (TypeError, ValueError):
            continue
        if number >= minimum and left > 0:
            levels.append(number)
    return min(levels) if levels else None


def asks_player(encounter, reactor) -> bool:
    """Hand play asks. A batch, or the side whose turn is being automated, does not."""
    if encounter.reaction_policy == "heuristic":
        return False
    if encounter.auto_side and reactor.side == encounter.auto_side:
        return False
    return True


def _ready(encounter, reactor) -> bool:
    if reactor.hp <= 0 or not reactor.reaction_available:
        return False
    return not any(condition in reactor.conditions for condition in ("stunned", "unconscious"))


def _can_cast_another(encounter, reactor) -> bool:
    """Shield and Counterspell are leveled spells. Cutting Words is not."""
    if not reactor.traits.get("leveled_cast"):
        return True
    try:
        return encounter.current().id != reactor.id
    except (IndexError, KeyError):
        return True


def _within(left, right, feet: int) -> bool:
    return left.gap_to(right) * 5 <= feet


def cutting(reactor) -> dict | None:
    feature = reactor.traits.get("cutting_words")
    if not isinstance(feature, dict):
        return None
    if int(reactor.uses_left.get(feature.get("pool") or _POOL, 0)) <= 0:
        return None
    return feature


def _free_casting(reactor, index: str) -> str:
    mapped = (reactor.traits.get("free_spells") or {}).get(index)
    if mapped and int(reactor.uses_left.get(mapped, 0)) > 0:
        return str(mapped)
    return ""


def can_shield(encounter, reactor, target) -> bool:
    if reactor.id != target.id or target.traits.get("shield_ac"):
        return False
    if not _ready(encounter, reactor) or not _can_cast_another(encounter, reactor):
        return False
    spell = reactor.spell_named("shield")
    if spell is None or spell.kind == "excluded":
        return False
    if _free_casting(reactor, "shield"):
        return True
    return lowest_slot(reactor, max(1, int(spell.level or 1))) is not None


def can_cut(encounter, reactor, roller, *, critical: bool) -> bool:
    if critical or reactor.side == roller.side:
        return False
    feature = cutting(reactor)
    if feature is None or not _ready(encounter, reactor):
        return False
    return _within(reactor, roller, int(feature.get("range") or 60))


def can_counter(encounter, reactor, caster, spell) -> bool:
    if reactor.id == caster.id or reactor.side == caster.side:
        return False
    if spell is None or not spell.has_components:
        return False
    if not _ready(encounter, reactor) or not _can_cast_another(encounter, reactor):
        return False
    if not _within(reactor, caster, 60):
        return False
    known = reactor.spell_named("counterspell")
    if known is None or known.kind == "excluded":
        return False
    return lowest_slot(reactor, max(3, int(known.level or 3))) is not None


def _reactions(reactor) -> list:
    return list(getattr(reactor, "reactions", []) or [])


def _choice_id(record, index: int) -> str:
    if record.effect == "spell" and record.spell_index in {"shield", "counterspell"}:
        return record.spell_index
    if record.effect == "subtract" and record.name.lower() == "cutting words":
        return "cutting_words"
    return f"{record.effect}:{index}:{record.name}"


def _holds_weapon(creature) -> bool:
    for attack in creature.attacks:
        if getattr(attack, "resolution", "") == "attack" and attack.reach_feet:
            return True
    return False


def _spell_record(reactor, index: str):
    for record in _reactions(reactor):
        if record.effect != "spell" or record.spell_index != index:
            continue
        if record.pool and int(reactor.uses_left.get(record.pool, 0)) <= 0:
            return None
        return record
    return None


def _can_record_shield(encounter, reactor, target, record) -> bool:
    if target is None or reactor.id != target.id or target.traits.get("shield_ac"):
        return False
    if not _ready(encounter, reactor) or not _can_cast_another(encounter, reactor):
        return False
    if record.pool and int(reactor.uses_left.get(record.pool, 0)) <= 0:
        return False
    return True


def _can_record_counter(encounter, reactor, caster, spell, record) -> bool:
    if reactor.id == caster.id or reactor.side == caster.side:
        return False
    if spell is None or not spell.has_components:
        return False
    if not _ready(encounter, reactor) or not _can_cast_another(encounter, reactor):
        return False
    if not _within(reactor, caster, int(record.range_feet or 60)):
        return False
    if record.pool and int(reactor.uses_left.get(record.pool, 0)) <= 0:
        return False
    return True


def _can_ac(encounter, reactor, target, window: dict, record) -> bool:
    if target is None or reactor.id != target.id:
        return False
    if record.melee_only and not window.get("melee"):
        return False
    if record.needs_weapon and not _holds_weapon(reactor):
        return False
    return _ready(encounter, reactor)


def _can_subtract(encounter, reactor, roller, record, *, critical: bool) -> bool:
    if critical or reactor.side == roller.side:
        return False
    if not record.pool or int(reactor.uses_left.get(record.pool, 0)) <= 0:
        return False
    if not _ready(encounter, reactor):
        return False
    return _within(reactor, roller, int(record.range_feet or 60))


def _subtract_offer(record, index: int) -> dict:
    if record.name.lower() == "cutting words":
        die = record.die or "1d6"
        return {"id": "cutting_words", "label": f"Cutting Words ({die})", "die": die, "pool": record.pool}
    return {"id": _choice_id(record, index), "label": record.name, "die": record.die, "pool": record.pool}


def _subtract_offers(encounter, reactor, roller, *, critical: bool) -> list[dict]:
    reactions = _reactions(reactor)
    offers = []
    for index, record in enumerate(reactions):
        if record.effect != "subtract":
            continue
        if _can_subtract(encounter, reactor, roller, record, critical=critical):
            offers.append(_subtract_offer(record, index))
    if any(record.effect == "subtract" for record in reactions):
        return offers
    if can_cut(encounter, reactor, roller, critical=critical):
        feature = cutting(reactor) or {}
        die = str(feature.get("die") or "1d6")
        offers.append({"id": "cutting_words", "label": f"Cutting Words ({die})", "die": die, "pool": feature.get("pool")})
    return offers


def _subtract_record(reactor, choice: str):
    for index, record in enumerate(_reactions(reactor)):
        if record.effect == "subtract" and _choice_id(record, index) == choice:
            return record
    if choice != "cutting_words":
        return None
    feature = reactor.traits.get("cutting_words")
    if not isinstance(feature, dict):
        return None
    return ReactionRecord(
        name="Cutting Words",
        effect="subtract",
        die=str(feature.get("die") or "1d6"),
        range_feet=int(feature.get("range") or 60),
        pool=str(feature.get("pool") or _POOL),
    )


def _record_for(reactor, choice: str):
    for index, record in enumerate(_reactions(reactor)):
        if _choice_id(record, index) == choice:
            return record
    return None


def _spend_known_or_pool(encounter, reactor, index: str, minimum: int) -> str:
    """Spend a free feature casting, a prepared slot, or one listed use."""
    known = reactor.spell_named(index)
    prepared = known is not None and known.kind != "excluded"
    if prepared and _free_casting(reactor, index):
        mapped = (reactor.traits.get("free_spells") or {})[index]
        reactor.uses_left[mapped] = int(reactor.uses_left.get(mapped, 0)) - 1
        reactor.reaction_available = False
        try:
            if encounter.current().id == reactor.id and int(known.level or 0) > 0:
                reactor.traits["leveled_cast"] = True
        except (IndexError, KeyError):
            pass
        return "a free casting"
    level = lowest_slot(reactor, minimum) if prepared else None
    record = _spell_record(reactor, index)
    if prepared and level is not None:
        _spend_spell(encounter, reactor, level)
        return ""
    if record is not None:
        if record.pool:
            reactor.uses_left[record.pool] = int(reactor.uses_left.get(record.pool, 0)) - 1
        reactor.reaction_available = False
        try:
            if encounter.current().id == reactor.id:
                reactor.traits["leveled_cast"] = True
        except (IndexError, KeyError):
            pass
        return record.pool or ""
    if level is not None:
        _spend_spell(encounter, reactor, level)
    return ""


def candidate_ids(encounter, target) -> list[str]:
    ids = []
    if target is not None and target.hp > 0:
        ids.append(target.id)
    for creature_id in encounter.order:
        if creature_id in ids:
            continue
        creature = encounter.by_id(creature_id)
        if creature.hp > 0:
            ids.append(creature_id)
    return ids


def _proficiency(creature) -> int:
    return int(creature.traits.get("proficiency") or 2)


def _holding_arms(creature) -> bool:
    if creature.traits.get("has_shield"):
        return True
    return any(attack.reach_feet or attack.range_feet for attack in creature.attacks)


def _finesse_in_hand(creature) -> bool:
    return any(attack.finesse and attack.reach_feet for attack in creature.attacks)


def _melee_option(actor, target):
    best = None
    for option in actor.attacks:
        if not option.reach_feet or option.resolution != "attack":
            continue
        if actor.gap_to(target) * 5 > int(option.reach_feet):
            continue
        bonus = int(option.attack_bonus or 0)
        if best is None or bonus > int(best.attack_bonus or 0):
            best = option
    return best


def _pole_option(actor, target):
    for option in actor.attacks:
        if not pole_weapon(option):
            continue
        if actor.gap_to(target) * 5 <= int(option.reach_feet or 0):
            return option
    return None


def _war_offers(reactor, mover) -> list[dict]:
    offers = []
    for spell in reactor.spells:
        if spell.activation != "action" or spell.area_type or spell.kind not in {"attack", "save", "auto"}:
            continue
        if spell.exclude_reason or spell.kind == "excluded":
            continue
        distance = reactor.gap_to(mover)
        if spell.range_mode == "self" or spell.range_mode == "touch" and distance > 1:
            continue
        if spell.range_mode not in {"self", "touch"} and not (distance > 0 and distance * 5 <= spell_range_feet(reactor, spell)):
            continue
        for slot in _slot_levels(reactor, spell, False):
            label = spell.name if slot == 0 else f"{spell.name} (level {slot})"
            offers.append({
                "id": f"war:{spell.index}:{slot}",
                "label": label,
                "slot": slot,
                "damaging": True,
                "spell_index": spell.index,
            })
    return offers


def _low_die_can_miss(window, target) -> bool:
    bonus = int(window.get("bonus") or 0)
    ac = opposed_ac(target, melee=bool(window.get("melee")), ac_bonus=int(window.get("ac_bonus") or 0))
    return 1 + bonus < ac


def _feat_offers(encounter, reactor, actor, target, window, phase) -> list[dict]:
    offers = []
    if phase == "luck" and reactor.id == actor.id and int(reactor.uses_left.get("Luck", 0)) > 0:
        offers.append({"id": "lucky", "label": "Lucky"})
    if phase == "save" and target is not None and reactor.id == target.id:
        ability = str(window.get("ability") or "").lower()
        if (
            not window.get("saved")
            and ability in {"intelligence", "wisdom", "charisma"}
            and int(reactor.uses_left.get("Guarded Mind", 0)) > 0
        ):
            offers.append({"id": "guarded_mind", "label": "Guarded Mind"})
        if (
            window.get("saved")
            and ability == "dexterity"
            and window.get("success") == "half"
            and reactor.traits.get("shield_master")
            and reactor.traits.get("has_shield")
            and _ready(encounter, reactor)
        ):
            offers.append({"id": "shield_interpose", "label": "Interpose Shield"})
        if (
            not window.get("saved")
            and int(reactor.uses_left.get("Luck", 0)) > 0
            and int(window.get("mode") or 0) <= 0
        ):
            offers.append({"id": "lucky", "label": "Lucky"})
    if phase == "hit" and target is not None and reactor.id == target.id and _ready(encounter, reactor):
        if (
            window.get("melee")
            and reactor.traits.get("defensive_duelist")
            and not reactor.traits.get("duelist_ac")
            and _finesse_in_hand(reactor)
        ):
            offers.append({
                "id": "duelist",
                "label": "Defensive Duelist",
                "bonus": _proficiency(reactor),
            })
        if (
            int(reactor.uses_left.get("Luck", 0)) > 0
            and int(window.get("mode") or 0) >= 0
            and _low_die_can_miss(window, target)
            and not (window.get("critical") and not _low_die_can_miss(window, target))
        ):
            offers.append({"id": "lucky", "label": "Lucky"})
    if phase == "hit" and target is not None and reactor.id != target.id and _ready(encounter, reactor):
        if (
            reactor.traits.get("protection")
            and reactor.traits.get("has_shield")
            and reactor.gap_to(target) <= 1
            and reactor.gap_to(target) > 0
        ):
            offers.append({"id": "protection", "label": "Protection"})
    if phase == "damage" and target is not None and reactor.id != target.id and _ready(encounter, reactor):
        if reactor.traits.get("interception") and _holding_arms(reactor) and 0 < reactor.gap_to(target) <= 1:
            die = str(reactor.traits.get("interception_die") or "1d10")
            offers.append({
                "id": "interception",
                "label": "Interception",
                "die": die,
                "max": die_size(die) + _proficiency(reactor),
            })
    if phase == "move" and _ready(encounter, reactor):
        if reactor.id in set(window.get("entered") or []) and _pole_option(reactor, actor) is not None:
            option = _pole_option(reactor, actor)
            offers.append({"id": "polearm", "label": f"Polearm Master ({option.name})", "weapon": option.name})
        if reactor.id in set(window.get("opportunity") or []):
            option = _melee_option(reactor, actor)
            if option is not None:
                if reactor.traits.get("sentinel"):
                    offers.append({"id": "sentinel", "label": "Sentinel", "weapon": option.name})
                else:
                    offers.append({
                        "id": "opportunity",
                        "label": f"Opportunity attack ({option.name})",
                        "weapon": option.name,
                    })
        if reactor.id in set(window.get("left") or []):
            offers.extend(_war_offers(reactor, actor))
    if phase == "sentinel" and target is not None and _ready(encounter, reactor) and reactor.traits.get("sentinel"):
        option = _melee_option(reactor, target)
        if option is not None:
            offers.append({"id": "sentinel", "label": "Sentinel", "weapon": option.name})
    return offers


def offers_for(encounter, reactor, window: dict, phase: str) -> list[dict]:
    actor = encounter.by_id(window["actor_id"])
    target = encounter.by_id(window["target_id"]) if window.get("target_id") else None
    offers = []
    if phase in {"hit", "missile"} and target is not None and can_shield(encounter, reactor, target):
        level = lowest_slot(reactor, 1)
        offers.append({"id": "shield", "label": f"Shield (level {level} slot, +5 AC)", "bonus": 5})
    elif phase in {"hit", "missile"} and target is not None:
        record = _spell_record(reactor, "shield")
        if record is not None and _can_record_shield(encounter, reactor, target, record):
            offers.append({"id": "shield", "label": record.name, "bonus": 5, "pool": record.pool})
    if phase == "hit":
        for index, record in enumerate(_reactions(reactor)):
            if record.effect == "ac" and _can_ac(encounter, reactor, target, window, record):
                offers.append({"id": _choice_id(record, index), "label": record.name, "bonus": record.bonus})
        offers.extend(_subtract_offers(encounter, reactor, actor, critical=bool(window.get("critical"))))
    if phase == "damage":
        offers.extend(_subtract_offers(encounter, reactor, actor, critical=False))
        deflect = _deflect_offer(reactor, target, window)
        if deflect is not None:
            offers.append(deflect)
    if phase == "counterspell":
        spell = actor.spell_named(window.get("spell_index"))
        if can_counter(encounter, reactor, actor, spell):
            level = lowest_slot(reactor, 3)
            offers.append({"id": "counterspell", "label": f"Counterspell (level {level} slot)"})
        else:
            record = _spell_record(reactor, "counterspell")
            if record is not None and _can_record_counter(encounter, reactor, actor, spell, record):
                offers.append({"id": "counterspell", "label": record.name, "pool": record.pool})
    offers.extend(_feat_offers(encounter, reactor, actor, target, window, phase))
    return offers


def prompt_for(encounter, window: dict, phase: str, reactor) -> str:
    actor = encounter.by_id(window["actor_id"])
    if phase == "luck":
        return f"{reactor.name} can spend a luck point on this roll."
    if phase == "save":
        target = encounter.by_id(window["target_id"])
        return f"{target.name} makes a {window.get('ability')} save against {actor.name}'s {window['label']}."
    if phase == "move":
        leaving = set(window.get("opportunity") or []) | set(window.get("left") or [])
        if reactor.id in leaving:
            return f"{actor.name} leaves {reactor.name}'s reach. {reactor.name} can make an opportunity attack."
        return f"{actor.name} moves. {reactor.name} can react."
    if phase == "sentinel":
        return f"{actor.name} provokes Sentinel. {reactor.name} can attack."
    if phase == "counterspell":
        spell = actor.spell_named(window.get("spell_index"))
        name = spell.name if spell is not None else "a spell"
        return f"{actor.name} casts {name}. {reactor.name} can counter it."
    target = encounter.by_id(window["target_id"])
    if phase == "missile":
        return f"{actor.name}'s Magic Missile targets {target.name}."
    if phase == "damage":
        amount = sum(int(value) for _, value in window.get("rolled_parts") or [])
        return f"{window['label']} deals {amount} damage to {target.name}."
    total = int(window["total"]) - int(window.get("penalty") or 0)
    crit = " Critical hit." if window.get("critical") else ""
    ac = opposed_ac(target, melee=bool(window.get("melee")), ac_bonus=int(window.get("ac_bonus") or 0))
    return f"{actor.name}'s {window['label']} hits {target.name}, {total} vs AC {ac}.{crit}"


def still_hits(window: dict, target) -> bool:
    if int(window["roll"]) == 1:
        return False
    if window.get("critical"):
        return True
    total = int(window["total"]) - int(window.get("penalty") or 0)
    return total >= opposed_ac(target, melee=bool(window.get("melee")), ac_bonus=int(window.get("ac_bonus") or 0))


def _worth_counter(encounter, caster, spell, window: dict) -> bool:
    if spell is None or int(spell.level or 0) <= 0:
        return False
    targets = [encounter.by_id(creature_id) for creature_id in window.get("target_ids") or []]
    if spell.kind == "heal":
        for target in targets:
            if target.side == caster.side and target.max_hp and target.hp / target.max_hp <= 0.35:
                return True
        return False
    if spell.kind in {"attack", "save", "auto"}:
        return any(target.side != caster.side for target in targets)
    return False


def _offer_damage(reactor, target, offer) -> float:
    if offer is None:
        return 0.0
    from src.ai import _expected_parts, _parts_from_option, _save_bonus

    if offer["id"] in {"opportunity", "sentinel", "polearm"}:
        option = reactor.attack_named(offer.get("weapon") or "")
        if option is None:
            return 0.0
        return _expected_parts(_parts_from_option(option), target, bonus=int(option.attack_bonus or 0))
    if not str(offer["id"]).startswith("war:"):
        return 0.0
    spell = reactor.spell_named(offer.get("spell_index"))
    if spell is None:
        return 0.0
    from src.actions import spell_parts

    slot = int(offer.get("slot") or 0)
    parts = spell_parts(reactor, spell, slot)
    if spell.kind == "attack":
        return _expected_parts(parts, target, bonus=reactor.spell_attack_bonus)
    if spell.kind == "save":
        return _expected_parts(
            parts,
            target,
            dc=reactor.spell_save_dc,
            save_bonus=_save_bonus(target, spell.dc_ability),
            success=spell.dc_success,
        )
    return _expected_parts(parts, target)


def _better_war_spell(offer, best) -> bool:
    """Equal damage keeps a cantrip, then the lower slot."""
    if best is None:
        return True
    offer_slot = int(offer.get("slot") or 0)
    best_slot = int(best.get("slot") or 0)
    if offer_slot == 0 and best_slot != 0:
        return True
    if best_slot == 0 and offer_slot != 0:
        return False
    return offer_slot < best_slot


def _best_war_spell(reactor, actor, offers):
    best = None
    best_damage = 0.0
    for offer in offers:
        if not str(offer["id"]).startswith("war:"):
            continue
        damage = _offer_damage(reactor, actor, offer)
        if damage > best_damage or (damage == best_damage and damage > 0 and _better_war_spell(offer, best)):
            best = offer
            best_damage = damage
    return best


_BPS = {"bludgeoning", "piercing", "slashing"}


def _deflect_offer(reactor, target, window: dict):
    if target is None or reactor.id != target.id or not reactor.reaction_available:
        return None
    info = (reactor.traits.get("monk") or {}).get("deflect") if isinstance(reactor.traits.get("monk"), dict) else None
    if not info:
        return None
    parts = window.get("rolled_parts") or []
    if not any(str(kind).lower() in _BPS and int(amount) > 0 for kind, amount in parts):
        return None
    return {"id": "deflect", "label": "Deflect Attack"}


def _offer(offers, offer_id):
    for offer in offers:
        if offer["id"] == offer_id:
            return offer
    return None


def heuristic_choice(encounter, reactor, window: dict, phase: str, offers: list[dict]) -> tuple[str | None, str]:
    """Return the offer id and a short reason. None declines."""
    ids = {offer["id"] for offer in offers}
    if phase == "luck" and "lucky" in ids:
        mode = int(window.get("mode") or 0)
        if mode < 0:
            shown = tuple(window.get("shown") or ())
            higher = max(shown) if shown else int(window["roll"])
            bonus = int(window.get("bonus") or 0)
            ac = int(window.get("ac") or 0)
            crit = higher >= int(window.get("crit_min") or 20) and higher != 1
            if higher != int(window["roll"]) and (crit or higher + bonus >= ac):
                return "lucky", "the higher die can hit"
            return None, ""
        return "lucky", "a second die can turn the miss into a hit"
    if phase == "save" and "guarded_mind" in ids:
        return "guarded_mind", "the failed save can succeed"
    if phase == "save" and "shield_interpose" in ids:
        return "shield_interpose", "the shield can reduce the damage to 0"
    if phase == "save" and "lucky" in ids:
        mode = int(window.get("mode") or 0)
        bonus = int(window.get("bonus") or 0)
        dc = int(window.get("dc") or 0)
        if mode < 0:
            shown = tuple(window.get("shown") or ())
            higher = max(shown) if shown else int(window["roll"])
            if higher != int(window["roll"]) and higher + bonus >= dc:
                return "lucky", "the higher die can succeed"
        elif 20 + bonus >= dc:
            return "lucky", "a second die can succeed"
    if phase == "sentinel" and "sentinel" in ids:
        return "sentinel", "a melee weapon can reach"
    if phase == "move":
        if "polearm" in ids:
            return "polearm", "a creature entered reach"
        mover = encounter.by_id(window["actor_id"])
        weapon = _offer(offers, "sentinel") or _offer(offers, "opportunity")
        weapon_damage = _offer_damage(reactor, mover, weapon) if weapon is not None else 0.0
        spell = _best_war_spell(reactor, mover, offers)
        spell_damage = _offer_damage(reactor, mover, spell) if spell is not None else 0.0
        if spell is not None and spell_damage > weapon_damage and spell_damage > 0:
            return spell["id"], "the spell deals more"
        if weapon is not None and weapon_damage > 0:
            if weapon["id"] == "sentinel":
                return "sentinel", "a melee weapon can reach"
            return "opportunity", "the creature left reach"
        return None, ""
    if phase == "counterspell" and "counterspell" in ids:
        caster = encounter.by_id(window["actor_id"])
        spell = caster.spell_named(window.get("spell_index"))
        if not _worth_counter(encounter, caster, spell, window):
            return None, ""
        bonus = caster.save_bonuses.get("constitution", ability_mod(caster.abilities.get("constitution", 10)))
        dc = int(reactor.spell_save_dc or 0)
        if bonus >= dc - 1:
            return None, ""
        return "counterspell", "the spell would harm the party"
    if "shield" in ids and phase == "missile":
        return "shield", "Magic Missile would deal damage"
    if phase == "hit" and not window.get("critical"):
        target = encounter.by_id(window["target_id"])
        total = int(window["total"]) - int(window.get("penalty") or 0)
        ac = armor_class(target)
        if "shield" in ids and ac <= total < ac + 5:
            return "shield", f"{total} would miss AC {ac + 5}"
        for offer in offers:
            if not str(offer["id"]).startswith("ac:"):
                continue
            bonus = int(offer.get("bonus") or 0)
            if bonus and ac <= total < ac + bonus:
                return offer["id"], f"{total} would miss AC {ac + bonus}"
        if "duelist" in ids:
            duelist = _offer(offers, "duelist")
            bonus = int(duelist.get("bonus") or 0) if duelist else 0
            melee_ac = opposed_ac(target, melee=True, ac_bonus=int(window.get("ac_bonus") or 0))
            if bonus and melee_ac <= total < melee_ac + bonus:
                return "duelist", f"{total} would miss AC {melee_ac + bonus}"
        if "protection" in ids and int(window.get("mode") or 0) >= 0 and _low_die_can_miss(window, target):
            return "protection", "a lower die can miss"
        if "lucky" in ids and _low_die_can_miss(window, target):
            return "lucky", "a lower die can miss"
    if phase == "hit" and window.get("critical") and "target_id" in window:
        target = encounter.by_id(window["target_id"])
        if _low_die_can_miss(window, target) and int(window.get("mode") or 0) >= 0:
            if "protection" in ids:
                return "protection", "a lower die can miss"
            if "lucky" in ids:
                return "lucky", "a lower die can miss"
    for offer in offers:
        if offer["id"] != "cutting_words" and not str(offer["id"]).startswith("subtract:"):
            continue
        size = die_size(str(offer.get("die") or ""))
        if phase == "hit" and not window.get("critical"):
            target = encounter.by_id(window["target_id"])
            total = int(window["total"]) - int(window.get("penalty") or 0)
            needed = total - armor_class(target) + 1
            if 1 <= needed <= size:
                return offer["id"], f"the margin {needed} fits the die"
        if phase == "damage":
            target = encounter.by_id(window["target_id"])
            amount = sum(int(value) for _, value in window.get("rolled_parts") or [])
            if target.side == reactor.side and target.hp - amount <= 0 and amount - size < target.hp:
                return offer["id"], "the die can keep an ally above 0"
    if phase == "damage" and "deflect" in ids:
        return "deflect", "Deflect Attack reduces the damage."
    if phase == "damage" and "interception" in ids:
        target = encounter.by_id(window["target_id"])
        amount = sum(int(value) for _, value in window.get("rolled_parts") or [])
        offer = _offer(offers, "interception")
        size = int(offer.get("max") or 0) if offer else 0
        if target.side == reactor.side and target.hp - amount <= 0 and amount - size < target.hp:
            return "interception", "the reduction can keep an ally above 0"
    return None, ""


def _spend_spell(encounter, reactor, level: int) -> None:
    reactor.slots[level] = int(reactor.slots.get(level, 0)) - 1
    encounter.note_slot(reactor, level)
    reactor.reaction_available = False
    if encounter.current().id == reactor.id:
        reactor.traits["leveled_cast"] = True


def _reduce(parts: list[tuple[str, int]], amount: int) -> None:
    left = amount
    for index, (kind, value) in enumerate(parts):
        cut = min(int(value), left)
        parts[index] = (kind, int(value) - cut)
        left -= cut
        if left <= 0:
            return


def apply_choice(encounter, reactor, choice: str | None, window: dict, phase: str, reason: str) -> bool:
    """Spend the reaction. Return True when the triggering success is negated."""
    if not choice:
        if window.get("spends_reaction") is False:
            return False
        encounter.log.append(f"{reactor.name} does not use a reaction.")
        return False
    if choice == "shield":
        pool = _spend_known_or_pool(encounter, reactor, "shield", 1)
        reactor.traits["shield_ac"] = 5
        ac = armor_class(reactor)
        because = f" {reason}." if reason else ""
        via = f" using {pool}" if pool else ""
        encounter.log.append(f"{reactor.name} casts Shield{via}. AC is {ac} until their next turn.{because}")
        if phase == "missile":
            return True
        target = encounter.by_id(window["target_id"])
        return not still_hits(window, target)
    if str(choice).startswith("ac:"):
        # The bonus applies to this attack only. Shield's lasting bonus stays on shield_ac.
        record = _record_for(reactor, choice)
        bonus = int(record.bonus if record is not None else 0)
        reactor.reaction_available = False
        window["ac_bonus"] = int(window.get("ac_bonus") or 0) + bonus
        target = encounter.by_id(window["target_id"])
        ac = armor_class(target) + int(window["ac_bonus"])
        stands = still_hits(window, target)
        because = f" {reason}." if reason else ""
        outcome = "The attack still hits." if stands else "The attack misses."
        label = record.name if record is not None else "a reaction"
        encounter.log.append(f"{reactor.name} uses {label}. AC is {ac} against this attack. {outcome}{because}")
        return not stands
    if choice == "cutting_words" or str(choice).startswith("subtract:"):
        record = _subtract_record(reactor, choice)
        if record is None:
            encounter.log.append(f"{reactor.name} does not use a reaction.")
            return False
        pool = str(record.pool or _POOL)
        reactor.uses_left[pool] = int(reactor.uses_left.get(pool, 0)) - 1
        reactor.reaction_available = False
        rolled = roll_dice(str(record.die or "1d6"), encounter.rng)
        because = f" {reason}." if reason else ""
        if phase == "damage":
            _reduce(window["rolled_parts"], rolled.total)
            left = sum(int(value) for _, value in window["rolled_parts"])
            encounter.log.append(
                f"{reactor.name} uses {record.name} ({rolled.expression} -> {rolled.total}). Damage is now {left}.{because}"
            )
            return False
        window["penalty"] = int(window.get("penalty") or 0) + rolled.total
        total = int(window["total"]) - int(window["penalty"])
        encounter.log.append(
            f"{reactor.name} uses {record.name} ({rolled.expression} -> {rolled.total}). The attack total is {total}.{because}"
        )
        target = encounter.by_id(window["target_id"])
        return not still_hits(window, target)
    if choice == "counterspell":
        pool = _spend_known_or_pool(encounter, reactor, "counterspell", 3)
        via = f" using {pool}" if pool else ""
        caster = encounter.by_id(window["actor_id"])
        bonus = caster.save_bonuses.get("constitution", ability_mod(caster.abilities.get("constitution", 10)))
        roll, shown = d20(encounter.rng, 0)
        total = roll + bonus
        dc = int(reactor.spell_save_dc or 0)
        saved = total >= dc
        spell = caster.spell_named(window.get("spell_index"))
        name = spell.name if spell is not None else "The spell"
        because = f" {reason}." if reason else ""
        if saved:
            encounter.log.append(
                f"{reactor.name} casts Counterspell{via}. {caster.name} succeeds on the Constitution save "
                f"({shown[0]}+{bonus}={total} vs DC {dc}). {name} is cast.{because}"
            )
            return False
        window["countered"] = True
        encounter.log.append(
            f"{reactor.name} casts Counterspell{via}. {caster.name} fails the Constitution save "
            f"({shown[0]}+{bonus}={total} vs DC {dc}). {name} fizzles, and its spell slot is not spent.{because}"
        )
        return True
    if choice == "duelist":
        reactor.reaction_available = False
        reactor.traits["duelist_ac"] = _proficiency(reactor)
        target = encounter.by_id(window["target_id"])
        ac = opposed_ac(target, melee=True, ac_bonus=int(window.get("ac_bonus") or 0))
        because = f" {reason}." if reason else ""
        stands = still_hits(window, target)
        outcome = "The attack still hits." if stands else "The attack misses."
        encounter.log.append(
            f"{reactor.name} uses Defensive Duelist. AC is {ac} against melee attacks until their next turn. {outcome}{because}"
        )
        return not stands
    if choice == "lucky":
        _apply_luck(encounter, reactor, window, phase)
        if phase == "hit":
            target = encounter.by_id(window["target_id"])
            return not still_hits(window, target)
        return False
    if choice == "protection":
        reactor.reaction_available = False
        target = encounter.by_id(window["target_id"])
        target.traits["protected_by"] = reactor.id
        mode = int(window.get("mode") or 0)
        if mode < 0:
            pass
        elif mode > 0:
            _keep_shown(window, min)
        else:
            _keep_new_die(encounter, window, min)
        because = f" {reason}." if reason else ""
        stands = still_hits(window, target)
        outcome = "The attack still hits." if stands else "The attack misses."
        encounter.log.append(
            f"{reactor.name} uses Protection. Attacks against {target.name} have disadvantage while {reactor.name} is within 5 feet. {outcome}{because}"
        )
        return not stands
    if choice == "deflect":
        encounter.deflect_hit(reactor, window)
        return False
    if str(choice).startswith("redirect:"):
        creature_id = str(choice).split(":", 1)[1]
        try:
            target = encounter.by_id(creature_id)
        except KeyError:
            return False
        encounter.redirect_deflect(reactor, target, window)
        return False
    if choice == "interception":
        reactor.reaction_available = False
        die = str(reactor.traits.get("interception_die") or "1d10")
        rolled = roll_dice(die, encounter.rng)
        amount = rolled.total + _proficiency(reactor)
        _reduce(window["rolled_parts"], amount)
        left = sum(int(value) for _, value in window["rolled_parts"])
        because = f" {reason}." if reason else ""
        encounter.log.append(
            f"{reactor.name} uses Interception ({rolled.expression}+{_proficiency(reactor)} -> {amount}). Damage is now {left}.{because}"
        )
        return False
    if choice == "guarded_mind":
        reactor.uses_left["Guarded Mind"] = int(reactor.uses_left.get("Guarded Mind", 0)) - 1
        window["forced_success"] = True
        window["saved"] = True
        because = f" {reason}." if reason else ""
        encounter.log.append(f"{reactor.name} uses Guarded Mind and succeeds on the save.{because}")
        return False
    if choice == "shield_interpose":
        reactor.reaction_available = False
        window["negate_damage"] = True
        because = f" {reason}." if reason else ""
        encounter.log.append(f"{reactor.name} interposes a shield. The damage becomes 0.{because}")
        return False
    if choice in {"sentinel", "polearm", "opportunity"}:
        reactor.reaction_available = False
        weapon = ""
        halt = choice == "sentinel"
        for offer in window.get("offers") or []:
            if offer.get("id") == choice:
                weapon = str(offer.get("weapon") or "")
                break
        window["follow_attack"] = {
            "reactor_id": reactor.id,
            "target_id": window.get("target_id"),
            "weapon": weapon,
            "halt": halt,
        }
        because = f" {reason}." if reason else ""
        if choice == "opportunity":
            encounter.log.append(f"{reactor.name} makes an opportunity attack.{because}")
        else:
            label = "Sentinel" if choice == "sentinel" else "Polearm Master"
            encounter.log.append(f"{reactor.name} uses {label}.{because}")
        return False
    if str(choice).startswith("war:"):
        reactor.reaction_available = False
        parts = str(choice).split(":")
        spell_index = parts[1] if len(parts) > 1 else ""
        slot_text = parts[2] if len(parts) > 2 else "0"
        try:
            slot = int(slot_text)
        except ValueError:
            slot = 0
        window["follow_spell"] = {"reactor_id": reactor.id, "target_id": window.get("actor_id"), "spell_index": spell_index, "slot": slot}
        spell = reactor.spell_named(spell_index)
        name = spell.name if spell is not None else "a spell"
        because = f" {reason}." if reason else ""
        encounter.log.append(f"{reactor.name} casts {name} as a reaction.{because}")
        return False
    encounter.log.append(f"{reactor.name} does not use a reaction.")
    return False


def _keep_shown(window, pick) -> None:
    shown = tuple(window.get("shown") or (window.get("roll"),))
    roll = pick(shown) if shown else int(window["roll"])
    bonus = int(window.get("bonus") or 0)
    window["roll"] = roll
    window["shown"] = shown
    window["total"] = roll + bonus
    window["critical"] = roll >= int(window.get("crit_min") or 20) and roll != 1
    window["mode_override"] = 1 if pick is max else -1


def _keep_new_die(encounter, window, pick) -> None:
    current = int(window["roll"])
    extra = encounter.rng.randint(1, 20)
    roll = pick((current, extra))
    bonus = int(window.get("bonus") or 0)
    window["roll"] = roll
    window["shown"] = (current, extra)
    window["total"] = roll + bonus
    window["critical"] = roll >= int(window.get("crit_min") or 20) and roll != 1
    window["mode_override"] = 1 if pick is max else -1


def _apply_luck(encounter, reactor, window, phase) -> None:
    reactor.uses_left["Luck"] = int(reactor.uses_left.get("Luck", 0)) - 1
    mode = int(window.get("mode") or 0)
    if phase in {"luck", "save"} and mode < 0:
        _keep_shown(window, max)
    elif phase in {"luck", "save"}:
        _keep_new_die(encounter, window, max)
    else:
        _keep_new_die(encounter, window, min)
    if phase == "save":
        total = int(window["roll"]) + int(window.get("bonus") or 0)
        window["total"] = total
        window["saved"] = total >= int(window.get("dc") or 0) or bool(window.get("forced_success"))
    encounter.log.append(f"{reactor.name} spends a luck point and keeps {window['roll']}.")
