"""Actions read from an imported character, plus turn bookkeeping for weapon mastery.

A class action exists only when the public sheet gives this fight a use, a die,
or a feature name the resolver already runs. Weapon mastery is a property of
the weapon, so Nick and Cleave stay available for any creature that has them.
"""

from __future__ import annotations

from src.bestiary import ability_mod
from src.conditions import base_speed, is_incapacitated
from src.dice import roll_dice
from src.feats import pole_weapon
from src.grid import chebyshev, footprint, in_bounds
from src.rules import Effect, apply_heal, combine, d20


def begin_turn_state(encounter, actor) -> None:
    for other in encounter.combatants:
        if other.traits.get("stunned_by") == actor.id:
            other.conditions.pop("stunned", None)
            other.traits.pop("stunned_by", None)
        if other.traits.get("slow_from") == actor.id:
            other.traits.pop("slow", None)
            other.traits.pop("slow_from", None)
        if other.traits.get("protected_by") == actor.id:
            other.traits.pop("protected_by", None)
        if other.traits.get("crusher_from") == actor.id:
            other.traits.pop("crusher_from", None)
        if other.traits.get("slasher_disadvantage") == actor.id:
            other.traits.pop("slasher_disadvantage", None)
        if other.traits.get("frightened_by") == actor.id:
            other.conditions.pop("frightened", None)
            other.traits.pop("frightened_by", None)
    actor.attacks_made = 0
    actor.traits.pop("shield_ac", None)
    actor.reaction_available = not is_incapacitated(actor)
    for key in (
        "reckless", "light_used", "nick_used", "cleave_used", "fury_used",
        "hew_ready", "hew_strike", "leveled_cast", "duelist_ac", "savage_used",
        "charger_used", "piercer_used", "slasher_used", "crusher_used",
        "tavern_push_used", "shield_bash_used", "pole_ready", "moved_feet",
        "on_attack_action", "strip_mod", "reaction_strike", "disengaged", "move_path",
        "surge_blocks_magic", "banked_action", "surged", "shift_feet",
        "flurry_left", "monk_attacked", "flurry_swing", "stunning_used",
        "swing_budget", "swing_tag",
    ):
        actor.traits.pop(key, None)
    if int(getattr(encounter, "round", 1) or 1) > 1:
        actor.traits["metabolism_closed"] = True


def break_hide(encounter, actor) -> None:
    for other in encounter.combatants:
        if other.hp <= 0 or other.id == actor.id:
            continue
        if other.side == actor.side or actor.gap_to(other) > 1:
            continue
        actor.traits.pop("hidden", None)
        other.traits.pop("hidden", None)


def _away_step(actor, target) -> tuple[int, int]:
    best = None
    for origin in sorted(actor.squares()):
        for square in sorted(target.squares()):
            dist = chebyshev(origin, square)
            if dist == 0:
                continue
            dx = square[0] - origin[0]
            dy = square[1] - origin[1]
            step = (
                0 if dx == 0 else (1 if dx > 0 else -1),
                0 if dy == 0 else (1 if dy > 0 else -1),
            )
            candidate = (dist, abs(dx) + abs(dy), step)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        return (0, 0)
    return best[2]


def push_from(encounter, actor, target, squares: int) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    step_x, step_y = _away_step(actor, target)
    current = tuple(target.position)
    visited: list[tuple[int, int]] = []
    if step_x == 0 and step_y == 0:
        return current, visited
    occupied: set[tuple[int, int]] = set()
    for creature in encounter.combatants:
        if creature.hp > 0 and creature.id != target.id:
            occupied.update(creature.squares())
    for _ in range(squares):
        nxt = (current[0] + step_x, current[1] + step_y)
        body = footprint(nxt, target.span)
        if any(not in_bounds(cell, encounter.width, encounter.height) or cell in occupied for cell in body):
            break
        current = nxt
        visited.append(current)
    target.position = current
    return current, visited


def add_character_actions(actions, encounter, actor, automated: bool, append) -> None:
    if actor.traits.get("reckless_feature") and not actor.traits.get("reckless"):
        append(actions, kind="feature", label="Reckless Attack", attack_name="Reckless Attack", budget="action")
    if _left(actor, "Action Surge") > 0 and not actor.traits.get("surged"):
        append(actions, kind="feature", label="Action Surge", attack_name="Action Surge", budget="action")
    _metabolism(actions, encounter, actor, append)
    if not actor.bonus_available:
        return
    if _left(actor, "Rage") > 0 and not actor.traits.get("rage"):
        append(actions, kind="feature", label="Rage", attack_name="Rage", budget="bonus")
    for heal in actor.traits.get("bonus_heals") or []:
        name = str(heal.get("name") or "")
        if name and _left(actor, name) > 0:
            append(actions, kind="feature", label=name, attack_name=name, budget="bonus")
    _hew(actions, encounter, actor, append)
    _dual_wielder(actions, encounter, actor, append)
    _pole_butt(actions, encounter, actor, append)
    _telekinetic(actions, encounter, actor, append)
    _spiritual_weapon(actions, encounter, actor, automated, append)
    _zone_features(actions, encounter, actor, automated, append)
    _monk_bonus(actions, encounter, actor, append)


def apply_feature(encounter, actor, action) -> Effect:
    name = action.attack_name or ""
    if name == "Rage":
        return _rage(actor)
    if name == "Reckless Attack":
        actor.traits["reckless"] = True
        return Effect(lines=[f"{actor.name} attacks recklessly until their next turn."])
    if name == "Action Surge":
        return _surge(encounter, actor)
    if name == "Flurry of Blows":
        return _flurry(actor)
    if name == "Focused Patient Defense":
        return _focused_patient(actor)
    if name == "Focused Step of the Wind":
        return _focused_step(actor)
    metabolism = str((monk_profile(actor).get("metabolism") or {}).get("name") or "")
    if metabolism and name == metabolism:
        return _metabolism_heal(encounter, actor)
    for heal in actor.traits.get("bonus_heals") or []:
        if name and name == heal.get("name"):
            return _heal(encounter, actor, name, str(heal.get("die") or "1d8"))
    if name == "Telekinetic Shove":
        return _shove(encounter, actor, action)
    if name == "Spiritual Weapon":
        return encounter.move_spiritual_weapon(actor, action)
    if action.tag in {"zone", "bolt"}:
        return encounter.move_zone(actor, action)
    if action.tag == "escape":
        return encounter.break_free(actor, action)
    return Effect(lines=[f"{actor.name} has no resolution for {name}."])


def _hew(actions, encounter, actor, append) -> None:
    name = str(actor.traits.get("hew_ready") or "")
    option = actor.attack_named(name) if name else None
    if option is None or not option.reach_feet:
        return
    for foe in encounter.combatants:
        if foe.hp <= 0 or foe.side == actor.side:
            continue
        if actor.gap_to(foe) * 5 > option.reach_feet:
            continue
        append(
            actions,
            kind="attack",
            label=f"Hew: {option.name} -> {foe.name}",
            attack_name=option.name,
            target_id=foe.id,
            budget="bonus",
            tag="hew",
        )


def _rage(actor) -> Effect:
    _spend(actor, "Rage")
    actor.bonus_available = False
    actor.traits["rage"] = True
    added = []
    for kind in ("bludgeoning", "piercing", "slashing"):
        if kind not in actor.resistances:
            actor.resistances.append(kind)
            added.append(kind)
    actor.traits["rage_added"] = added
    bonus = int(actor.traits.get("rage_bonus") or 2)
    return Effect(lines=[
        f"{actor.name} rages. Strength attacks deal +{bonus}, "
        "and bludgeoning, piercing, and slashing damage is halved."
    ])


def _surge(encounter, actor) -> Effect:
    if actor.traits.get("surged") or _left(actor, "Action Surge") <= 0:
        return Effect(lines=["Action Surge has no uses left."])
    _spend(actor, "Action Surge")
    actor.traits["surged"] = True
    actor.traits["banked_action"] = True
    if not actor.action_available:
        encounter._open_banked(actor)
    return Effect(lines=[
        f"{actor.name} uses Action Surge and can take one additional action, except the Magic action."
    ])


def _heal(encounter, actor, name: str, expression: str) -> Effect:
    if _left(actor, name) <= 0:
        return Effect(lines=[f"{name} has no uses left."])
    _spend(actor, name)
    actor.bonus_available = False
    rolled = roll_dice(expression, encounter.rng)
    effect = apply_heal(actor, rolled.total)
    effect.lines.insert(0, f"{actor.name} uses {name} ({rolled.expression} -> {rolled.total}).")
    if actor.traits.get("tactical_shift"):
        from src.actions import movement_budget

        gained = base_speed(actor) // 2
        if gained:
            actor.movement_left = movement_budget(actor) + gained
            actor.traits["shift_feet"] = int(actor.traits.get("shift_feet") or 0) + gained
            effect.lines.append(f"{actor.name} can move {gained} feet without provoking opportunity attacks.")
    return effect


def _dual_wielder(actions, encounter, actor, append) -> None:
    if not actor.traits.get("dual_wielder") or not actor.bonus_available or actor.traits.get("nick_used"):
        return
    used = str(actor.traits.get("light_used") or "")
    if not used:
        return
    from src.actions import using_melee, weapon_reaches

    for option in actor.attacks:
        if option.name == used or option.two_handed or not option.reach_feet or option.resolution != "attack":
            continue
        for foe in encounter.combatants:
            if foe.hp <= 0 or foe.side == actor.side:
                continue
            if not weapon_reaches(actor, foe, option) or not using_melee(actor, foe, option):
                continue
            append(
                actions,
                kind="attack",
                label=f"Dual Wielder: {option.name} -> {foe.name}",
                attack_name=option.name,
                target_id=foe.id,
                budget="bonus",
                tag="dual",
            )


def _pole_butt(actions, encounter, actor, append) -> None:
    if not actor.bonus_available:
        return
    name = str(actor.traits.get("pole_ready") or "")
    option = actor.attack_named(name) if name else None
    if option is None or not pole_weapon(option):
        return
    reach = int(option.reach_feet or 0)
    for foe in encounter.combatants:
        if foe.hp <= 0 or foe.side == actor.side:
            continue
        if actor.gap_to(foe) * 5 > reach:
            continue
        append(
            actions,
            kind="attack",
            label=f"Pole strike -> {foe.name}",
            attack_name=option.name,
            target_id=foe.id,
            budget="bonus",
            tag="pole",
        )


def _spiritual_weapon(actions, encounter, actor, automated: bool, append) -> None:
    if not actor.bonus_available or actor.concentrating_on != "spiritual-weapon":
        return
    placed = actor.traits.get("spiritual_weapon")
    if not isinstance(placed, dict):
        return
    origin = (int(placed.get("x", 0)), int(placed.get("y", 0)))
    from src.actions import _occupied, adjacent_foe

    taken = _occupied(encounter)
    for x in range(encounter.width):
        for y in range(encounter.height):
            square = (x, y)
            if chebyshev(origin, square) > 4 or square in taken:
                continue
            if automated and adjacent_foe(encounter, actor, square) is None:
                continue
            append(
                actions,
                kind="feature",
                label=f"Spiritual Weapon at ({square[0]}, {square[1]})",
                attack_name="Spiritual Weapon",
                aim=square,
                budget="bonus",
            )


def _zone_features(actions, encounter, actor, automated, append) -> None:
    """Repeat a placed token or bolt, or spend the action to break free."""
    from src.actions import _foes, _token_target
    from src.grid import sphere
    from src.zones import cells_of, footprint_free, read_zone

    if actor.action_available and actor.traits.get("restrained_by"):
        try:
            caster = encounter.by_id(actor.traits.get("restrained_by"))
        except KeyError:
            caster = None
        found = read_zone(caster) if caster is not None else None
        if found is not None and found[0].escape:
            append(actions, kind="feature", label="Break free", attack_name="Break free", budget="action", tag="escape")
    found = read_zone(actor)
    if found is None:
        return
    spec, placed = found
    origin = (int(placed.get("x", 0)), int(placed.get("y", 0)))
    if spec.move == "bolt" and actor.action_available:
        reach = max(1, spec.size_feet // 5)
        for square in sorted(sphere(origin, spec.size_feet)):
            if not (0 <= square[0] < encounter.width and 0 <= square[1] < encounter.height):
                continue
            if chebyshev(origin, square) > reach:
                continue
            if automated and not any(foe.in_cells(sphere(square, spec.bolt)) for foe in _foes(encounter, actor)):
                continue
            append(
                actions,
                kind="feature",
                label=f"Call Lightning at ({square[0]}, {square[1]})",
                attack_name="Call Lightning",
                aim=square,
                budget="action",
                tag="bolt",
            )
        return
    if spec.move not in {"bonus", "action"}:
        return
    budget = spec.move
    if budget == "bonus" and not actor.bonus_available:
        return
    if budget == "action" and not actor.action_available:
        return
    spell = actor.spell_named(spec.index)
    name = spell.name if spell is not None else spec.index
    limit = max(0, spec.move_feet // 5)
    for x in range(encounter.width):
        for y in range(encounter.height):
            square = (x, y)
            if chebyshev(origin, square) > limit:
                continue
            if spec.index != "flaming-sphere" and spec.shape == "token" and not footprint_free(encounter, square, spec.span):
                continue
            if automated and spec.attack and _token_target(encounter, actor, square, spec) is None:
                continue
            if automated and spec.index == "flaming-sphere" and _token_target(encounter, actor, square, spec) is None:
                continue
            if automated and spec.move == "action":
                probe = {"x": square[0], "y": square[1], "form": spec.shape, "dx": 0, "dy": 0}
                if not any(foe.in_cells(cells_of(spec, probe)) for foe in _foes(encounter, actor)):
                    continue
            append(
                actions,
                kind="feature",
                label=f"{name} at ({square[0]}, {square[1]})",
                attack_name=name,
                aim=square,
                budget=budget,
                tag="zone",
            )


def _telekinetic(actions, encounter, actor, append) -> None:
    if not actor.bonus_available or not actor.traits.get("telekinetic"):
        return
    for creature in encounter.combatants:
        if creature.hp <= 0 or creature.id == actor.id:
            continue
        gap = actor.gap_to(creature)
        if gap <= 0 or gap * 5 > 30:
            continue
        append(
            actions,
            kind="feature",
            label=f"Telekinetic Shove -> {creature.name}",
            attack_name="Telekinetic Shove",
            target_id=creature.id,
            budget="bonus",
            tag="tele",
        )


def _shove(encounter, actor, action) -> Effect:
    actor.bonus_available = False
    target = None
    for creature in encounter.combatants:
        if creature.id == action.target_id:
            target = creature
            break
    if target is None or target.hp <= 0:
        return Effect(lines=[f"{actor.name} has no one to shove."])
    dc = int(actor.traits.get("telekinetic") or 10)
    bonus = target.save_bonuses.get("strength", ability_mod(target.abilities.get("strength", 10)))
    roll, shown = d20(encounter.rng, 0)
    total = roll + bonus
    if total >= dc:
        return Effect(lines=[f"{target.name} resists Telekinetic Shove ({shown[0]}+{bonus}={total} vs DC {dc})."])
    origin = tuple(target.position)
    dest, visited = push_from(encounter, actor, target, 1)
    moved = Effect(lines=[f"{target.name} is shoved to {dest} ({shown[0]}+{bonus}={total} vs DC {dc})."])
    return combine([moved, encounter._sweep_emanation(target, origin, visited)])


def _left(actor, name: str) -> int:
    return int(actor.uses_left.get(name, 0))


def _spend(actor, name: str, count: int = 1) -> None:
    actor.uses_left[name] = _left(actor, name) - count


def monk_profile(actor) -> dict:
    found = actor.traits.get("monk")
    return found if isinstance(found, dict) else {}


def focus_name(actor) -> str:
    return str(monk_profile(actor).get("focus") or "")


def spend_focus(actor, count: int = 1) -> bool:
    name = focus_name(actor)
    if not name or _left(actor, name) < count:
        return False
    _spend(actor, name, count)
    return True


def refill_focus(actor) -> None:
    name = focus_name(actor)
    if not name:
        return
    actor.uses_left[name] = int(actor.uses_max.get(name, _left(actor, name)))


def _metabolism(actions, encounter, actor, append) -> None:
    profile = monk_profile(actor)
    info = profile.get("metabolism") or {}
    name = str(info.get("name") or "")
    if not name or _left(actor, name) <= 0 or actor.traits.get("metabolism_closed"):
        return
    if int(getattr(encounter, "round", 1) or 1) > 1 or actor.hp >= actor.max_hp:
        return
    append(actions, kind="feature", label=name, attack_name=name, budget="free")


def _foe_in_unarmed_reach(encounter, actor) -> bool:
    option = actor.attack_named("Unarmed Strike")
    reach = int(option.reach_feet or 5) if option is not None else 5
    for foe in encounter.combatants:
        if foe.hp > 0 and foe.side != actor.side and 0 < actor.gap_to(foe) * 5 <= reach:
            return True
    return False


def _monk_bonus(actions, encounter, actor, append) -> None:
    profile = monk_profile(actor)
    if not profile:
        return
    focus = _left(actor, focus_name(actor)) if focus_name(actor) else 0
    if profile.get("flurry") and focus > 0 and _foe_in_unarmed_reach(encounter, actor):
        append(actions, kind="feature", label="Flurry of Blows", attack_name="Flurry of Blows", budget="bonus")
    if profile.get("patient"):
        append(actions, kind="disengage", label="Patient Defense", attack_name="Patient Defense", budget="bonus")
    if profile.get("patient_focused") and focus > 0:
        append(
            actions,
            kind="feature",
            label="Focused Patient Defense",
            attack_name="Focused Patient Defense",
            budget="bonus",
        )
    if profile.get("step"):
        append(actions, kind="dash", label="Step of the Wind", attack_name="Step of the Wind", budget="bonus")
    if profile.get("step_focused") and focus > 0:
        append(
            actions,
            kind="feature",
            label="Focused Step of the Wind",
            attack_name="Focused Step of the Wind",
            budget="bonus",
        )


def _flurry(actor) -> Effect:
    if not spend_focus(actor):
        return Effect(lines=["Flurry of Blows has no Focus left."])
    actor.bonus_available = False
    actor.traits["flurry_left"] = 2
    return Effect(lines=[f"{actor.name} spends 1 Focus on Flurry of Blows."])


def _focused_patient(actor) -> Effect:
    if not spend_focus(actor):
        return Effect(lines=["Focused Patient Defense has no Focus left."])
    actor.bonus_available = False
    actor.traits["disengaged"] = True
    actor.dodging = True
    return Effect(lines=[f"{actor.name} spends 1 Focus. Patient Defense is Disengage and Dodge."])


def _focused_step(actor) -> Effect:
    if not spend_focus(actor):
        return Effect(lines=["Focused Step of the Wind has no Focus left."])
    actor.bonus_available = False
    actor.traits["disengaged"] = True
    from src.actions import movement_budget, turn_speed

    gained = turn_speed(actor)
    if actor.traits.get("charger"):
        gained += 10
    actor.movement_left = movement_budget(actor) + gained
    return Effect(lines=[
        f"{actor.name} spends 1 Focus. Step of the Wind is Disengage and Dash.",
        f"{actor.name}'s jump distance is doubled this turn.",
        f"{actor.movement_left} ft of movement left.",
    ])


def _metabolism_heal(encounter, actor) -> Effect:
    info = monk_profile(actor).get("metabolism") or {}
    name = str(info.get("name") or "Uncanny Metabolism")
    if _left(actor, name) <= 0:
        return Effect(lines=[f"{name} has no uses left."])
    _spend(actor, name)
    actor.traits["metabolism_closed"] = True
    refill_focus(actor)
    expression = str(info.get("die") or "1d8")
    rolled = roll_dice(expression, encounter.rng)
    effect = apply_heal(actor, rolled.total)
    effect.lines.insert(0, f"{actor.name} uses {name} ({rolled.expression} -> {rolled.total}) and regains Focus.")
    return effect
