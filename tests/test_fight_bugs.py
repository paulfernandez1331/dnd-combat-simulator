"""Attack reasons, Boulder Toss, and a cone that stays a facing. No character imports."""

import json
import re

from src.combatant import from_monster
from src.encounter import Encounter
from src.paths import CACHE
from src.rules import resolve_attack
from tests.test_combat import Scripted, _prepare, creature


def _text(effect) -> str:
    return "\n".join(effect.lines or [])


def _monster(name: str):
    payload = json.loads((CACHE / "monsters" / f"{name}.json").read_text(encoding="utf-8"))
    from src.bestiary import classify_monster

    return classify_monster(payload)


def test_a_prone_target_beyond_five_feet_names_the_disadvantage():
    attacker = creature("caster", "Druid", "party", 14, 30, (0, 0))
    target = creature("foe", "Foe", "enemy", 18, 30, (3, 0))
    target.conditions["prone"] = 99
    effect = resolve_attack(
        attacker,
        target,
        bonus=7,
        damage=[("radiant", "4d6")],
        rng=Scripted([14, 8]),
        melee=False,
        spell=True,
        label="Guiding Bolt",
    )
    assert "disadvantage 14 and 8, keeps 8 (the target is prone and more than 5 feet away)" in _text(effect)


def test_a_poisoned_attacker_names_that_one_source():
    attacker = creature("caster", "Druid", "party", 14, 30, (0, 0))
    attacker.conditions["poisoned"] = 99
    target = creature("foe", "Foe", "enemy", 18, 30, (2, 0))
    effect = resolve_attack(
        attacker,
        target,
        bonus=7,
        damage=[("radiant", "4d6")],
        rng=Scripted([14, 8]),
        melee=False,
        spell=True,
        label="Guiding Bolt",
    )
    text = _text(effect)
    assert "(the attacker is poisoned)" in text
    assert re.search(r"\badvantage\b", text) is None


def test_several_disadvantage_sources_are_joined():
    attacker = creature("caster", "Druid", "party", 14, 30, (0, 0))
    attacker.conditions["poisoned"] = 99
    target = creature("foe", "Foe", "enemy", 18, 30, (3, 0))
    target.conditions["prone"] = 99
    effect = resolve_attack(
        attacker,
        target,
        bonus=7,
        damage=[("radiant", "4d6")],
        rng=Scripted([14, 8]),
        melee=False,
        spell=True,
        label="Guiding Bolt",
    )
    assert "(the attacker is poisoned; the target is prone and more than 5 feet away)" in _text(effect)


def test_no_source_is_one_die_and_opposed_conditions_cancel():
    attacker = creature("caster", "Druid", "party", 14, 30, (0, 0))
    target = creature("foe", "Foe", "enemy", 30, 30, (2, 0))
    plain = resolve_attack(
        attacker,
        target,
        bonus=4,
        damage=[("radiant", "1d6")],
        rng=Scripted([11]),
        melee=False,
        spell=True,
        label="Guiding Bolt",
    )
    plain_text = _text(plain)
    assert re.search(r"\badvantage\b", plain_text) is None
    assert re.search(r"\bdisadvantage\b", plain_text) is None
    assert "(" not in plain_text

    attacker.conditions["prone"] = 99
    target.position = (1, 0)
    target.conditions["prone"] = 99
    cancelled = resolve_attack(
        attacker,
        target,
        bonus=4,
        damage=[("radiant", "1d6")],
        rng=Scripted([11]),
        melee=True,
        label="Rapier",
    )
    cancelled_text = _text(cancelled)
    assert re.search(r"\badvantage\b", cancelled_text) is None
    assert re.search(r"\bdisadvantage\b", cancelled_text) is None


def _boulder_rolls(save: int, face: int = 6) -> list[int]:
    return [save, *([face] * 7)]


def test_boulder_toss_is_an_aimed_save_that_knocks_a_medium_creature_prone():
    record = _monster("giant-ape")
    toss = next(attack for attack in record.attacks if attack.name == "Boulder Toss")
    assert toss.resolution == "save"
    assert toss.range_feet == 90
    assert toss.area_type == "sphere"
    assert toss.area_size == 5
    assert toss.dc_value == 17
    assert toss.dc_ability == "dex"
    assert toss.dc_success == "half"
    assert toss.recharge_min == 6
    assert toss.rider.get("aimed") is True
    assert toss.rider.get("prone_size") == "large"
    assert "Leap" in record.excluded_actions

    ape = from_monster(record, creature_id="ape", name="Giant Ape", side="enemy", position=(0, 0))
    foe = creature("foe", "Foe", "party", 14, 80, (8, 2))
    encounter = Encounter([ape, foe], seed=1)
    _prepare(encounter, ["ape", "foe"], _boulder_rolls(1))
    aims = [action for action in encounter.legal_actions(ape) if action.attack_name == "Boulder Toss"]
    assert aims
    assert all(action.aim is not None and action.facing is None for action in aims)
    assert any(action.attack_name == "Leap" for action in encounter.legal_actions(ape)) is False
    chosen = next(action for action in aims if action.aim == (8, 2))
    encounter.apply(chosen, "test")
    assert "prone" in foe.conditions
    assert foe.hp == 80 - 42
    assert "Boulder Toss" in ape.recharging
    assert all(action.attack_name != "Boulder Toss" for action in encounter.legal_actions(ape))


def test_boulder_toss_misses_past_ninety_feet_and_a_success_is_half_damage():
    record = _monster("giant-ape")
    ape = from_monster(record, creature_id="ape", name="Giant Ape", side="enemy", position=(0, 0))
    far = creature("far", "Far", "party", 14, 80, (22, 0))
    beyond = Encounter([ape, far], seed=1, width=30)
    _prepare(beyond, ["ape", "far"], [])
    assert all(action.attack_name != "Boulder Toss" for action in beyond.legal_actions(ape))

    saved = from_monster(record, creature_id="ape", name="Giant Ape", side="enemy", position=(0, 0))
    foe = creature("foe", "Foe", "party", 14, 80, (8, 2))
    hit = Encounter([saved, foe], seed=1)
    _prepare(hit, ["ape", "foe"], _boulder_rolls(17))
    chosen = next(
        action for action in hit.legal_actions(saved)
        if action.attack_name == "Boulder Toss" and action.aim == (8, 2)
    )
    hit.apply(chosen, "test")
    assert "prone" not in foe.conditions
    assert foe.hp == 80 - 21


def test_a_huge_creature_stays_standing_and_fire_breath_stays_a_facing():
    ape = from_monster(_monster("giant-ape"), creature_id="ape", name="Giant Ape", side="enemy", position=(0, 0))
    huge = creature("huge", "Huge", "party", 14, 120, (6, 0), size="Huge")
    encounter = Encounter([ape, huge], seed=1)
    _prepare(encounter, ["ape", "huge"], _boulder_rolls(1))
    chosen = next(
        action for action in encounter.legal_actions(ape)
        if action.attack_name == "Boulder Toss" and action.aim == (6, 0)
    )
    encounter.apply(chosen, "test")
    assert "prone" not in huge.conditions
    assert huge.hp == 120 - 42

    dragon = from_monster(
        _monster("adult-red-dragon"),
        creature_id="dragon",
        name="Adult Red Dragon",
        side="enemy",
        position=(0, 0),
    )
    foe = creature("foe", "Foe", "party", 16, 40, (4, 2))
    breath = Encounter([dragon, foe], seed=1)
    _prepare(breath, ["dragon", "foe"], [])
    cones = [action for action in breath.legal_actions(dragon) if action.attack_name == "Fire Breath"]
    assert cones
    assert all(action.facing is not None and action.aim is None for action in cones)
