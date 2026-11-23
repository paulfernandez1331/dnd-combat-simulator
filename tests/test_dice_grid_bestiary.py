import json
import random
from pathlib import Path

from src.bestiary import (
    EXCLUDED_CAST,
    EXCLUDED_FIELDS,
    area_phrase,
    classify_monster,
    classify_spell,
    coverage,
    spell_damage_parts,
    spell_heal_expression,
)
from src.dice import average, parse_dice, roll_dice
from src.grid import chebyshev, cone, cube, line, reachable_squares, sphere
from src.paths import FIXTURES


def raw(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def test_parse_and_crit_roll():
    assert parse_dice("2d6+1") == (2, 6, 1)
    assert parse_dice("3d4 + 3") == (3, 4, 3)
    assert average("1d6+2") == 5.5
    rng = random.Random(0)
    rng.randint = lambda a, b: 6  # type: ignore[method-assign]
    normal = roll_dice("1d6+2", rng)
    crit = roll_dice("1d6+2", rng, critical=True)
    assert normal.rolls == (6,)
    assert normal.total == 8
    assert crit.rolls == (6, 6)
    assert crit.total == 14


def test_modifier_token():
    assert spell_heal_expression(classify_spell(raw("cure-wounds")), 1, 3) == "1d8+3"
    assert spell_heal_expression(classify_spell(raw("cure-wounds")), 1, -1) == "1d8-1"


def test_spell_classes_and_scaling():
    fireball = classify_spell(raw("fireball"))
    assert fireball.kind == "save"
    assert fireball.dc_success == "half"
    assert fireball.area_type == "sphere"
    assert fireball.area_size == 20
    assert spell_damage_parts(fireball, 3, 5) == [("fire", "8d6")]
    assert spell_damage_parts(fireball, 4, 5) == [("fire", "9d6")]

    bolt = classify_spell(raw("fire-bolt"))
    assert bolt.kind == "attack"
    assert spell_damage_parts(bolt, 0, 5) == [("fire", "2d10")]
    assert spell_damage_parts(bolt, 0, 1) == [("fire", "1d10")]

    missile = classify_spell(raw("magic-missile"))
    assert missile.kind == "auto"
    assert missile.activation == "action"
    assert spell_damage_parts(missile, 1, 5) == [("force", "3d4+3")]
    assert spell_damage_parts(missile, 2, 5) == [("force", "4d4+4")]

    word = classify_spell(raw("healing-word"))
    assert word.kind == "heal"
    assert word.activation == "bonus"
    assert spell_heal_expression(word, 1, 3) == "2d4+3"
    assert spell_heal_expression(word, 2, 3) == "4d4+3"

    assert classify_spell(raw("cure-wounds")).kind == "heal"
    assert classify_spell(raw("burning-hands")).area_type == "cone"
    assert classify_spell(raw("lightning-bolt")).area_type == "line"
    assert area_phrase("A 100-foot-long, 5-foot-wide Line blasts out.") == ("line", 100)

    alarm = classify_spell(raw("alarm"))
    assert alarm.kind == "excluded"
    assert alarm.exclude_reason == EXCLUDED_CAST

    guardians = classify_spell(raw("spirit-guardians"))
    assert guardians.kind == "excluded"
    assert guardians.exclude_reason == EXCLUDED_FIELDS


def test_goblin_and_dragon_actions():
    goblin = classify_monster(raw("goblin-warrior"))
    assert goblin.size == "Small"
    assert goblin.ac == 15
    assert goblin.hp == 10
    assert goblin.speed == 30
    assert goblin.challenge_rating == 0.25
    weapons = [attack for attack in goblin.attacks if attack.resolution == "attack"]
    assert [attack.name for attack in weapons] == ["Scimitar", "Shortbow"]
    scimitar = weapons[0]
    assert scimitar.attack_bonus == 4
    assert scimitar.reach_feet == 5
    assert scimitar.damage == [{"type": "slashing", "expr": "1d6+2"}]
    assert weapons[1].range_feet == 80
    escape = next(attack for attack in goblin.attacks if attack.name == "Nimble Escape")
    assert escape.resolution == "disengage"
    assert escape.activation == "bonus"

    dragon = classify_monster(raw("adult-red-dragon"))
    assert dragon.size == "Huge"
    assert dragon.immunities == ["fire"]
    assert dragon.multiattack == [("Rend", 3)]
    assert "Scorching Ray" in dragon.excluded_actions
    breath = next(attack for attack in dragon.attacks if attack.name == "Fire Breath")
    assert breath.resolution == "save"
    assert breath.dc_value == 21
    assert breath.dc_success == "half"
    assert breath.disagrees is False
    assert breath.recharge_min == 5
    assert breath.area_type == "cone"
    assert breath.area_size == 60
    assert breath.damage == [{"type": "fire", "expr": "17d6"}]
    assert "Fire Breath" not in dragon.disagreements
    rend = next(attack for attack in dragon.attacks if attack.name == "Rend")
    assert rend.reach_feet == 10
    assert len(rend.damage) == 2


def test_coverage_counts_exclusions():
    spells = [
        classify_spell(raw(name))
        for name in (
            "fireball",
            "fire-bolt",
            "magic-missile",
            "cure-wounds",
            "alarm",
            "spirit-guardians",
        )
    ]
    monsters = [classify_monster(raw("goblin-warrior")), classify_monster(raw("adult-red-dragon"))]
    report = coverage(spells, monsters)
    assert report["spell_total"] == 6
    assert report["spells_by_kind"]["save"] == 1
    assert report["spells_by_kind"]["excluded"] == 2
    assert report["monsters_with_resolvable_action"] == 2
    assert report["save_text_disagreements"] == 0


def test_fireball_sphere_and_burning_hands_cone():
    center = (5, 5)
    blast = sphere(center, 20)
    assert (5, 5) in blast
    assert (9, 5) in blast
    assert (9, 9) in blast
    assert (10, 5) not in blast
    assert chebyshev((5, 5), (9, 9)) == 4

    cells = cone((0, 0), 15, (1, 0))
    assert (1, 0) in cells and (2, 0) in cells and (3, 0) in cells
    assert (3, 3) in cells
    assert (0, 0) not in cells
    assert (0, 1) not in cells
    assert (-1, 0) not in cells
    assert (3, 4) not in cells
    expected = {
        (dx, dy)
        for dx in range(1, 4)
        for dy in range(-3, 4)
        if max(abs(dx), abs(dy)) <= 3 and abs(dy) <= dx
    }
    assert cells == expected


def test_line_cube_and_movement():
    bolt = line((0, 0), 100, (1, 0))
    assert bolt == {(step, 0) for step in range(1, 21)}
    assert line((0, 0), 15, (1, 1)) == {(1, 1), (2, 2), (3, 3)}

    box = cube((0, 0), 15, (1, 0))
    assert box == {(x, y) for x in range(3) for y in (-1, 0, 1)}

    opened = reachable_squares((0, 0), 30, width=12, height=12, blocked=set(), passable=set())
    assert (6, 0) in opened
    assert (6, 6) in opened
    assert (7, 0) not in opened
    assert (0, 0) not in opened

    blocked = {(1, 0)}
    ally = {(1, 1)}
    around = reachable_squares((0, 0), 10, width=12, height=12, blocked=blocked, passable=ally)
    assert (1, 0) not in around
    assert (1, 1) not in around
    assert (2, 1) in around
    assert (2, 0) in around
