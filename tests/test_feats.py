"""Player's Handbook feats. Synthetic creatures only. No D&D Beyond requests."""

import random
import re

from src.actions import _creature_in_range
from src.bestiary import AttackOption
from src.ddb import public_character
from src.encounter import Encounter
from src.rules import apply_damage
from src.sheets import sheet_for
from tests.test_combat import Scripted, _prepare, creature, load_spell, sword


def _fight(hero, foe, rolls, order=None):
    encounter = Encounter([hero, foe], seed=1)
    _prepare(encounter, order or [hero.id, foe.id], rolls)
    encounter.reaction_policy = "heuristic"
    return encounter


def _text(encounter) -> str:
    return " ".join(encounter.log)


def test_alert_adds_its_bonus_to_initiative():
    hero = creature("hero", "Hero", "party", 15, 20, (0, 0), abilities={"dexterity": 14})
    hero.traits["alert"] = True
    hero.traits["initiative_bonus"] = 3
    foe = creature("foe", "Foe", "enemy", 12, 20, (3, 0), abilities={"dexterity": 10})
    rng = random.Random(1)
    hero_total = rng.randint(1, 20) + 5
    encounter = Encounter([hero, foe], seed=1)
    encounter.start()
    assert encounter.log[0].startswith("Dice seed")
    assert encounter.log[1].startswith("Initiative:")
    assert re.search(rf"Hero {hero_total}(,|$)", encounter.log[1])
    assert not any("swaps initiative" in line for line in encounter.log)


def test_alert_swaps_with_the_higher_ally():
    hero = creature("hero", "Hero", "party", 15, 20, (0, 0), abilities={"dexterity": 10})
    hero.traits["alert"] = True
    hero.traits["initiative_bonus"] = 2
    ally = creature("ally", "Ally", "party", 15, 20, (1, 0), abilities={"dexterity": 10})
    foe = creature("foe", "Foe", "enemy", 12, 20, (5, 0), abilities={"dexterity": 10})
    seed = None
    for candidate in range(1, 400):
        rng = random.Random(candidate)
        rolls = [rng.randint(1, 20) for _ in range(3)]
        if rolls[0] + 2 < rolls[1]:
            seed = candidate
            break
    assert seed is not None
    encounter = Encounter([hero, ally, foe], seed=seed)
    encounter.start()
    assert encounter.log[0].startswith("Dice seed")
    assert encounter.log[1].startswith("Initiative:")
    assert "Hero swaps initiative with Ally." in encounter.log
    assert encounter.order.index("hero") < encounter.order.index("ally")


def test_archery_adds_two_to_a_ranged_attack_and_not_a_thrown_one():
    bow = AttackOption(
        name="Bow", resolution="attack", attack_bonus=3, range_feet=80,
        damage=[{"type": "piercing", "expr": "1d8"}],
    )
    hero = creature("hero", "Hero", "party", 14, 20, (0, 0), attacks=[bow])
    hero.traits["archery"] = True
    foe = creature("foe", "Foe", "enemy", 16, 20, (4, 0))
    hit = _fight(hero, foe, [11, 4])
    hit.apply(next(action for action in hit.legal_actions() if action.attack_name == "Bow"), "test")
    assert foe.hp < 20

    thrown = AttackOption(
        name="Bola", resolution="attack", attack_bonus=3, range_feet=20, thrown=True,
        damage=[{"type": "bludgeoning", "expr": "1"}],
    )
    thrower = creature("hero", "Hero", "party", 14, 20, (0, 0), attacks=[thrown])
    thrower.traits["archery"] = True
    target = creature("foe", "Foe", "enemy", 16, 20, (3, 0))
    miss = _fight(thrower, target, [11])
    miss.apply(next(action for action in miss.legal_actions() if action.attack_name == "Bola"), "test")
    assert target.hp == 20


def test_great_weapon_fighting_turns_a_one_or_two_into_a_three():
    blade = AttackOption(
        name="Greatsword", resolution="attack", attack_bonus=5, reach_feet=5, two_handed=True,
        damage=[{"type": "slashing", "expr": "1d6"}], ability="strength",
    )
    hero = creature("hero", "Hero", "party", 15, 20, (0, 0), attacks=[blade])
    hero.traits["great_weapon_fighting"] = True
    foe = creature("foe", "Foe", "enemy", 10, 20, (1, 0))
    encounter = _fight(hero, foe, [15, 1])
    encounter.apply(next(action for action in encounter.legal_actions() if action.attack_name == "Greatsword"), "test")
    assert "Great Weapon Fighting treats 1 or 2 as 3." in _text(encounter)
    assert foe.hp == 17


def test_two_weapon_fighting_keeps_the_nick_modifier():
    dagger = AttackOption(
        name="Dagger", resolution="attack", attack_bonus=5, reach_feet=5,
        damage=[{"type": "piercing", "expr": "1d4"}], light=True, ability="dexterity",
    )
    axe = AttackOption(
        name="Handaxe", resolution="attack", attack_bonus=5, reach_feet=5,
        damage=[{"type": "slashing", "expr": "1d6+4"}], mastery="nick", light=True, ability="strength",
    )

    def swing(style: bool):
        hero = creature(
            "hero", "Hero", "party", 15, 20, (0, 0), attacks=[dagger, axe],
            attacks_per_action=1, abilities={"strength": 18},
        )
        if style:
            hero.traits["two_weapon_fighting"] = True
        foe = creature("foe", "Foe", "enemy", 10, 30, (1, 0))
        encounter = _fight(hero, foe, [10, 2, 10, 3])
        encounter.apply(next(action for action in encounter.legal_actions() if action.attack_name == "Dagger"), "test")
        nick = next(action for action in encounter.legal_actions() if action.attack_name == "Handaxe")
        encounter.apply(nick, "test")
        return _text(encounter)

    assert "1d6+4" not in swing(False)
    assert "1d6+4" in swing(True)


def test_heavy_armor_master_reduces_bludgeoning_piercing_and_slashing():
    hero = creature("hero", "Hero", "party", 18, 30, (0, 0))
    hero.traits["heavy_armor_master"] = True
    hero.traits["armor"] = "heavy"
    hero.traits["proficiency"] = 3
    foe = creature("foe", "Foe", "enemy", 12, 20, (1, 0))
    effect = apply_damage(hero, [("slashing", 10)], Scripted([]), source=foe)
    text = " ".join(effect.lines)
    assert effect.damage == 7
    assert hero.hp == 23
    assert "Heavy Armor Master reduces the damage by 3." in text
    assert "resisted" not in text


def test_spell_sniper_adds_sixty_feet_to_an_attack_spell():
    spell = load_spell("fire-bolt")
    hero = creature("hero", "Hero", "party", 12, 20, (0, 0), spells=[spell])
    foe = creature("foe", "Foe", "enemy", 12, 20, (25, 0))
    assert not _creature_in_range(hero, foe, spell)
    hero.traits["spell_sniper"] = True
    assert _creature_in_range(hero, foe, spell)


def test_epic_boons_and_grappler_stay_on_the_note_list():
    hero = public_character(
        {
            "id": 1,
            "name": "Notes",
            "overrideHitPoints": 10,
            "stats": [{"id": stat, "value": 10} for stat in range(1, 7)],
            "classes": [{"level": 1, "definition": {"name": "Fighter"}}],
            "feats": [
                {"definition": {"name": "Grappler"}},
                {"definition": {"name": "Boon of Fate"}},
            ],
        },
        None,
        (0, 0),
        "notes",
    )
    text = " ".join(sheet_for(hero, None).lines)
    assert "Grappler" in text
    assert "Boon of Fate" in text
    assert "not a button" in text


def test_leaving_reach_provokes_an_opportunity_attack():
    hero = creature("hero", "Hero", "party", 15, 20, (0, 0), attacks=[sword()])
    foe = creature("foe", "Foe", "enemy", 12, 20, (1, 0), attacks=[sword()], speed=30)
    encounter = _fight(foe, hero, [15, 4], [foe.id, hero.id])
    move = next(action for action in encounter.legal_actions(foe) if action.destination == (3, 0))
    encounter.apply(move, "test")
    assert foe.position == (3, 0)
    assert foe.hp == 14
    assert hero.reaction_available is False
    assert "makes an opportunity attack" in _text(encounter)


def test_disengage_prevents_an_opportunity_attack():
    hero = creature("hero", "Hero", "party", 15, 20, (0, 0), attacks=[sword()])
    foe = creature("foe", "Foe", "enemy", 12, 20, (1, 0), attacks=[sword()], speed=30)
    encounter = _fight(foe, hero, [15, 4], [foe.id, hero.id])
    disengage = next(
        action for action in encounter.legal_actions(foe) if action.kind == "disengage" and action.budget == "action"
    )
    encounter.apply(disengage, "test")
    assert foe.action_available is False
    assert foe.bonus_available is True
    move = next(action for action in encounter.legal_actions(foe) if action.destination == (3, 0))
    encounter.apply(move, "test")
    assert foe.position == (3, 0)
    assert foe.hp == 20
    assert hero.reaction_available is True
    assert "opportunity attack" not in _text(encounter)


def test_a_push_out_of_reach_does_not_provoke():
    from src.features import push_from

    hero = creature("hero", "Hero", "party", 15, 20, (0, 0), attacks=[sword()])
    foe = creature("foe", "Foe", "enemy", 12, 20, (1, 0), attacks=[sword()], speed=30)
    encounter = _fight(hero, foe, [15, 4], [hero.id, foe.id])
    dest, _visited = push_from(encounter, hero, foe, 2)
    assert dest[0] > 1
    assert foe.position == dest
    assert foe.hp == 20
    assert hero.reaction_available is True
    assert "opportunity attack" not in _text(encounter)


def test_a_ten_foot_weapon_provokes_only_when_that_reach_is_left():
    glaive = AttackOption(
        name="Glaive", resolution="attack", attack_bonus=5, reach_feet=10, heavy=True,
        damage=[{"type": "slashing", "expr": "1d10"}], ability="strength",
    )
    hero = creature("hero", "Hero", "party", 15, 20, (0, 0), attacks=[glaive])
    near = creature("foe", "Foe", "enemy", 12, 30, (1, 0), attacks=[sword()], speed=30)
    stayed = _fight(near, hero, [15, 6], [near.id, hero.id])
    stayed.apply(next(action for action in stayed.legal_actions(near) if action.destination == (2, 0)), "test")
    assert near.position == (2, 0)
    assert near.hp == 30
    assert hero.reaction_available is True

    hero = creature("hero", "Hero", "party", 15, 20, (0, 0), attacks=[glaive])
    far = creature("foe", "Foe", "enemy", 12, 30, (2, 0), attacks=[sword()], speed=30)
    left = _fight(far, hero, [15, 6], [far.id, hero.id])
    left.apply(next(action for action in left.legal_actions(far) if action.destination == (3, 0)), "test")
    assert far.position == (3, 0)
    assert far.hp == 24
    assert hero.reaction_available is False


def test_a_spent_reaction_does_not_provoke():
    hero = creature("hero", "Hero", "party", 15, 20, (0, 0), attacks=[sword()])
    foe = creature("foe", "Foe", "enemy", 12, 20, (1, 0), attacks=[sword()], speed=30)
    encounter = _fight(foe, hero, [15, 4], [foe.id, hero.id])
    encounter.begin_turn()
    hero.reaction_available = False
    move = next(action for action in encounter.legal_actions(foe) if action.destination == (3, 0))
    encounter.apply(move, "test")
    assert foe.position == (3, 0)
    assert foe.hp == 20
    assert hero.reaction_available is False


def test_sentinel_opportunity_attack_stops_the_mover_on_that_square():
    hero = creature("hero", "Hero", "party", 15, 20, (0, 0), attacks=[sword()])
    hero.traits["sentinel"] = True
    foe = creature("foe", "Foe", "enemy", 12, 30, (1, 0), attacks=[sword()], speed=30)
    encounter = _fight(foe, hero, [15, 4], [foe.id, hero.id])
    encounter.apply(next(action for action in encounter.legal_actions(foe) if action.destination == (3, 0)), "test")
    assert foe.position == (1, 0)
    assert foe.movement_left == 0
    assert foe.hp == 24
    assert "uses Sentinel" in _text(encounter)
    assert "stops" in _text(encounter)
    assert "moves to" not in _text(encounter)


def test_sentinel_attacks_a_creature_that_disengages_and_sets_speed_to_zero():
    escape = AttackOption(name="Nimble Escape", resolution="disengage", activation="bonus")
    foe = creature("foe", "Foe", "enemy", 12, 20, (1, 0), attacks=[escape, sword()], speed=30)
    plain = creature("hero", "Hero", "party", 15, 20, (0, 0), attacks=[sword()])
    quiet = _fight(plain, foe, [], [foe.id, plain.id])
    quiet.apply(
        next(action for action in quiet.legal_actions(foe) if action.kind == "disengage" and action.budget == "bonus"),
        "test",
    )
    assert "Sentinel" not in _text(quiet)
    assert foe.movement_left == 30

    hero = creature("hero", "Hero", "party", 15, 20, (0, 0), attacks=[sword()])
    hero.traits["sentinel"] = True
    marked = creature("foe", "Foe", "enemy", 12, 30, (1, 0), attacks=[escape, sword()], speed=30)
    encounter = _fight(hero, marked, [15, 4], [marked.id, hero.id])
    encounter.apply(
        next(action for action in encounter.legal_actions(marked) if action.kind == "disengage" and action.budget == "bonus"),
        "test",
    )
    assert "uses Sentinel" in _text(encounter)
    assert marked.hp < 30
    assert marked.movement_left == 0
    assert "stops" in _text(encounter)


def test_war_caster_casts_when_a_creature_leaves_reach():
    bolt = load_spell("fire-bolt")
    foe = creature("foe", "Foe", "enemy", 12, 30, (1, 0), attacks=[sword()], speed=30)
    plain = creature("hero", "Hero", "party", 12, 20, (0, 0), spells=[bolt], spell_attack_bonus=5)
    quiet = _fight(plain, foe, [], [foe.id, plain.id])
    quiet.apply(next(action for action in quiet.legal_actions(foe) if action.destination == (3, 0)), "test")
    assert foe.hp == 30
    assert "as a reaction" not in _text(quiet)

    hero = creature("hero", "Hero", "party", 12, 20, (0, 0), spells=[bolt], spell_attack_bonus=5)
    hero.traits["war_caster"] = True
    mover = creature("foe", "Foe", "enemy", 12, 30, (1, 0), attacks=[sword()], speed=30)
    encounter = _fight(hero, mover, [16, 5], [mover.id, hero.id])
    encounter.apply(next(action for action in encounter.legal_actions(mover) if action.destination == (3, 0)), "test")
    assert "casts Fire Bolt as a reaction" in _text(encounter)
    assert mover.hp < 30
    assert mover.position == (3, 0)


def test_polearm_master_attacks_a_creature_that_enters_reach():
    glaive = AttackOption(
        name="Glaive", resolution="attack", attack_bonus=5, reach_feet=10, heavy=True,
        damage=[{"type": "slashing", "expr": "1d10"}], ability="strength",
    )
    foe = creature("foe", "Foe", "enemy", 12, 30, (3, 0), attacks=[sword()], speed=30)
    plain = creature("hero", "Hero", "party", 15, 20, (0, 0), attacks=[glaive])
    quiet = _fight(plain, foe, [], [foe.id, plain.id])
    quiet.apply(next(action for action in quiet.legal_actions(foe) if action.destination == (2, 0)), "test")
    assert foe.hp == 30

    hero = creature("hero", "Hero", "party", 15, 20, (0, 0), attacks=[glaive])
    hero.traits["polearm_master"] = True
    mover = creature("foe", "Foe", "enemy", 12, 30, (3, 0), attacks=[sword()], speed=30)
    encounter = _fight(hero, mover, [14, 6], [mover.id, hero.id])
    encounter.apply(next(action for action in encounter.legal_actions(mover) if action.destination == (2, 0)), "test")
    assert "uses Polearm Master" in _text(encounter)
    assert "Glaive" in _text(encounter)
    assert mover.hp == 24
    assert mover.movement_left == 25
    assert "stops" not in _text(encounter)


def test_lucky_spends_a_point_on_the_attackers_miss_and_not_the_reaction():
    hero = creature("hero", "Hero", "party", 14, 20, (0, 0), attacks=[sword()])
    hero.traits["lucky"] = True
    hero.uses_left["Luck"] = 1
    hero.uses_max["Luck"] = 1
    foe = creature("foe", "Foe", "enemy", 18, 20, (1, 0))
    encounter = _fight(hero, foe, [10, 16, 3])
    encounter.apply(next(action for action in encounter.legal_actions() if action.attack_name == "Sword"), "test")
    assert "spends a luck point" in _text(encounter)
    assert foe.hp < 20
    assert hero.uses_left["Luck"] == 0
    assert hero.reaction_available is True


def test_defensive_duelist_adds_proficiency_against_a_melee_hit():
    rapier = AttackOption(
        name="Rapier", resolution="attack", attack_bonus=5, reach_feet=5, finesse=True,
        damage=[{"type": "piercing", "expr": "1d8"}], ability="dexterity",
    )
    hero = creature("hero", "Hero", "party", 14, 20, (0, 0), attacks=[sword()])
    foe = creature("foe", "Foe", "enemy", 15, 20, (1, 0), attacks=[rapier])
    foe.traits["defensive_duelist"] = True
    foe.traits["proficiency"] = 3
    encounter = _fight(hero, foe, [11])
    encounter.apply(next(action for action in encounter.legal_actions() if action.attack_name == "Sword"), "test")
    assert "Defensive Duelist" in _text(encounter)
    assert foe.hp == 20
    assert foe.traits.get("duelist_ac") == 3


def test_protection_imposes_disadvantage_on_the_triggering_attack():
    hero = creature("hero", "Hero", "party", 14, 20, (0, 0), attacks=[sword()])
    foe = creature("foe", "Foe", "enemy", 18, 20, (1, 0))
    guard = creature("guard", "Guard", "enemy", 16, 20, (1, 1), attacks=[sword()])
    guard.traits["protection"] = True
    guard.traits["has_shield"] = True
    encounter = Encounter([hero, foe, guard], seed=1)
    _prepare(encounter, ["hero", "foe", "guard"], [15, 1])
    encounter.reaction_policy = "heuristic"
    encounter.apply(next(action for action in encounter.legal_actions() if action.attack_name == "Sword"), "test")
    assert "uses Protection" in _text(encounter)
    assert foe.hp == 20
    assert foe.traits.get("protected_by") == "guard"


def test_interception_reduces_damage_to_an_adjacent_ally():
    hero = creature("hero", "Hero", "party", 14, 20, (0, 0), attacks=[sword()])
    foe = creature("foe", "Foe", "enemy", 10, 6, (1, 0))
    guard = creature("guard", "Guard", "enemy", 16, 20, (1, 1), attacks=[sword()])
    guard.traits["interception"] = True
    guard.traits["proficiency"] = 2
    encounter = Encounter([hero, foe, guard], seed=1)
    _prepare(encounter, ["hero", "foe", "guard"], [15, 4, 6])
    encounter.reaction_policy = "heuristic"
    encounter.apply(next(action for action in encounter.legal_actions() if action.attack_name == "Sword"), "test")
    assert "uses Interception" in _text(encounter)
    assert foe.hp == 6


def test_charger_adds_a_d8_after_ten_feet_of_movement():
    hero = creature("hero", "Hero", "party", 15, 20, (0, 0), attacks=[sword()], speed=30)
    hero.traits["charger"] = True
    foe = creature("foe", "Foe", "enemy", 10, 30, (3, 0))
    encounter = _fight(hero, foe, [15, 3, 4])
    encounter.apply(next(action for action in encounter.legal_actions() if action.destination == (2, 0)), "test")
    encounter.apply(next(action for action in encounter.legal_actions() if action.attack_name == "Sword"), "test")
    assert "Charger 1d8" in _text(encounter)
    assert foe.hp < 30


def test_healer_rerolls_a_one_on_a_healing_spell():
    spell = load_spell("cure-wounds")
    hero = creature(
        "hero", "Hero", "party", 12, 20, (0, 0), spells=[spell], slots={1: 1}, spellcasting_mod=3,
    )
    hero.traits["healer"] = True
    ally = creature("ally", "Ally", "party", 12, 20, (1, 0))
    foe = creature("foe", "Foe", "enemy", 12, 20, (4, 0), attacks=[sword()])
    encounter = Encounter([hero, ally, foe], seed=1)
    _prepare(encounter, ["hero", "ally", "foe"], [1, 4])
    ally.hp = 5
    encounter.apply(next(action for action in encounter.legal_actions() if action.spell_index == "cure-wounds"), "test")
    assert ally.hp == 5 + 7


def test_piercer_rerolls_one_low_damage_die():
    rapier = AttackOption(
        name="Rapier", resolution="attack", attack_bonus=5, reach_feet=5, finesse=True,
        damage=[{"type": "piercing", "expr": "1d8"}], ability="dexterity",
    )
    hero = creature("hero", "Hero", "party", 14, 20, (0, 0), attacks=[rapier])
    hero.traits["piercer"] = True
    foe = creature("foe", "Foe", "enemy", 10, 20, (1, 0))
    encounter = _fight(hero, foe, [12, 1, 6])
    encounter.apply(next(action for action in encounter.legal_actions() if action.attack_name == "Rapier"), "test")
    assert "Piercer rerolls one damage die." in _text(encounter)
    assert foe.hp == 14


def test_shield_master_bashes_on_the_first_melee_hit():
    hero = creature("hero", "Hero", "party", 18, 20, (0, 0), attacks=[sword()])
    hero.traits["shield_master"] = True
    hero.traits["has_shield"] = True
    hero.traits["proficiency"] = 2
    foe = creature("foe", "Foe", "enemy", 10, 20, (1, 0))
    encounter = _fight(hero, foe, [12, 3, 5])
    encounter.apply(next(action for action in encounter.legal_actions() if action.attack_name == "Sword"), "test")
    assert "knocked prone by Shield Bash" in _text(encounter)
    assert "prone" in foe.conditions


def test_guarded_mind_turns_a_failed_mental_save_into_a_success():
    gaze = AttackOption(
        name="Gaze", resolution="save", dc_ability="wis", dc_value=15, dc_success="none",
        damage=[{"type": "psychic", "expr": "1d6"}],
    )
    foe = creature("foe", "Foe", "enemy", 12, 20, (1, 0), attacks=[gaze])
    hero = creature("hero", "Hero", "party", 15, 20, (0, 0))
    hero.traits["mage_slayer"] = True
    hero.uses_left["Guarded Mind"] = 1
    hero.uses_max["Guarded Mind"] = 1
    encounter = _fight(hero, foe, [1, 4], [foe.id, hero.id])
    encounter.apply(next(action for action in encounter.legal_actions(foe) if action.attack_name == "Gaze"), "test")
    assert "Guarded Mind" in _text(encounter)
    assert hero.hp == 20
    assert hero.uses_left["Guarded Mind"] == 0


def test_dual_wielder_offers_a_different_melee_weapon():
    dagger = AttackOption(
        name="Dagger", resolution="attack", attack_bonus=5, reach_feet=5,
        damage=[{"type": "piercing", "expr": "1d4"}], light=True, ability="dexterity",
    )
    sword_row = AttackOption(
        name="Longsword", resolution="attack", attack_bonus=6, reach_feet=5,
        damage=[{"type": "slashing", "expr": "1d8+4"}], ability="strength",
    )
    hero = creature(
        "hero", "Hero", "party", 15, 20, (0, 0), attacks=[dagger, sword_row],
        abilities={"strength": 18},
    )
    hero.traits["dual_wielder"] = True
    foe = creature("foe", "Foe", "enemy", 10, 30, (1, 0))
    encounter = _fight(hero, foe, [12, 2, 12, 3])
    encounter.apply(next(action for action in encounter.legal_actions() if action.attack_name == "Dagger"), "test")
    bonus = [action for action in encounter.legal_actions() if action.tag == "dual"]
    assert bonus
    assert bonus[0].budget == "bonus"
    assert bonus[0].attack_name == "Longsword"
    encounter.apply(bonus[0], "test")
    assert "1d8+4" not in _text(encounter)
    assert hero.traits.get("nick_used") is True


def test_telekinetic_shove_pushes_on_a_failed_strength_save():
    hero = creature("hero", "Hero", "party", 12, 20, (0, 0))
    hero.traits["telekinetic"] = 15
    foe = creature("foe", "Foe", "enemy", 12, 20, (2, 0), attacks=[sword()])
    encounter = _fight(hero, foe, [2])
    shove = next(action for action in encounter.legal_actions() if action.attack_name == "Telekinetic Shove")
    assert shove.budget == "bonus"
    encounter.apply(shove, "test")
    assert "shoved" in _text(encounter)
    assert foe.position != (2, 0)
