"""Stat blocks and saved public characters. No live request."""

import base64
import json
import re

import pytest
from streamlit.testing.v1 import AppTest

from src.actions import attack_window, legal_actions, nick_weapon_name
from src.bestiary import AttackOption
from src.api_client import CatalogError, load_catalog
from src.board import TOKENS_DIR, battle_payload, token_image
from src.combatant import Combatant, blank_abilities, save_bonuses_for
from src.ddb import (
    PORTRAIT_BYTES,
    fetch_public_character,
    parse_character_id,
    public_character,
    sample_characters,
    save_character_portrait,
)
from src.ai import _decision_menu, _needs_model, heuristic_advice
from src.conditions import base_speed, marks_of
from src.encounter import Encounter
from src.paths import CACHE, IMPORTS, ROOT, SAMPLES
from src.presets import four_goblins
from src.sheets import sheet_for
from tests.test_combat import _prepare, cache_spell, creature, finish_turn, sword

_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def test_goblin_sheet_names_the_weapons_and_has_no_beyond_link():
    catalog = load_catalog()
    goblin = next(creature for creature in four_goblins(catalog).combatants if creature.id == "goblin-warrior-1")
    sheet = sheet_for(goblin, catalog)
    text = " ".join(sheet.lines)
    assert sheet.kind_label == "SRD stat block"
    assert sheet.url is None
    assert "Scimitar" in text
    assert "Shortbow" in text


def test_character_link_parsing_and_a_private_sheet():
    assert parse_character_id("https://www.dndbeyond.com/characters/12345/AbC") == 12345
    assert parse_character_id("99") == 99
    assert parse_character_id("not-a-character") is None

    def fetcher(_url):
        return {"success": False, "_status": 403, "data": {"serverMessage": "Unauthorized Access Attempt."}}

    with pytest.raises(CatalogError, match="not public"):
        fetch_public_character(1, fetcher=fetcher)


def _zealot() -> dict:
    return {
        "id": 77,
        "name": "Synthetic Zealot",
        "baseHitPoints": 40,
        "stats": [
            {"id": 1, "value": 15}, {"id": 2, "value": 14}, {"id": 3, "value": 15},
            {"id": 4, "value": 8}, {"id": 5, "value": 10}, {"id": 6, "value": 8},
        ],
        "race": {"fullName": "Human", "weightSpeeds": {"normal": {"walk": 30}}},
        "classes": [{
            "level": 5,
            "definition": {"name": "Barbarian"},
            "subclassDefinition": {
                "name": "Path of the Zealot",
                "classFeatures": [
                    {"definition": {"id": 900, "name": "Divine Fury", "requiredLevel": 3}},
                    {"definition": {"id": 901, "name": "Warrior of the Gods", "requiredLevel": 3}},
                    {"definition": {"id": 902, "name": "Fanatical Focus", "requiredLevel": 6}},
                ],
            },
            "classFeatures": [
                {"definition": {"id": 800, "name": "Rage", "requiredLevel": 1}},
                {"definition": {"id": 801, "name": "Unarmored Defense", "requiredLevel": 1}},
                {"definition": {"id": 802, "name": "Reckless Attack", "requiredLevel": 2}},
                {"definition": {"id": 803, "name": "Danger Sense", "requiredLevel": 2}},
                {"definition": {"id": 804, "name": "Primal Knowledge", "requiredLevel": 3}},
                {"definition": {"id": 805, "name": "Extra Attack", "requiredLevel": 5}},
            ],
        }],
        "modifiers": {
            "class": [
                {"type": "set", "subType": "unarmored-armor-class", "value": None, "statId": 3, "isGranted": True},
                {"type": "set", "subType": "extra-attacks", "value": 1, "isGranted": True},
                {"type": "bonus", "subType": "speed", "value": 10, "isGranted": True, "restriction": "while you aren't wearing Heavy armor."},
                {"type": "proficiency", "subType": "strength-saving-throws", "isGranted": True},
                {"type": "proficiency", "subType": "constitution-saving-throws", "isGranted": True},
            ],
            "feat": [
                {"type": "bonus", "subType": "strength-score", "value": 1, "isGranted": True, "componentId": 50},
                {"type": "bonus", "subType": "strength-score", "value": 2, "isGranted": False, "componentId": 40},
                {"type": "bonus", "subType": "constitution-score", "value": 1, "isGranted": False, "componentId": 41},
                {"type": "bonus", "subType": "hit-points-per-level", "value": 2, "isGranted": True},
            ],
        },
        "choices": {
            "feat": [
                {"componentId": 40, "optionValue": 6224, "label": "Choose +2 to Strength, Constitution, or Wisdom"},
            ],
            "choiceDefinitions": [{"id": "asi", "options": [{"id": 6224, "label": "Strength Score"}]}],
        },
        "feats": [
            {"definition": {"id": 50, "name": "Great Weapon Master"}},
            {"definition": {"id": 60, "name": "Savage Attacker"}},
        ],
        "inventory": [
            {"id": 1, "equipped": True, "definition": {
                "name": "Glaive", "filterType": "Weapon", "attackType": 1,
                "damage": {"diceString": "1d10"}, "damageType": "Slashing",
                "properties": [{"name": "Heavy"}, {"name": "Reach"}, {"name": "Graze"}],
            }},
            {"id": 2, "equipped": True, "definition": {
                "name": "Bola", "filterType": "Weapon", "attackType": 2,
                "damage": None, "fixedDamage": 1, "damageType": "Bludgeoning",
                "range": 20, "longRange": 40,
                "properties": [{"name": "Finesse"}, {"name": "Thrown"}, {"name": "Topple"}],
            }},
        ],
        "actions": {
            "class": [
                {"name": "Rage (Enter)", "componentId": 800, "activation": {"activationType": 3},
                 "limitedUse": {"maxUses": 3, "numberUsed": 0}, "snippet": "You can enter Rage."},
                {"name": "Warrior of the Gods: Expend Dice", "componentId": 901, "activation": {"activationType": 3},
                 "dice": {"diceString": "1d12"}, "limitedUse": {"maxUses": 4, "numberUsed": 0},
                 "snippet": "As a Bonus Action, you can expend dice from the pool to heal yourself."},
                {"name": "Plant Totem", "componentId": 13774176, "activation": {"activationType": 3},
                 "snippet": "Plant a totem."},
            ],
            "feat": [
                {"name": "Hew", "componentId": 50, "activation": {"activationType": 3},
                 "snippet": "After a critical hit, make one attack."},
                {"name": "Heavy Weapon Mastery", "componentId": 50, "activation": {"activationType": 1},
                 "snippet": "When you hit with a Heavy weapon as part of the Attack action, deal extra {{proficiency}} damage."},
            ],
        },
    }


def test_public_character_uses_chosen_scores_and_owned_actions(tmp_path):
    catalog = load_catalog()
    payload = _zealot()

    def fetcher(_url):
        return {"success": True, "data": payload}

    saved = fetch_public_character(77, fetcher=fetcher, imports_dir=tmp_path)
    assert json.loads((tmp_path / "77.json").read_text(encoding="utf-8"))["name"] == "Synthetic Zealot"
    hero = public_character(saved, catalog, (1, 1), "ddb-77")
    assert hero.abilities["strength"] == 18
    assert hero.abilities["constitution"] == 15
    assert hero.ac == 14
    assert hero.max_hp == 60
    assert hero.speed == 40
    assert hero.attacks_per_action == 2
    glaive = hero.attack_named("Glaive")
    assert glaive.attack_bonus == 7
    assert glaive.damage[0]["expr"] == "1d10+4"
    assert glaive.heavy is True
    assert glaive.mastery == "graze"
    assert glaive.reach_feet == 10
    bola = hero.attack_named("Bola")
    assert bola.damage[0]["expr"] == "1"
    assert bola.mastery == "topple"
    assert hero.uses_max["Rage"] == 3
    assert hero.traits["rage_bonus"] == 2
    assert hero.traits["reckless_feature"] is True
    assert hero.traits["danger_sense"] is True
    assert hero.traits["divine_fury"]["flat"] == 2
    assert hero.uses_max["Warrior of the Gods"] == 4
    assert hero.save_bonuses["strength"] == 7
    sheet = sheet_for(hero, catalog)
    text = " ".join(sheet.lines)
    assert sheet.url == "https://www.dndbeyond.com/characters/77"
    assert "Savage Attacker" in text
    assert "Primal Knowledge" in text
    assert "Plant Totem" not in text
    assert "Fanatical Focus" not in text
    assert "Extra Attack" not in text
    assert "radiant" in text.lower()


def test_heavy_armor_and_a_shield_keep_their_totals():
    catalog = load_catalog()
    knight = public_character({
        "id": 8,
        "name": "Armored",
        "baseHitPoints": 40,
        "stats": [{"id": 1, "value": 16}, {"id": 2, "value": 12}, {"id": 3, "value": 14},
                  {"id": 4, "value": 8}, {"id": 5, "value": 10}, {"id": 6, "value": 10}],
        "classes": [{"level": 5, "definition": {"name": "Fighter"}, "classFeatures": [
            {"definition": {"name": "Extra Attack", "requiredLevel": 5}},
        ]}],
        "inventory": [
            {"id": 1, "equipped": True, "definition": {"name": "Chain Mail", "filterType": "Armor", "armorClass": 16, "armorTypeId": 3}},
            {"id": 2, "equipped": True, "definition": {"name": "Shield", "filterType": "Shield"}},
            {"id": 3, "equipped": True, "definition": {
                "name": "Longsword", "filterType": "Weapon", "attackType": 1,
                "damage": {"diceString": "1d8"}, "damageType": "Slashing",
            }},
        ],
    }, catalog, (1, 1), "ddb-8")
    assert knight.ac == 18
    assert knight.max_hp == 50
    assert knight.attacks[0].attack_bonus == 6
    assert knight.attacks[0].damage[0]["expr"] == "1d8+3"
    assert knight.attacks_per_action == 2


def test_a_beyond_shield_adds_two_on_top_of_medium_armor():
    catalog = load_catalog()
    cleric = public_character({
        "id": 9,
        "name": "Cleric",
        "baseHitPoints": 10,
        "stats": [{"id": 1, "value": 8}, {"id": 2, "value": 14}, {"id": 3, "value": 16},
                  {"id": 4, "value": 10}, {"id": 5, "value": 17}, {"id": 6, "value": 8}],
        "classes": [{"level": 5, "definition": {"name": "Cleric"}}],
        "inventory": [
            {"id": 1, "equipped": True, "definition": {
                "name": "Shield", "filterType": "Armor", "type": "Shield",
                "armorClass": 2, "armorTypeId": 4, "baseArmorName": "Shield",
            }},
            {"id": 2, "equipped": True, "definition": {
                "name": "Chain Shirt", "filterType": "Armor", "type": "",
                "armorClass": 13, "armorTypeId": 2, "baseArmorName": "Chain Shirt",
            }},
        ],
    }, catalog, (0, 0), "ddb-9")
    assert cleric.ac == 17
    assert cleric.traits["has_shield"] is True


class _Rolls:
    def __init__(self, rolls):
        self.rolls = list(rolls)

    def randint(self, low, high):
        value = self.rolls.pop(0)
        if not low <= value <= high:
            raise AssertionError(f"{value} is outside {low}-{high}")
        return value


def _foe(hit_points: int) -> Combatant:
    abilities = blank_abilities()
    return Combatant(
        id="foe", name="Foe", side="enemy", ac=10, max_hp=hit_points, hp=hit_points, speed=30,
        abilities=abilities, save_bonuses=save_bonuses_for(abilities, [], 2), position=(2, 1),
    )


def _ready(hero, foe) -> Encounter:
    encounter = Encounter([hero, foe], seed=1)
    encounter.start()
    encounter.order = [hero.id, foe.id]
    encounter.turn_index = 0
    encounter.round = 1
    encounter._prepared = None
    return encounter


def test_rage_adds_damage_and_a_heavy_hit_adds_proficiency():
    catalog = load_catalog()
    hero = public_character(_zealot(), catalog, (1, 1), "ddb-77")
    encounter = _ready(hero, _foe(40))
    assert any(action.attack_name == "Rage" and action.label == "Rage" for action in encounter.legal_actions())
    assert not any("Totem" in action.label for action in encounter.legal_actions())
    encounter.apply(next(action for action in encounter.legal_actions() if action.attack_name == "Rage"), "test")
    assert hero.traits["rage"] is True
    assert hero.uses_left["Rage"] == 2
    assert "slashing" in hero.resistances
    encounter.rng = _Rolls([12, 5, 1, 3])
    encounter.apply(next(action for action in encounter.legal_actions() if action.attack_name == "Glaive"), "test")
    text = " ".join(encounter.log)
    assert "Rage +2" in text
    assert "Great Weapon Master +3" in text
    assert "Divine Fury: 1d6 dice 3, modifier +2, total 5 radiant." in text


def test_divine_fury_lists_both_dice_on_a_critical_hit():
    catalog = load_catalog()
    hero = public_character(_zealot(), catalog, (1, 1), "ddb-77")
    encounter = _ready(hero, _foe(40))
    encounter.apply(next(action for action in encounter.legal_actions() if action.attack_name == "Rage"), "test")
    encounter.rng = _Rolls([20, 6, 4, 1, 1, 3, 5])
    encounter.apply(next(action for action in encounter.legal_actions() if action.attack_name == "Glaive"), "test")
    text = " ".join(encounter.log)
    assert "Critical hit" in text
    assert "Dice are doubled. Flat bonuses are not." in text
    assert "1d10+4 critical: dice 6 + 4, modifier +4 not doubled, total 14 slashing." in text
    assert "Divine Fury critical: 1d6 dice 3 + 5, modifier +2 not doubled, total 10 radiant." in text
    assert "takes 19 slashing and 10 radiant" in text


def test_warrior_heal_spends_a_use_and_hew_follows_a_drop():
    catalog = load_catalog()
    hero = public_character(_zealot(), catalog, (1, 1), "ddb-77")
    encounter = _ready(hero, _foe(30))
    hero.hp = 40
    encounter.rng = _Rolls([6])
    encounter.apply(next(action for action in encounter.legal_actions() if action.attack_name == "Warrior of the Gods"), "test")
    assert hero.hp == 46
    assert hero.uses_left["Warrior of the Gods"] == 3
    assert hero.bonus_available is False

    fresh = public_character(_zealot(), catalog, (1, 1), "ddb-77")
    victim = _foe(1)
    other = _foe(20)
    other.id = "other"
    other.name = "Other"
    other.position = (1, 2)
    fight = Encounter([fresh, victim, other], seed=1)
    fight.start()
    fight.order = [fresh.id, victim.id, other.id]
    fight.turn_index = 0
    fight.round = 1
    fight._prepared = None
    fight.rng = _Rolls([15, 1, 1])
    fight.apply(next(action for action in fight.legal_actions() if action.attack_name == "Glaive"), "test")
    hew = [action for action in fight.legal_actions() if action.tag == "hew"]
    assert hew
    assert hew[0].budget == "bonus"
    assert hew[0].attack_name == "Glaive"
    assert hew[0].target_id == "other"


def test_saved_sample_barbarian_matches_the_public_sheet():
    catalog = load_catalog()
    payload = json.loads((SAMPLES / "173717514.json").read_text(encoding="utf-8"))
    hero = public_character(payload, catalog, (2, 5), "ddb-173717514")
    assert sample_characters()[0][0] == "173717514"
    assert hero.name == "Sample Barbarian"
    assert hero.abilities["strength"] == 18
    assert hero.abilities["constitution"] == 16
    assert hero.ac == 15
    assert hero.max_hp == 65
    assert hero.speed == 40
    assert hero.attacks_per_action == 2
    glaive = hero.attack_named("Glaive")
    assert glaive.attack_bonus == 7
    assert glaive.damage[0]["expr"] == "1d10+4"
    assert glaive.heavy is True
    assert glaive.mastery == "graze"
    bola = hero.attack_named("Bola")
    assert bola.damage[0]["expr"] == "1"
    assert bola.mastery == "topple"
    assert hero.uses_max["Rage"] == 3
    assert hero.traits["rage_bonus"] == 2
    text = " ".join(sheet_for(hero, catalog).lines)
    assert "Plant Totem" not in text
    assert "Savage Attacker" in text
    assert "radiant" in text.lower()


def _spell(name: str, level: int, *, prepared=False, always=False, known=False) -> dict:
    return {
        "prepared": prepared,
        "alwaysPrepared": always,
        "countsAsKnownSpell": known,
        "definition": {"name": name, "level": level},
    }


def test_a_bard_imports_known_spells_and_the_class_slot_table():
    catalog = load_catalog()
    names = [
        ("Starry Wisp", 0), ("True Strike", 0), ("Mage Hand", 0),
        ("Cure Wounds", 1), ("Healing Word", 1), ("Sleep", 1),
        ("Cloud of Daggers", 2), ("Hold Person", 2), ("Mirror Image", 2), ("Shatter", 2),
        ("Hypnotic Pattern", 3), ("Slow", 3),
    ]
    table = [[0] * 9 for _ in range(6)]
    table[5] = [4, 3, 2, 0, 0, 0, 0, 0, 0]
    hero = public_character({
        "id": 173718057,
        "name": "Sample Bard",
        "baseHitPoints": 28,
        "stats": [
            {"id": 1, "value": 10}, {"id": 2, "value": 13}, {"id": 3, "value": 14},
            {"id": 4, "value": 10}, {"id": 5, "value": 12}, {"id": 6, "value": 14},
        ],
        "race": {"fullName": "Human"},
        "classes": [{
            "id": 242489937,
            "level": 5,
            "definition": {
                "name": "Bard",
                "spellCastingAbilityId": 6,
                "spellPrepareType": None,
                "spellRules": {"levelPreparedSpellMaxes": [], "levelSpellSlots": table},
            },
            "subclassDefinition": {"name": "College of Lore"},
        }],
        "spells": {"class": [], "race": [], "item": [], "feat": []},
        "classSpells": [{
            "characterClassId": 242489937,
            "spells": [_spell(name, level, known=True) for name, level in names],
        }],
        "spellSlots": [{"level": level, "used": 0, "available": 0} for level in range(1, 10)],
        "pactMagic": [{"level": level, "used": 0, "available": 0} for level in range(1, 6)],
    }, catalog, (2, 5), "ddb-173718057")
    assert {spell.name for spell in hero.spells} == {
        "Starry Wisp", "Cure Wounds", "Healing Word", "Shatter",
        "Sleep", "Hold Person", "Hypnotic Pattern",
    }
    assert hero.slots == {1: 4, 2: 3, 3: 2}
    notes = " ".join(hero.traits["unresolved"])
    assert "not prepared" not in notes
    assert "Cloud of Daggers: not in the local spell cache" in notes


def test_a_prepared_caster_keeps_an_unprepared_spell_and_a_listed_slot():
    catalog = load_catalog()
    table = [[0] * 9 for _ in range(6)]
    table[5] = [4, 3, 2, 0, 0, 0, 0, 0, 0]
    hero = public_character({
        "id": 8,
        "name": "Preparer",
        "baseHitPoints": 20,
        "stats": [{"id": index, "value": 10} for index in range(1, 7)],
        "classes": [{
            "id": 5,
            "level": 5,
            "definition": {
                "name": "Cleric",
                "spellCastingAbilityId": 5,
                "spellRules": {
                    "levelPreparedSpellMaxes": [0, 4, 5, 6, 7, 9],
                    "levelSpellSlots": table,
                },
            },
        }],
        "classSpells": [{
            "characterClassId": 5,
            "spells": [
                _spell("Cure Wounds", 1, known=True),
                _spell("Healing Word", 1, prepared=True),
            ],
        }],
        "spellSlots": [
            {"level": 1, "used": 0, "available": 2},
            {"level": 2, "used": 0, "available": 0},
            {"level": 3, "used": 0, "available": 0},
        ],
    }, catalog, (0, 0), "ddb-8")
    assert {spell.name for spell in hero.spells} == {"Healing Word"}
    assert "Cure Wounds: on the sheet, not prepared" in hero.traits["unresolved"]
    assert hero.slots == {1: 2}


def test_a_feature_grant_is_castable_when_the_class_list_is_not():
    catalog = load_catalog()
    hero = public_character({
        "id": 9,
        "name": "Initiate",
        "baseHitPoints": 20,
        "stats": [
            {"id": 1, "value": 10}, {"id": 2, "value": 10}, {"id": 3, "value": 10},
            {"id": 4, "value": 10}, {"id": 5, "value": 16}, {"id": 6, "value": 10},
        ],
        "classes": [{
            "id": 5,
            "level": 5,
            "definition": {
                "name": "Cleric",
                "spellCastingAbilityId": 5,
                "spellPrepareType": 1,
                "spellRules": {"levelPreparedSpellMaxes": [0, 4]},
            },
        }],
        "feats": [{"definition": {"id": 1, "name": "Magic Initiate (Wizard)"}}],
        "spells": {
            "class": [_spell("Bane", 1), _spell("Healing Word", 1)],
            "race": [_spell("Sacred Flame", 0)],
            "feat": [
                _spell("Fire Bolt", 0),
                _spell("True Strike", 0),
                _spell("Cure Wounds", 1),
                {
                    **_spell("Shield", 1),
                    "usesSpellSlot": False,
                    "limitedUse": {"maxUses": 1, "numberUsed": 0},
                    "spellCastingAbilityId": 5,
                },
            ],
        },
        "classSpells": [{
            "characterClassId": 5,
            "spells": [
                _spell("Cure Wounds", 1, known=True),
                _spell("Healing Word", 1, prepared=True),
                _spell("Guidance", 0, known=True),
                _spell("Command", 1, known=True),
            ],
        }],
        "spellSlots": [{"level": 1, "used": 0, "available": 2}],
    }, catalog, (0, 0), "ddb-9")
    assert {spell.name for spell in hero.spells} >= {
        "Fire Bolt", "Shield", "Sacred Flame", "Cure Wounds", "Healing Word",
    }
    notes = hero.traits["unresolved"]
    assert "Fire Bolt: on the sheet, not prepared" not in notes
    assert "Shield: on the sheet, not prepared" not in notes
    assert "Cure Wounds: on the sheet, not prepared" not in notes
    assert "Sacred Flame: on the sheet, not prepared" not in notes
    assert "Guidance: on the sheet, not prepared" not in notes
    assert any(note.startswith("Guidance:") and "not prepared" not in note for note in notes)
    assert "Command: on the sheet, not prepared" in notes
    assert "Bane: on the sheet, not prepared" in notes
    assert any(note.startswith("True Strike:") and "not prepared" not in note for note in notes)
    assert hero.traits["free_spells"]["shield"] == "Shield"
    assert hero.uses_max["Shield"] == 1
    assert hero.uses_left["Shield"] == 1
    assert "Magic Initiate (Wizard)" not in notes
    assert hero.spell_attack_bonus == 6


def test_a_granted_spell_uses_the_ability_written_on_the_spell():
    catalog = load_catalog()
    hero = public_character({
        "id": 10,
        "name": "Fighter",
        "baseHitPoints": 12,
        "stats": [{"id": 4, "value": 14}],
        "classes": [{"id": 1, "level": 1, "definition": {"name": "Fighter"}}],
        "spells": {"feat": [{
            **_spell("Fire Bolt", 0),
            "spellCastingAbilityId": 4,
        }]},
    }, catalog, (0, 0), "ddb-10")
    assert {spell.name for spell in hero.spells} == {"Fire Bolt"}
    assert hero.spell_attack_bonus == 4
    assert hero.spell_save_dc == 12
    assert "Spell attack bonus was not in the JSON." not in hero.traits["unresolved"]


def test_a_preparing_class_keeps_a_known_cantrip_and_a_feature_grant():
    catalog = load_catalog()
    hero = public_character({
        "id": 11,
        "name": "Stars",
        "baseHitPoints": 30,
        "stats": [
            {"id": 1, "value": 10}, {"id": 2, "value": 10}, {"id": 3, "value": 10},
            {"id": 4, "value": 10}, {"id": 5, "value": 15}, {"id": 6, "value": 10},
        ],
        "classes": [{
            "id": 7,
            "level": 5,
            "definition": {
                "name": "Druid",
                "spellCastingAbilityId": 5,
                "spellPrepareType": 1,
                "spellRules": {"levelPreparedSpellMaxes": [0, 4]},
            },
        }],
        "spells": {
            "class": [
                _spell("Bane", 1),
                {**_spell("Guiding Bolt", 1), "componentId": 10292558, "usesSpellSlot": True},
                {
                    **_spell("Guiding Bolt", 1),
                    "componentId": 10292558,
                    "usesSpellSlot": False,
                    "limitedUse": {"maxUses": 0, "numberUsed": 0, "statModifierUsesId": 5},
                },
            ],
        },
        "classSpells": [{
            "characterClassId": 7,
            "spells": [
                _spell("Poison Spray", 0, known=True),
                _spell("Cure Wounds", 1, known=True),
            ],
        }],
        "spellSlots": [{"level": 1, "used": 0, "available": 2}],
    }, catalog, (0, 0), "ddb-11")
    names = {spell.name for spell in hero.spells}
    assert {"Poison Spray", "Guiding Bolt"} <= names
    assert "Bane" not in names
    assert "Cure Wounds" not in names
    notes = hero.traits["unresolved"]
    assert "Poison Spray: on the sheet, not prepared" not in notes
    assert "Guiding Bolt: on the sheet, not prepared" not in notes
    assert "Bane: on the sheet, not prepared" in notes
    assert "Cure Wounds: on the sheet, not prepared" in notes
    assert hero.traits["free_spells"]["guiding-bolt"] == "Guiding Bolt"
    assert hero.uses_max["Guiding Bolt"] == 2
    assert hero.uses_left["Guiding Bolt"] == 2
    abilities = blank_abilities()
    foe = Combatant(
        id="foe",
        name="Foe",
        side="enemy",
        ac=12,
        max_hp=20,
        hp=20,
        speed=30,
        abilities=abilities,
        save_bonuses=save_bonuses_for(abilities, [], 2),
        position=(1, 0),
    )
    encounter = Encounter([hero, foe], width=8, height=8)
    indexes = {action.spell_index for action in legal_actions(encounter, hero, automated=False)}
    assert "poison-spray" in indexes
    assert "guiding-bolt" in indexes


def _portrait_sheet(url):
    sheet = _zealot()
    sheet["decorations"] = {
        "avatarUrl": url,
        "backdropAvatarUrl": "https://www.dndbeyond.com/avatars/1/backdrop.jpeg",
    }
    return sheet


def test_a_public_portrait_is_saved_and_a_backdrop_is_ignored(tmp_path):
    catalog = load_catalog()
    seen = []

    def images(url):
        seen.append(url)
        return "image/png", _PNG

    sheet = _portrait_sheet("https://www.dndbeyond.com/avatars/12/34/face.png?width=150&height=150")
    fetch_public_character(77, fetcher=lambda _url: {"success": True, "data": sheet}, imports_dir=tmp_path, image_fetcher=images)
    assert seen == ["https://www.dndbeyond.com/avatars/12/34/face.png?width=150&height=150"]
    assert (tmp_path / "77.portrait").read_bytes() == _PNG
    hero = public_character(sheet, catalog, (1, 1), "ddb-77", imports_dir=tmp_path)
    assert hero.traits["portrait"] == "77.portrait"
    assert token_image(hero, imports_dir=tmp_path).startswith("data:image/png;base64,")

    bare = _zealot()
    bare["decorations"] = {"avatarUrl": None, "defaultBackdrop": {"backdropAvatarUrl": "https://www.dndbeyond.com/avatars/9/theme.jpeg"}}
    fetch_public_character(77, fetcher=lambda _url: {"success": True, "data": bare}, imports_dir=tmp_path, image_fetcher=images)
    assert not (tmp_path / "77.portrait").exists()
    assert "portrait" not in public_character(bare, catalog, (1, 1), "ddb-77", imports_dir=tmp_path).traits


def test_a_portrait_from_another_host_is_rejected_and_a_failed_download_keeps_the_old_file(tmp_path):
    called = []

    def images(url):
        called.append(url)
        raise OSError("offline")

    (tmp_path / "77.portrait").write_bytes(_PNG)
    rejected = _portrait_sheet("https://example.com/avatars/face.png")
    fetch_public_character(77, fetcher=lambda _url: {"success": True, "data": rejected}, imports_dir=tmp_path, image_fetcher=images)
    assert called == []
    assert not (tmp_path / "77.portrait").exists()

    (tmp_path / "77.portrait").write_bytes(_PNG)
    kept = _portrait_sheet("https://www.dndbeyond.com/avatars/12/face.png")
    fetch_public_character(77, fetcher=lambda _url: {"success": True, "data": kept}, imports_dir=tmp_path, image_fetcher=images)
    assert (tmp_path / "77.portrait").read_bytes() == _PNG
    huge = _PNG + b"\x00" * (PORTRAIT_BYTES + 1)
    assert save_character_portrait(kept, folder=tmp_path, fetcher=lambda _url: ("image/png", huge)) == "77.portrait"
    assert (tmp_path / "77.portrait").read_bytes() == _PNG


def test_token_art_uses_a_creature_file_or_a_class_icon(tmp_path, monkeypatch):
    icons = tmp_path / "tokens"
    icons.mkdir()
    (icons / "wolf.png").write_bytes(_PNG)
    (icons / "class-cleric.png").write_bytes(_PNG)
    monkeypatch.setattr("src.board.TOKENS_DIR", icons)
    wolf = Combatant(
        id="wolf",
        name="Wolf",
        side="enemy",
        ac=13,
        max_hp=11,
        hp=11,
        speed=40,
        abilities=blank_abilities(),
        save_bonuses=save_bonuses_for(blank_abilities(), [], 2),
        position=(1, 1),
        creature_index="wolf",
    )
    cleric = Combatant(
        id="cleric",
        name="Sample Cleric",
        side="party",
        ac=17,
        max_hp=43,
        hp=43,
        speed=30,
        abilities=blank_abilities(),
        save_bonuses=save_bonuses_for(blank_abilities(), [], 3),
        position=(2, 2),
        traits={"class_name": "Cleric / Fighter"},
    )
    weapon = Combatant(
        id="caster",
        name="Cleric",
        side="party",
        ac=17,
        max_hp=43,
        hp=43,
        speed=30,
        abilities=blank_abilities(),
        save_bonuses=save_bonuses_for(blank_abilities(), [], 3),
        position=(0, 0),
        concentrating_on="spiritual-weapon",
        traits={"spiritual_weapon": {"x": 3, "y": 3}, "class_name": "Cleric"},
    )
    encounter = Encounter([wolf, cleric, weapon], width=8, height=8)
    payload = battle_payload(
        encounter,
        mode="place",
        actor=None,
        actions=[],
        abilities=[],
        armed=None,
        slot=None,
        selected_id=None,
    )
    images = {token["id"]: token.get("image") for token in payload["tokens"]}
    assert images["wolf"] == "tokens/wolf.png"
    assert images["cleric"] == "tokens/class-cleric.png"
    assert images["spiritual-weapon-caster"] is None
    assert "DND_TOKEN_ART" not in str(images)
    assert "dndbeyond.com" not in str(images)


def test_every_cached_monster_has_a_token():
    missing = [
        path.stem
        for path in sorted((CACHE / "monsters").glob("*.json"))
        if not (TOKENS_DIR / f"{path.stem}.png").is_file()
    ]
    classes = [
        "barbarian", "bard", "cleric", "druid", "fighter", "monk",
        "paladin", "ranger", "rogue", "sorcerer", "warlock", "wizard",
    ]
    missing.extend(
        f"class-{name}"
        for name in classes
        if not (TOKENS_DIR / f"class-{name}.png").is_file()
    )
    assert missing == []


def _fighter(position=(8, 6)):
    payload = json.loads((IMPORTS / "173924955.json").read_text(encoding="utf-8"))
    return public_character(payload, load_catalog(), position, "fighter")


def _log(encounter) -> str:
    return " ".join(encounter.log)


def test_saved_fighter_has_second_wind_and_action_surge():
    hero = _fighter()
    assert hero.name == "Sample Fighter"
    assert hero.traits["ddb_id"] == 173924955
    assert hero.traits["bonus_heals"] == [{"name": "Second Wind", "die": "1d10 + 5"}]
    assert hero.uses_max["Second Wind"] == 3
    assert hero.uses_left["Second Wind"] == 3
    assert hero.traits["tactical_shift"] is True
    assert hero.uses_max["Action Surge"] == 1
    assert hero.uses_left["Action Surge"] == 1
    notes = " ".join(hero.traits["unresolved"])
    assert "Tactical Mind" in notes
    assert "Action Surge" not in notes
    assert "Second Wind" not in notes
    assert "Tactical Shift" not in notes
    foe = creature("foe", "Foe", "enemy", 12, 30, (4, 6))
    encounter = Encounter([hero, foe], seed=1)
    _prepare(encounter, [hero.id, foe.id], [])
    actions = encounter.legal_actions(hero)
    assert any(action.attack_name == "Second Wind" and action.budget == "bonus" for action in actions)
    assert any(action.attack_name == "Action Surge" and action.budget == "action" for action in actions)
    labels = [action.attack_name for action in actions]
    assert "Tactical Mind" not in labels
    assert "Tactical Shift" not in labels


def test_second_wind_heals_and_the_shift_does_not_provoke_until_it_runs_out():
    hero = _fighter((8, 6))
    gained = base_speed(hero) // 2
    steps = gained // 5
    assert steps >= 1
    arrival = (8 - steps, 6)
    assert arrival[0] >= 1
    watch = creature("watch", "Watch", "enemy", 12, 30, (9, 6), attacks=[sword()])
    watch.traits["sentinel"] = True
    bandit = creature("bandit", "Bandit", "enemy", 12, 30, (arrival[0] - 1, 6), attacks=[sword()])
    encounter = Encounter([hero, watch, bandit], seed=1)
    _prepare(encounter, [hero.id, watch.id, bandit.id], [8, 15, 3])
    hero.hp = 20
    encounter.reaction_policy = "heuristic"
    wind = next(action for action in encounter.legal_actions(hero) if action.attack_name == "Second Wind")
    encounter.apply(wind, "test")
    assert hero.hp == 33
    assert hero.uses_left["Second Wind"] == 2
    assert hero.traits.get("disengaged") is not True
    assert f"can move {gained} feet without provoking opportunity attacks." in _log(encounter)
    first = next(action for action in encounter.legal_actions(hero) if action.destination == arrival)
    encounter.apply(first, "test")
    assert hero.position == arrival
    assert "makes an opportunity attack" not in _log(encounter)
    assert "Sentinel" not in _log(encounter)
    assert watch.reaction_available is True
    back = (arrival[0] + 1, 6)
    second = next(action for action in encounter.legal_actions(hero) if action.destination == back)
    encounter.apply(second, "test")
    assert hero.position == back
    assert "makes an opportunity attack" in _log(encounter)
    assert bandit.reaction_available is False
    assert watch.reaction_available is True


def test_action_surge_opens_another_action_and_blocks_magic_until_the_next_turn():
    hero = _fighter((1, 1))
    hero.spells.append(cache_spell("fire-bolt"))
    foe = creature("foe", "Foe", "enemy", 14, 40, (8, 1))
    encounter = Encounter([hero, foe], seed=1)
    _prepare(encounter, [hero.id, foe.id], [])
    assert any(action.spell_index == "fire-bolt" for action in encounter.legal_actions(hero))
    surge = next(action for action in encounter.legal_actions(hero) if action.attack_name == "Action Surge")
    encounter.apply(surge, "test")
    assert hero.action_available is True
    assert hero.traits.get("banked_action") is True
    assert hero.uses_left["Action Surge"] == 0
    assert "can take one additional action, except the Magic action." in _log(encounter)
    dash = next(action for action in encounter.legal_actions(hero) if action.kind == "dash")
    encounter.apply(dash, "test")
    assert hero.action_available is True
    assert hero.traits.get("surge_blocks_magic") is True
    opened = encounter.legal_actions(hero)
    assert any(action.kind == "attack" and action.budget == "action" for action in opened)
    assert all(action.spell_index != "fire-bolt" for action in opened)
    assert all(action.attack_name != "Action Surge" for action in opened)
    finish_turn(encounter)
    finish_turn(encounter)
    encounter.begin_turn()
    assert encounter.current().id == hero.id
    assert hero.traits.get("surge_blocks_magic") is None
    assert hero.uses_left["Action Surge"] == 0
    assert any(action.spell_index == "fire-bolt" for action in encounter.legal_actions(hero))


def test_the_heuristic_takes_action_surge_before_the_attack():
    hero = _fighter((1, 1))
    foe = creature("foe", "Foe", "enemy", 12, 30, (2, 1), attacks=[sword()])
    encounter = Encounter([hero, foe], seed=1)
    _prepare(encounter, [hero.id, foe.id], [])
    pick, reason = heuristic_advice(encounter, hero)
    assert pick.attack_name == "Action Surge"
    assert reason == "Action Surge grants another action."
    actions = encounter.legal_actions(hero, automated=True)
    assert _needs_model(pick, _decision_menu(encounter, hero, actions, pick)) is False


def _monk(position=(5, 5)):
    payload = json.loads((IMPORTS / "173926371.json").read_text(encoding="utf-8"))
    return public_character(payload, load_catalog(), position, "monk")


def _silence_luck(hero) -> None:
    hero.uses_left["Luck"] = 0


def _unarmed(encounter, hero):
    return next(
        action
        for action in encounter.legal_actions(hero)
        if action.attack_name == "Unarmed Strike" and action.budget == "action" and not action.tag
    )


def test_saved_monk_uses_martial_arts_and_focus():
    hero = _monk()
    assert hero.name == "paulfernandez1331's Character"
    assert hero.abilities["dexterity"] == 18
    assert hero.abilities["wisdom"] == 16
    assert hero.ac == 17
    assert hero.speed == 50
    assert hero.hp == 38
    assert hero.max_hp == 38
    assert hero.traits["proficiency"] == 3
    assert hero.attacks_per_action == 2
    assert hero.uses_max["Focus Points"] == 5
    assert hero.traits["monk"]["save_dc"] == 14
    assert hero.traits["monk"]["focus"] == "Focus Points"
    daggers = [attack for attack in hero.attacks if attack.name == "Dagger"]
    assert len(daggers) == 2
    for attack in daggers:
        assert attack.attack_bonus == 7
        assert attack.ability == "dexterity"
        assert attack.damage == [{"type": "piercing", "expr": "1d8+4"}]
        assert attack.mastery == "nick"
    unarmed = hero.attack_named("Unarmed Strike")
    assert unarmed.attack_bonus == 7
    assert unarmed.ability == "dexterity"
    assert unarmed.damage == [{"type": "bludgeoning", "expr": "1d8+4"}]
    assert (unarmed.rider or {}).get("unarmed") is True
    shove = hero.attack_named("Shove")
    prone = hero.attack_named("Shove Prone")
    assert shove is not None and prone is not None
    assert shove.dc_value == 15
    assert shove.attack_bonus is None
    assert hero.attack_named("Grapple") is None
    notes = hero.traits["unresolved"]
    assert "Slow Fall" in notes
    for name in (
        "Focus Points",
        "Monk's Focus",
        "Monk\u2019s Focus",
        "Ki",
        "Flurry of Blows",
        "Patient Defense",
        "Step of the Wind",
        "Uncanny Metabolism",
        "Deflect Attack",
        "Deflect Attacks",
        "Stunning Strike",
        "Open Hand Technique",
        "Martial Arts",
        "Unarmored Movement",
    ):
        assert name not in notes


def test_declining_stunning_strike_and_the_push_leaves_the_target():
    hero = _monk((5, 5))
    foe = creature("foe", "Foe", "enemy", 16, 40, (6, 5))
    encounter = Encounter([hero, foe], seed=1)
    _prepare(encounter, [hero.id, foe.id], [12, 4])
    _silence_luck(hero)
    encounter.reaction_policy = "ask"
    origin = tuple(foe.position)
    encounter.apply(_unarmed(encounter, hero), "test")
    window = encounter.pending
    assert window["phase"] == "rider"
    assert window["spends_reaction"] is False
    assert "reaction" not in window["prompt"].lower()
    assert window["decline"] == "Don't use Stunning Strike"
    assert window["offers"][0]["label"] == "Stunning Strike"
    encounter.answer_reaction(None)
    assert hero.uses_left["Focus Points"] == 5
    assert "stunning_used" not in hero.traits
    window = encounter.pending
    assert window["phase"] == "rider"
    assert window["decline"] == "Don't push"
    assert window["offers"][0]["label"] == "Push 5 feet"
    assert "reaction" not in window["prompt"].lower()
    encounter.answer_reaction(None)
    assert encounter.pending is None
    assert tuple(foe.position) == origin
    assert "tavern_push_used" not in hero.traits
    text = _log(encounter)
    assert "+ 7" in text
    assert "1d8" in text


def test_the_heuristic_pushes_only_when_no_later_melee_remains():
    hero = _monk((5, 5))
    foe = creature("foe", "Foe", "enemy", 16, 40, (6, 5))
    encounter = Encounter([hero, foe], seed=1)
    _prepare(encounter, [hero.id, foe.id], [12, 3])
    encounter.begin_turn()
    _silence_luck(hero)
    hero.bonus_available = False
    hero.attacks_made = 1
    hero.action_available = True
    hero.traits["stunning_used"] = True
    encounter.reaction_policy = "heuristic"
    encounter.apply(_unarmed(encounter, hero), "test")
    assert encounter.pending is None
    assert foe.position == (7, 5)
    assert hero.traits.get("tavern_push_used") is True
    assert "Tavern Brawler pushes Foe" in _log(encounter)


def test_deflect_attack_uses_the_sheet_die():
    hero = _monk((2, 2))
    foe = creature("foe", "Foe", "enemy", 14, 40, (3, 2), attacks=[sword()])
    encounter = Encounter([hero, foe], seed=1)
    _prepare(encounter, [foe.id, hero.id], [15, 3, 1])
    _silence_luck(hero)
    hero.uses_left["Focus Points"] = 0
    encounter.reaction_policy = "heuristic"
    strike = next(action for action in encounter.legal_actions(foe) if action.attack_name == "Sword")
    encounter.apply(strike, "test")
    assert encounter.pending is None
    assert hero.hp == 38
    assert hero.reaction_available is False
    text = _log(encounter)
    assert "1d10+9" in text
    assert "1d10+5" not in text


def test_a_reduced_hit_can_be_redirected():
    hero = _monk((2, 2))
    foe = creature("foe", "Foe", "enemy", 14, 40, (3, 2), attacks=[sword()])
    encounter = Encounter([hero, foe], seed=1)
    _prepare(encounter, [foe.id, hero.id], [15, 1, 1, 4, 4, 10])
    _silence_luck(hero)
    encounter.reaction_policy = "heuristic"
    strike = next(action for action in encounter.legal_actions(foe) if action.attack_name == "Sword")
    encounter.apply(strike, "test")
    assert encounter.pending is None
    assert hero.hp == 38
    assert foe.hp == 28
    assert hero.uses_left["Focus Points"] == 4
    assert hero.reaction_available is False
    assert "1d10+9" in _log(encounter)


def test_stunning_strike_lasts_until_the_monks_next_turn():
    hero = _monk((5, 5))
    foe = creature("foe", "Foe", "enemy", 16, 40, (6, 5))
    encounter = Encounter([hero, foe], seed=1)
    _prepare(encounter, [hero.id, foe.id], [12, 4, 5, 12, 4, 18, 15, 8, 3])
    _silence_luck(hero)
    encounter.reaction_policy = "heuristic"
    origin = tuple(foe.position)
    encounter.apply(_unarmed(encounter, hero), "test")
    assert hero.uses_left["Focus Points"] == 4
    assert "stunned" in foe.conditions
    assert tuple(foe.position) == origin
    finish_turn(encounter)
    finish_turn(encounter)
    assert "stunned" in foe.conditions
    encounter.begin_turn()
    assert "stunned" not in foe.conditions
    encounter.apply(_unarmed(encounter, hero), "test")
    assert hero.uses_left["Focus Points"] == 3
    assert "stunned" not in foe.conditions
    assert base_speed(foe) == 15
    assert any(mark.get("next_attack") for mark in marks_of(foe))
    assert tuple(foe.position) == origin
    encounter.apply(_unarmed(encounter, hero), "test")
    assert hero.uses_left["Focus Points"] == 3
    assert base_speed(foe) == 15
    assert not any(mark.get("next_attack") for mark in marks_of(foe))
    assert re.search(r"\badvantage\b", _log(encounter))
    assert "Stunning Strike" in _log(encounter)


def test_flurry_of_blows_offers_open_hand_topple():
    hero = _monk((5, 5))
    foe = creature("foe", "Foe", "enemy", 16, 40, (6, 5))
    encounter = Encounter([hero, foe], seed=1)
    _prepare(encounter, [hero.id, foe.id], [15, 4, 19, 15, 4, 2])
    encounter.begin_turn()
    _silence_luck(hero)
    hero.action_available = False
    hero.traits["monk_attacked"] = True
    hero.traits["stunning_used"] = True
    encounter.reaction_policy = "heuristic"
    pick, reason = heuristic_advice(encounter, hero)
    assert pick.attack_name == "Flurry of Blows"
    assert reason == "Flurry of Blows spends 1 Focus for two unarmed strikes."
    actions = encounter.legal_actions(hero, automated=True)
    assert _needs_model(pick, _decision_menu(encounter, hero, actions, pick)) is False
    bonus_disengage = [action for action in actions if action.kind == "disengage" and action.budget == "bonus"]
    bonus_dash = [action for action in actions if action.kind == "dash" and action.budget == "bonus"]
    assert bonus_disengage[0].attack_name == "Patient Defense"
    assert bonus_dash[0].attack_name == "Step of the Wind"
    encounter.apply(pick, "test")
    assert hero.uses_left["Focus Points"] == 4
    assert hero.bonus_available is False
    assert hero.traits.get("flurry_left") == 2
    first = next(action for action in encounter.legal_actions(hero) if action.tag == "flurry")
    encounter.apply(first, "test")
    assert "prone" not in foe.conditions
    second = next(action for action in encounter.legal_actions(hero) if action.tag == "flurry")
    encounter.apply(second, "test")
    assert "prone" in foe.conditions
    assert hero.uses_left["Focus Points"] == 4
    assert "Open Hand" in _log(encounter)


def test_patient_defense_and_step_of_the_wind_spend_the_bonus():
    hero = _monk((4, 4))
    foe = creature("foe", "Foe", "enemy", 16, 40, (5, 4))
    encounter = Encounter([hero, foe], seed=1)
    _prepare(encounter, [hero.id, foe.id], [])
    defense = next(
        action
        for action in encounter.legal_actions(hero)
        if action.attack_name == "Patient Defense" and action.budget == "bonus"
    )
    encounter.apply(defense, "test")
    assert hero.action_available is True
    assert hero.bonus_available is False
    assert hero.uses_left["Focus Points"] == 5
    assert hero.traits.get("disengaged") is True

    hero = _monk((4, 4))
    foe = creature("foe", "Foe", "enemy", 16, 40, (5, 4))
    encounter = Encounter([hero, foe], seed=1)
    _prepare(encounter, [hero.id, foe.id], [])
    step = next(action for action in encounter.legal_actions(hero) if action.attack_name == "Step of the Wind")
    encounter.apply(step, "test")
    assert hero.action_available is True
    assert hero.bonus_available is False
    assert hero.speed == 50
    assert hero.movement_left == 100
    assert hero.uses_left["Focus Points"] == 5
    assert "jump distance is doubled" not in _log(encounter)

    hero = _monk((4, 4))
    foe = creature("foe", "Foe", "enemy", 16, 40, (5, 4))
    encounter = Encounter([hero, foe], seed=1)
    _prepare(encounter, [hero.id, foe.id], [])
    focused = next(action for action in encounter.legal_actions(hero) if action.attack_name == "Focused Step of the Wind")
    encounter.apply(focused, "test")
    assert hero.action_available is True
    assert hero.speed == 50
    assert hero.uses_left["Focus Points"] == 4
    assert hero.traits.get("disengaged") is True
    assert "jump distance is doubled" in _log(encounter)

    hero = _monk((4, 4))
    foe = creature("foe", "Foe", "enemy", 16, 40, (5, 4))
    encounter = Encounter([hero, foe], seed=1)
    _prepare(encounter, [hero.id, foe.id], [])
    focused = next(action for action in encounter.legal_actions(hero) if action.attack_name == "Focused Patient Defense")
    encounter.apply(focused, "test")
    assert hero.action_available is True
    assert hero.bonus_available is False
    assert hero.dodging is True
    assert hero.traits.get("disengaged") is True
    assert hero.uses_left["Focus Points"] == 4


def test_uncanny_metabolism_heals_and_then_closes():
    hero = _monk((4, 4))
    foe = creature("foe", "Foe", "enemy", 16, 40, (5, 4))
    encounter = Encounter([hero, foe], seed=1)
    _prepare(encounter, [hero.id, foe.id], [])
    _silence_luck(hero)
    pick, _reason = heuristic_advice(encounter, hero)
    assert pick.attack_name != "Uncanny Metabolism"

    hero.hp = 20
    hero.uses_left["Focus Points"] = 1
    pick, reason = heuristic_advice(encounter, hero)
    assert pick.attack_name == "Uncanny Metabolism"
    assert reason == "Uncanny Metabolism restores Focus and hit points."
    actions = encounter.legal_actions(hero, automated=True)
    assert _needs_model(pick, _decision_menu(encounter, hero, actions, pick)) is False
    encounter.rng.rolls = [4]
    encounter.apply(pick, "test")
    assert hero.hp == 29
    assert hero.uses_left["Focus Points"] == 5
    assert hero.uses_left["Uncanny Metabolism"] == 0

    hero = _monk((4, 4))
    foe = creature("foe", "Foe", "enemy", 16, 40, (5, 4))
    encounter = Encounter([hero, foe], seed=1)
    _prepare(encounter, [hero.id, foe.id], [])
    hero.hp = 20
    assert any(action.attack_name == "Uncanny Metabolism" for action in encounter.legal_actions(hero))
    finish_turn(encounter)
    finish_turn(encounter)
    assert encounter.round == 2
    assert hero.uses_left["Uncanny Metabolism"] == 1
    assert all(action.attack_name != "Uncanny Metabolism" for action in encounter.legal_actions(hero))


def test_two_daggers_of_the_same_name_still_nick():
    hero = _monk((4, 4))
    foe = creature("foe", "Foe", "enemy", 16, 40, (5, 4))
    encounter = Encounter([hero, foe], seed=1)
    _prepare(encounter, [hero.id, foe.id], [])
    hero.traits["light_used"] = "Dagger"
    hero.attacks_made = 2
    assert nick_weapon_name(hero) == "Dagger"
    assert attack_window(hero) == "nick"
    swings = [
        action
        for action in encounter.legal_actions(hero)
        if action.kind == "attack" and action.attack_name == "Dagger"
    ]
    assert swings
    assert swings[0].budget == "action"


def test_a_shove_is_one_attack_and_moves_the_target():
    hero = _monk((5, 5))
    foe = creature("foe", "Foe", "enemy", 16, 40, (6, 5))
    encounter = Encounter([hero, foe], seed=1)
    _prepare(encounter, [hero.id, foe.id], [1])
    _silence_luck(hero)
    encounter.reaction_policy = "heuristic"
    shove = next(action for action in encounter.legal_actions(hero) if action.attack_name == "Shove")
    encounter.apply(shove, "test")
    assert encounter.pending is None
    assert foe.position == (7, 5)
    assert hero.attacks_made == 1
    assert hero.traits.get("monk_attacked") is True


def test_a_forced_push_does_not_ask():
    ram = AttackOption(
        name="Ram",
        resolution="attack",
        attack_bonus=6,
        reach_feet=5,
        damage=[{"type": "force", "expr": "1d4"}],
        rider={"push_feet": 5},
    )
    hero = creature("hero", "Hero", "party", 15, 20, (4, 4), attacks=[ram])
    foe = creature("foe", "Foe", "enemy", 10, 20, (5, 4))
    encounter = Encounter([hero, foe], seed=1)
    _prepare(encounter, [hero.id, foe.id], [15, 2])
    encounter.reaction_policy = "ask"
    strike = next(action for action in encounter.legal_actions(hero) if action.attack_name == "Ram")
    encounter.apply(strike, "test")
    assert encounter.pending is None
    assert foe.position == (6, 4)
    assert "Ram pushes Foe" in _log(encounter)


def test_the_choice_prompt_declines_stunning_strike_and_the_push():
    app = AppTest.from_file(ROOT / "app.py", default_timeout=90)
    app.run()
    hero = _monk((5, 5))
    foe = creature("foe", "Foe", "enemy", 16, 40, (6, 5))
    encounter = Encounter([hero, foe], seed=1)
    _prepare(encounter, [hero.id, foe.id], [12, 4])
    _silence_luck(hero)
    encounter.reaction_policy = "ask"
    origin = tuple(foe.position)
    encounter.apply(_unarmed(encounter, hero), "test")
    app.session_state["encounter"] = encounter
    app.session_state["page"] = "Combat"
    app.run()
    assert not app.exception
    assert any(block.value == "Choice" for block in app.subheader)
    captions = [block.value for block in app.caption]
    assert f"{hero.name} can choose." in captions
    labels = [button.label for button in app.button]
    assert "Don't use a reaction" not in labels
    assert "Don't use Stunning Strike" in labels
    visible = "\n".join(str(block.value) for block in app.markdown)
    assert "A choice is waiting." in visible
    assert "A reaction is waiting." not in visible
    assert "reaction" not in encounter.pending["prompt"].lower()
    next(button for button in app.button if button.label == "Don't use Stunning Strike").click().run()
    assert not app.exception
    fight = app.session_state["encounter"]
    assert fight.pending["decline"] == "Don't push"
    assert "Don't use a reaction" not in [button.label for button in app.button]
    next(button for button in app.button if button.label == "Don't push").click().run()
    assert not app.exception
    fight = app.session_state["encounter"]
    assert fight.pending is None
    assert tuple(fight.by_id("foe").position) == origin
    text = " ".join(fight.log)
    assert "+ 7" in text
    assert "1d8" in text
