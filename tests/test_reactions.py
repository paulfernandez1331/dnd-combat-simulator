"""Shield, Counterspell, and Cutting Words. No D&D Beyond requests."""

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.actions import GameAction
from src.ai import play_turn
from src.bestiary import ABILITIES, EXCLUDED_CAST, AttackOption, SpellRecord, classify_monster, classify_spell
from src.combatant import Combatant, from_monster
from src.ddb import public_character
from src.encounter import Encounter, IllegalAction
from src.features import begin_turn_state
from src.paths import CACHE, FIXTURES, ROOT
from src.reactions import can_cut, can_shield, heuristic_choice as reaction_choice
from src.simulate import run_batch
from test_combat import _prepare, creature, finish_turn, load_spell, sword


def _spell(index: str, name: str, level: int) -> SpellRecord:
    return SpellRecord(
        index=index,
        name=name,
        level=level,
        kind="reaction",
        casting_time="Reaction",
        duration="1 round" if index == "shield" else "Instantaneous",
        concentration=False,
        range_mode="self" if index == "shield" else "ranged",
        range_feet=0 if index == "shield" else 60,
        area_type=None,
        area_size=None,
        attack_type=None,
        dc_ability=None,
        dc_success=None,
        damage=[],
        healing={},
        ongoing=False,
        exclude_reason=None,
        classes=[],
        activation="reaction",
        has_components=True,
    )


def shield():
    return _spell("shield", "Shield", 1)


def counterspell():
    return _spell("counterspell", "Counterspell", 3)


def _attack(encounter: Encounter, attacker_id: str, target_id: str):
    actor = encounter.by_id(attacker_id)
    return next(
        item
        for item in encounter.legal_actions(actor)
        if item.attack_name == "Sword" and item.target_id == target_id
    )


def _cast(encounter: Encounter, actor_id: str, index: str, target_id: str):
    actor = encounter.by_id(actor_id)
    return next(
        item
        for item in encounter.legal_actions(actor)
        if item.spell_index == index and item.target_id == target_id
    )


def _logged(encounter: Encounter) -> str:
    return "\n".join(encounter.log)


def test_shield_and_counterspell_are_reactions_and_alarm_stays_out():
    shield_spell = classify_spell(json.loads((CACHE / "spells" / "shield.json").read_text(encoding="utf-8")))
    counter = classify_spell(json.loads((CACHE / "spells" / "counterspell.json").read_text(encoding="utf-8")))
    alarm = classify_spell(json.loads((FIXTURES / "alarm.json").read_text(encoding="utf-8")))
    assert shield_spell.kind == "reaction"
    assert shield_spell.activation == "reaction"
    assert shield_spell.has_components is True
    assert shield_spell.exclude_reason is None
    assert counter.kind == "reaction"
    assert counter.activation == "reaction"
    assert counter.has_components is True
    assert alarm.kind == "excluded"
    assert alarm.exclude_reason == EXCLUDED_CAST


def test_cutting_words_comes_from_the_sheet_reaction():
    class Catalog:
        spells: dict = {}

    hero = public_character(
        {
            "id": 5,
            "name": "Lore",
            "baseHitPoints": 28,
            "stats": [
                {"id": 1, "value": 10},
                {"id": 2, "value": 13},
                {"id": 3, "value": 14},
                {"id": 4, "value": 10},
                {"id": 5, "value": 12},
                {"id": 6, "value": 14},
            ],
            "classes": [
                {
                    "id": 9,
                    "level": 5,
                    "definition": {"name": "Bard"},
                    "classFeatures": [
                        {"definition": {"id": 10292532, "name": "Cutting Words", "requiredLevel": 3}},
                        {"definition": {"id": 55, "name": "Bardic Inspiration", "requiredLevel": 1}},
                    ],
                }
            ],
            "actions": {
                "class": [
                    {
                        "name": "Cutting Words",
                        "componentId": 10292532,
                        "activation": {"activationType": 4},
                        "dice": {"diceString": "1d8"},
                        "range": {"range": 60},
                    },
                    {
                        "name": "Bardic Inspiration",
                        "componentId": 55,
                        "activation": {"activationType": 3},
                        "dice": {"diceString": "1d8"},
                        "limitedUse": {"maxUses": 0, "numberUsed": 0, "statModifierUsesId": 6},
                    },
                    {
                        "name": "Bardic Inspiration",
                        "componentId": 55,
                        "activation": {"activationType": 3},
                        "dice": {"diceString": "1d8"},
                        "limitedUse": {"maxUses": 0, "numberUsed": 0, "statModifierUsesId": 6},
                    },
                    {
                        "name": "Cutting Words",
                        "componentId": 999,
                        "activation": {"activationType": 4},
                        "dice": {"diceString": "1d6"},
                    },
                ]
            },
        },
        Catalog(),
        (0, 0),
        "ddb-5",
    )
    assert hero.traits["cutting_words"] == {"die": "1d8", "range": 60, "pool": "Bardic Inspiration"}
    assert hero.uses_max["Bardic Inspiration"] == 2
    assert hero.uses_left["Bardic Inspiration"] == 2
    notes = " ".join(hero.traits["unresolved"])
    assert "Cutting Words" not in notes
    assert "Bardic Inspiration" in notes


def _pair(policy: str, rolls: list[int], *, ac: int = 15, slots: dict | None = None, spells=None, traits=None, uses=1, hp: int = 30, uses_max=None):
    goblin = creature("gob", "Goblin", "enemy", 12, 20, (0, 0), attacks=[sword()])
    pools = {"Bardic Inspiration": uses} if traits else {}
    pools.update(uses_max or {})
    wizard = creature(
        "wiz",
        "Wizard",
        "party",
        ac,
        hp,
        (1, 0),
        spells=spells if spells is not None else [shield()],
        slots=slots if slots is not None else {1: 1, 2: 1},
        traits=traits or {},
        uses_max=pools,
        uses_left=dict(pools),
        spell_save_dc=15,
    )
    encounter = Encounter([goblin, wizard], seed=1)
    _prepare(encounter, ["gob", "wiz"], rolls)
    encounter.reaction_policy = policy
    encounter.begin_turn()
    return encounter, goblin, wizard


def test_shield_flips_a_close_hit_and_lasts_until_the_casters_turn():
    encounter, goblin, wizard = _pair("heuristic", [12])
    before = wizard.hp
    encounter.apply(_attack(encounter, "gob", "wiz"), "test")
    assert wizard.hp == before
    assert wizard.slots == {1: 0, 2: 1}
    assert wizard.traits["shield_ac"] == 5
    assert wizard.reaction_available is False
    assert "casts Shield" in _logged(encounter)
    assert "AC is 20" in _logged(encounter)
    assert all(item.spell_index != "shield" for item in encounter.legal_actions(wizard))
    finish_turn(encounter)
    assert wizard.traits.get("shield_ac") == 5
    encounter.begin_turn()
    assert "shield_ac" not in wizard.traits
    assert goblin.hp == 20


def test_a_free_shield_is_spent_before_a_spell_slot():
    traits = {"free_spells": {"shield": "Shield"}}
    kept, _, wizard = _pair("heuristic", [12], slots={1: 1}, traits=traits, uses_max={"Shield": 1})
    before = wizard.hp
    kept.apply(_attack(kept, "gob", "wiz"), "test")
    assert wizard.hp == before
    assert wizard.slots == {1: 1}
    assert wizard.uses_left["Shield"] == 0
    assert "free casting" in _logged(kept)

    empty, _, caster = _pair("ask", [12], slots={}, traits=traits, uses_max={"Shield": 1})
    empty.apply(_attack(empty, "gob", "wiz"), "test")
    assert empty.pending["phase"] == "hit"
    assert any(offer["id"] == "shield" for offer in empty.pending["offers"])
    empty.answer_reaction("shield")
    assert caster.traits["shield_ac"] == 5
    assert caster.uses_left["Shield"] == 0
    assert caster.slots == {}


def test_shield_does_not_stop_a_natural_20_and_the_heuristic_skips_it():
    asked, _, wizard = _pair("ask", [20, 3, 3])
    asked.apply(_attack(asked, "gob", "wiz"), "test")
    assert asked.pending["phase"] == "hit"
    assert any(offer["id"] == "shield" for offer in asked.pending["offers"])
    before = wizard.hp
    asked.answer_reaction("shield")
    assert wizard.hp < before
    assert wizard.traits["shield_ac"] == 5
    assert "Critical" in _logged(asked)

    encounter, _, guarded = _pair("heuristic", [20, 4, 4])
    encounter.apply(_attack(encounter, "gob", "wiz"), "test")
    assert guarded.slots[1] == 1
    assert guarded.hp < guarded.max_hp
    assert "does not use a reaction" in _logged(encounter)
    assert encounter.pending is None


def test_heuristic_shield_only_when_five_points_flip_the_hit():
    close, _, wizard = _pair("heuristic", [12])
    close.apply(_attack(close, "gob", "wiz"), "test")
    assert wizard.slots[1] == 0
    assert wizard.hp == wizard.max_hp

    wide, _, other = _pair("heuristic", [18, 4])
    wide.apply(_attack(wide, "gob", "wiz"), "test")
    assert other.slots[1] == 1
    assert other.hp == other.max_hp - 6
    assert "does not use a reaction" in _logged(wide)


def test_a_declined_reaction_keeps_the_original_hit():
    encounter, _, wizard = _pair("ask", [12, 4])
    encounter.apply(_attack(encounter, "gob", "wiz"), "test")
    assert encounter.pending is not None
    assert wizard.hp == wizard.max_hp
    assert wizard.slots[1] == 1
    with pytest.raises(IllegalAction, match="reaction"):
        encounter.apply(GameAction(id=0, kind="end", label="End turn"), "test")
    encounter.answer_reaction(None)
    assert encounter.pending is None
    assert wizard.hp == wizard.max_hp - 6
    assert wizard.slots[1] == 1
    assert "does not use a reaction" in _logged(encounter)


def test_magic_missile_deals_nothing_when_shield_is_used():
    missile = load_spell("magic-missile")
    caster = creature("wiz", "Wizard", "enemy", 12, 20, (0, 0), spells=[missile], slots={1: 1})
    guard = creature("guard", "Guard", "party", 15, 30, (1, 0), spells=[shield()], slots={1: 1})
    encounter = Encounter([caster, guard], seed=1)
    _prepare(encounter, ["wiz", "guard"], [])
    encounter.reaction_policy = "heuristic"
    encounter.begin_turn()
    encounter.apply(_cast(encounter, "wiz", "magic-missile", "guard"), "test")
    assert guard.hp == guard.max_hp
    assert caster.slots[1] == 0
    assert guard.slots[1] == 0
    assert "no damage" in _logged(encounter).lower()
    assert encounter.pending is None

    asked = Encounter(
        [
            creature("wiz", "Wizard", "enemy", 12, 20, (0, 0), spells=[missile], slots={1: 1}),
            creature("guard", "Guard", "party", 15, 30, (1, 0), spells=[shield()], slots={1: 1}),
        ],
        seed=1,
    )
    _prepare(asked, ["wiz", "guard"], [1, 1, 1])
    asked.begin_turn()
    asked.apply(_cast(asked, "wiz", "magic-missile", "guard"), "test")
    assert asked.pending["phase"] == "missile"
    target = asked.by_id("guard")
    asked.answer_reaction(None)
    assert target.hp == target.max_hp - 6
    assert target.slots[1] == 1


def test_counterspell_refunds_the_slot_only_when_the_save_fails():
    missile = load_spell("magic-missile")
    caster = creature("wiz", "Wizard", "party", 12, 20, (0, 0), spells=[missile], slots={1: 1})
    mage = creature(
        "mage",
        "Mage",
        "enemy",
        12,
        30,
        (1, 0),
        spells=[counterspell()],
        slots={3: 1},
        spell_save_dc=15,
    )
    failed = Encounter([caster, mage], seed=1)
    _prepare(failed, ["wiz", "mage"], [1])
    failed.reaction_policy = "heuristic"
    failed.begin_turn()
    failed.apply(_cast(failed, "wiz", "magic-missile", "mage"), "test")
    assert mage.hp == mage.max_hp
    assert caster.slots[1] == 1
    assert mage.slots[3] == 0
    assert caster.action_available is False
    assert "fizzles" in _logged(failed)
    assert "not spent" in _logged(failed)

    caster = creature("wiz", "Wizard", "party", 12, 20, (0, 0), spells=[missile], slots={1: 1})
    mage = creature(
        "mage",
        "Mage",
        "enemy",
        12,
        30,
        (1, 0),
        spells=[counterspell()],
        slots={3: 1},
        spell_save_dc=15,
    )
    saved = Encounter([caster, mage], seed=1)
    _prepare(saved, ["wiz", "mage"], [15, 1, 1, 1])
    saved.reaction_policy = "heuristic"
    saved.begin_turn()
    saved.apply(_cast(saved, "wiz", "magic-missile", "mage"), "test")
    assert mage.hp == mage.max_hp - 6
    assert caster.slots[1] == 0
    assert mage.slots[3] == 0
    assert "is cast" in _logged(saved)


def test_a_second_counterspell_tries_after_a_successful_save():
    missile = load_spell("magic-missile")
    caster = creature("wiz", "Wizard", "party", 12, 20, (0, 0), spells=[missile], slots={1: 1})
    first = creature("m1", "Mage", "enemy", 12, 20, (1, 0), spells=[counterspell()], slots={3: 1}, spell_save_dc=13)
    second = creature("m2", "Adept", "enemy", 12, 20, (1, 1), spells=[counterspell()], slots={3: 1}, spell_save_dc=15)
    encounter = Encounter([caster, first, second], seed=1)
    _prepare(encounter, ["wiz", "m1", "m2"], [15, 1])
    encounter.reaction_policy = "heuristic"
    encounter.begin_turn()
    encounter.apply(_cast(encounter, "wiz", "magic-missile", "m1"), "test")
    assert caster.slots[1] == 1
    assert first.slots[3] == 0
    assert second.slots[3] == 0
    assert first.hp == first.max_hp
    assert second.hp == second.max_hp


def test_counterspell_skips_a_cantrip_and_a_save_that_cannot_fail():
    bolt = load_spell("fire-bolt")
    caster = creature("wiz", "Wizard", "party", 12, 20, (0, 0), spells=[bolt], spell_attack_bonus=6, caster_level=1)
    mage = creature("mage", "Mage", "enemy", 12, 20, (1, 0), spells=[counterspell()], slots={3: 1}, spell_save_dc=18)
    encounter = Encounter([caster, mage], seed=1)
    _prepare(encounter, ["wiz", "mage"], [1])
    encounter.reaction_policy = "heuristic"
    encounter.begin_turn()
    encounter.apply(_cast(encounter, "wiz", "fire-bolt", "mage"), "test")
    assert mage.slots[3] == 1
    assert "does not use a reaction" in _logged(encounter)

    missile = load_spell("magic-missile")
    caster = creature("wiz", "Wizard", "party", 12, 20, (0, 0), spells=[missile], slots={1: 1})
    caster.save_bonuses["constitution"] = 14
    mage = creature("mage", "Mage", "enemy", 12, 20, (1, 0), spells=[counterspell()], slots={3: 1}, spell_save_dc=15)
    fight = Encounter([caster, mage], seed=1)
    choice, _reason = reaction_choice(
        fight,
        mage,
        {"actor_id": "wiz", "spell_index": "magic-missile", "target_ids": ["mage"]},
        "counterspell",
        [{"id": "counterspell", "label": "Counterspell"}],
    )
    assert choice is None


def test_cutting_words_can_turn_the_hit_into_a_miss():
    traits = {"cutting_words": {"die": "1d8", "range": 60, "pool": "Bardic Inspiration"}}
    encounter, _, bard = _pair("heuristic", [12, 2], spells=[], traits=traits, uses=1, slots={})
    encounter.apply(_attack(encounter, "gob", "wiz"), "test")
    assert bard.hp == bard.max_hp
    assert bard.uses_left["Bardic Inspiration"] == 0
    assert bard.reaction_available is False
    assert "Cutting Words" in _logged(encounter)
    assert "total is 14" in _logged(encounter)


def test_cutting_words_is_not_offered_without_inspiration():
    traits = {"cutting_words": {"die": "1d8", "range": 60, "pool": "Bardic Inspiration"}}
    encounter, _, bard = _pair("ask", [12, 4], spells=[], traits=traits, uses=0, slots={})
    encounter.apply(_attack(encounter, "gob", "wiz"), "test")
    assert encounter.pending is None
    assert bard.hp == bard.max_hp - 6
    assert "Cutting Words" not in _logged(encounter)


def test_heuristic_cutting_words_only_when_the_margin_fits_the_die():
    traits = {"cutting_words": {"die": "1d8", "range": 60, "pool": "Bardic Inspiration"}}
    close, _, bard = _pair("heuristic", [12, 2], spells=[], traits=traits, uses=1, slots={})
    close.apply(_attack(close, "gob", "wiz"), "test")
    assert bard.uses_left["Bardic Inspiration"] == 0
    assert bard.hp == bard.max_hp

    wide_traits = {"cutting_words": {"die": "1d6", "range": 60, "pool": "Bardic Inspiration"}}
    wide, _, other = _pair("heuristic", [18, 4], spells=[], traits=wide_traits, uses=1, slots={})
    wide.apply(_attack(wide, "gob", "wiz"), "test")
    assert other.uses_left["Bardic Inspiration"] == 1
    assert other.hp == other.max_hp - 6
    assert "Cutting Words" not in _logged(wide)


def test_shield_is_preferred_when_it_guarantees_the_miss():
    traits = {"cutting_words": {"die": "1d8", "range": 60, "pool": "Bardic Inspiration"}}
    encounter, _, wizard = _pair("heuristic", [12], traits=traits, uses=1)
    encounter.apply(_attack(encounter, "gob", "wiz"), "test")
    assert wizard.slots[1] == 0
    assert wizard.uses_left["Bardic Inspiration"] == 1
    assert "casts Shield" in _logged(encounter)
    assert "Cutting Words" not in _logged(encounter)


def test_cutting_words_on_damage_only_when_it_can_keep_an_ally_up():
    traits = {"cutting_words": {"die": "1d8", "range": 60, "pool": "Bardic Inspiration"}}
    saved, _, bard = _pair("heuristic", [18, 6, 5], ac=10, hp=6, spells=[], traits=traits, uses=1, slots={})
    saved.apply(_attack(saved, "gob", "wiz"), "test")
    assert bard.hp == 3
    assert bard.uses_left["Bardic Inspiration"] == 0
    assert "Damage is now 3" in _logged(saved)

    heavy = AttackOption(
        name="Sword",
        resolution="attack",
        attack_bonus=4,
        reach_feet=5,
        damage=[{"type": "slashing", "expr": "1d4+16"}],
    )
    goblin = creature("gob", "Goblin", "enemy", 12, 20, (0, 0), attacks=[heavy])
    other = creature(
        "wiz",
        "Wizard",
        "party",
        10,
        4,
        (1, 0),
        traits=traits,
        uses_max={"Bardic Inspiration": 1},
        uses_left={"Bardic Inspiration": 1},
    )
    fight = Encounter([goblin, other], seed=1)
    _prepare(fight, ["gob", "wiz"], [18, 4])
    fight.reaction_policy = "heuristic"
    fight.begin_turn()
    fight.apply(_attack(fight, "gob", "wiz"), "test")
    assert other.hp == 0
    assert other.uses_left["Bardic Inspiration"] == 1


def test_cutting_words_ignores_the_leveled_spell_limit_and_unconsciousness():
    bard = creature(
        "bard",
        "Bard",
        "party",
        15,
        20,
        (1, 0),
        spells=[shield()],
        slots={1: 1},
        traits={"cutting_words": {"die": "1d8", "range": 60, "pool": "Bardic Inspiration"}, "leveled_cast": True},
        uses_max={"Bardic Inspiration": 1},
        uses_left={"Bardic Inspiration": 1},
    )
    goblin = creature("gob", "Goblin", "enemy", 12, 20, (0, 0), attacks=[sword()])
    encounter = Encounter([bard, goblin], seed=1)
    encounter.order = ["bard", "gob"]
    encounter.turn_index = 0
    assert can_cut(encounter, bard, goblin, critical=False)
    assert not can_shield(encounter, bard, bard)
    far = creature("far", "Far", "party", 15, 20, (13, 0), traits=bard.traits, uses_left={"Bardic Inspiration": 1})
    assert not can_cut(encounter, far, goblin, critical=False)
    bard.conditions["unconscious"] = 99
    assert not can_cut(encounter, bard, goblin, critical=False)
    bard.traits["shield_ac"] = 5
    begin_turn_state(encounter, bard)
    assert "shield_ac" not in bard.traits
    assert bard.reaction_available is False


def test_an_automated_turn_pauses_for_the_other_side():
    encounter, _, wizard = _pair("ask", [12, 4])
    play_turn(encounter, "heuristic")
    assert encounter.pending is not None
    assert encounter.auto_side == "enemy"
    assert encounter.auto_policy == "heuristic"
    assert wizard.hp == wizard.max_hp
    encounter.answer_reaction(None)
    assert wizard.hp == wizard.max_hp - 6
    assert encounter.current().id == "gob"
    play_turn(encounter, encounter.auto_policy)
    assert encounter.pending is None
    assert encounter.current().id == "wiz"
    assert encounter.auto_side is None


def test_a_batch_fight_finishes_with_no_reaction_prompt(tmp_path, monkeypatch):
    class Steady:
        def __init__(self, seed=None):
            self.seed = seed

        def randint(self, low, high):
            if high >= 20:
                return 12
            return min(high, max(low, 3))

    monkeypatch.setattr("src.encounter.random.Random", Steady)
    held = {}
    original = run_batch.__globals__["Encounter"].from_setup.__func__

    def capture(data):
        encounter = original(Encounter, data)
        held["encounter"] = encounter
        return encounter

    monkeypatch.setattr("src.simulate.Encounter.from_setup", capture)
    wizard = creature(
        "wiz",
        "Wizard",
        "party",
        15,
        40,
        (0, 0),
        attacks=[sword()],
        spells=[shield()],
        slots={1: 1},
    )
    goblin = creature("gob", "Goblin", "enemy", 12, 40, (1, 0), attacks=[sword()], challenge_rating=0.25)
    setup = Encounter([wizard, goblin], seed=1, round_cap=2, label="reactions").to_setup()
    result = run_batch(setup, count=1, policy="heuristic", seed=1, run_dir=Path(tmp_path))
    assert result["completed"] == 1
    assert held["encounter"].reaction_policy == "heuristic"
    assert held["encounter"].pending is None
    assert held["encounter"].over is True
    assert "casts Shield" in "\n".join(held["encounter"].log)


def test_the_reaction_panel_declines_and_the_hit_lands():
    app = AppTest.from_file(ROOT / "app.py", default_timeout=90)
    app.run()
    encounter, _, wizard = _pair("ask", [12, 4])
    encounter.apply(_attack(encounter, "gob", "wiz"), "test")
    assert encounter.pending is not None
    app.session_state["encounter"] = encounter
    app.session_state["page"] = "Combat"
    app.run()
    assert not app.exception
    labels = [button.label for button in app.button]
    assert "Don't use a reaction" in labels
    assert "End turn" not in labels
    visible = "\n".join(str(block.value) for block in app.markdown)
    assert "hits Wizard" in visible
    next(button for button in app.button if button.label == "Don't use a reaction").click().run()
    assert not app.exception
    fight = app.session_state["encounter"]
    assert fight.pending is None
    assert fight.by_id("wiz").hp == wizard.max_hp - 6
    assert "does not use a reaction" in "\n".join(fight.log)


def _block(**extra) -> dict:
    raw = {
        "index": "sample",
        "name": "Sample",
        "armor_class": 15,
        "hit_points": 30,
        "speed": {"walk": "30 ft."},
        "size": "Medium",
        "type": "humanoid",
        "challenge_rating": 1,
        "actions": [],
        "reactions": [],
    }
    raw.update({ability: 10 for ability in ABILITIES})
    raw.update(extra)
    return raw


def _melee_weapon(name: str = "Scimitar") -> dict:
    return {
        "name": name,
        "desc": "Melee Attack Roll: +4, reach 5 ft. Hit: 5 (1d6 + 2) Slashing damage.",
        "attack_bonus": 4,
        "damage": [{"damage_dice": "1d6+2", "damage_type": {"index": "slashing"}}],
    }


def _loaded(raw: dict, creature_id: str, name: str, ac: int | None = None):
    record = classify_monster(raw)
    creature = from_monster(record, creature_id=creature_id, name=name, side="party", position=(1, 0))
    if ac is not None:
        creature.ac = ac
    return record, creature


def _strike(defender, rolls, policy="heuristic", attack=None):
    goblin = creature("gob", "Goblin", "enemy", 12, 40, (0, 0), attacks=[attack or sword()])
    encounter = Encounter([goblin, defender], seed=1)
    _prepare(encounter, ["gob", defender.id], rolls)
    encounter.reaction_policy = policy
    encounter.begin_turn()
    action = next(
        item
        for item in encounter.legal_actions(goblin)
        if item.attack_name == (attack.name if attack is not None else "Sword") and item.target_id == defender.id
    )
    encounter.apply(action, "test")
    return encounter


def test_a_printed_ac_bonus_flips_a_close_melee_hit_and_stops_at_the_new_ac():
    raw = _block(
        actions=[_melee_weapon()],
        reactions=[
            {
                "name": "Parry",
                "desc": (
                    "Trigger: The bandit is hit by a melee attack roll while holding a weapon. "
                    "Response: The bandit adds 2 to its AC against that attack, possibly causing it to miss."
                ),
            }
        ],
    )
    record, bandit = _loaded(raw, "bandit", "Bandit")
    assert record.reactions[0].effect == "ac"
    assert record.reactions[0].bonus == 2
    assert record.reactions[0].melee_only is True
    assert record.reactions[0].needs_weapon is True
    assert "Parry" not in record.excluded_actions
    restored = Combatant.from_dict(bandit.to_dict())
    assert restored.reactions[0].bonus == 2

    asked = _strike(bandit, [12], "ask")
    assert asked.pending is not None
    assert [offer["label"] for offer in asked.pending["offers"]] == ["Parry"]
    asked.answer_reaction(asked.pending["offers"][0]["id"])
    assert bandit.hp == bandit.max_hp
    assert bandit.reaction_available is False
    assert "shield_ac" not in bandit.traits
    assert "uses Parry" in _logged(asked)
    assert "AC is 17" in _logged(asked)
    assert "The attack misses." in _logged(asked)

    _, skipped = _loaded(raw, "bandit", "Bandit")
    wide = _strike(skipped, [13, 4])
    assert skipped.hp == skipped.max_hp - 6
    assert skipped.reaction_available is True
    assert "uses Parry" not in _logged(wide)
    assert "does not use a reaction" in _logged(wide)

    _, crit = _loaded(raw, "bandit", "Bandit")
    critical = _strike(crit, [20, 3, 3])
    assert crit.hp < crit.max_hp
    assert crit.reaction_available is True
    assert "uses Parry" not in _logged(critical)

    bow = AttackOption(name="Bow", resolution="attack", attack_bonus=4, range_feet=80, damage=[{"type": "piercing", "expr": "1d6+2"}])
    _, ranged = _loaded(raw, "bandit", "Bandit")
    shot = _strike(ranged, [12, 4], attack=bow)
    assert shot.pending is None
    assert ranged.hp == ranged.max_hp - 6
    assert ranged.reaction_available is True

    bare = creature("bare", "Bare", "party", 15, 30, (1, 0), reactions=list(bandit.reactions))
    naked = _strike(bare, [12, 4], "ask")
    assert naked.pending is None
    assert bare.hp == bare.max_hp - 6

    choice, _reason = reaction_choice(
        asked,
        bandit,
        {"actor_id": "gob", "target_id": "bandit", "total": 16, "penalty": 0, "critical": False},
        "hit",
        [
            {"id": "ac:0:Parry", "label": "Parry", "bonus": 2},
            {"id": "cutting_words", "label": "Cutting Words (1d8)", "die": "1d8"},
        ],
    )
    assert choice == "ac:0:Parry"


def test_a_split_reaction_is_stored_and_not_offered():
    raw = _block(
        reactions=[
            {
                "name": "Split",
                "desc": (
                    "Trigger: While the pudding is Large or Medium and has 10+ Hit Points, it becomes Bloodied "
                    "or is subjected to Lightning or Slashing damage. "
                    "Response: The pudding splits into two new Black Puddings."
                ),
            },
            {
                "name": "Riposte",
                "desc": (
                    "Trigger: The pirate is hit by a melee attack roll while holding a weapon. "
                    "Response: The pirate adds 3 to its AC against that attack, possibly causing it to miss. "
                    "On a miss, the pirate makes one Rapier attack against the triggering creature if within range."
                ),
            },
        ]
    )
    record, pudding = _loaded(raw, "pudding", "Pudding")
    assert [item.effect for item in record.reactions] == ["unresolved", "unresolved"]
    assert record.excluded_actions == ["Split", "Riposte"]
    assert [item.name for item in pudding.reactions] == ["Split", "Riposte"]
    assert pudding.traits["excluded_features"] == ["Split", "Riposte"]
    fight = _strike(pudding, [12, 4], "ask")
    assert fight.pending is None
    assert pudding.hp == pudding.max_hp - 6
    assert "uses Split" not in _logged(fight)
    assert "uses Riposte" not in _logged(fight)


def test_a_sheet_reaction_subtracts_under_any_name():
    class Catalog:
        spells: dict = {}

    hero = public_character(
        {
            "id": 8,
            "name": "Lore",
            "overrideHitPoints": 30,
            "stats": [{"id": stat, "value": 10} for stat in range(1, 7)],
            "classes": [
                {
                    "id": 9,
                    "level": 5,
                    "definition": {"name": "Bard"},
                    "classFeatures": [
                        {"definition": {"id": 77, "name": "Mocking Word", "requiredLevel": 3}},
                        {"definition": {"id": 78, "name": "Battle Wit", "requiredLevel": 1}},
                        {"definition": {"id": 88, "name": "Hellish Rebuke", "requiredLevel": 1}},
                    ],
                }
            ],
            "actions": {
                "class": [
                    {
                        "name": "Mocking Word",
                        "componentId": 77,
                        "activation": {"activationType": 4},
                        "dice": {"diceString": "1d6"},
                        "range": {"range": 30},
                        "snippet": (
                            "When a creature you can see makes an attack roll or a damage roll, "
                            "you can expend one use of Battle Wit and subtract the number rolled from the roll."
                        ),
                    },
                    {
                        "name": "Hellish Rebuke",
                        "componentId": 88,
                        "activation": {"activationType": 4},
                        "snippet": "You point your finger, and the creature that damaged you is surrounded by flames.",
                    },
                    {
                        "name": "Battle Wit",
                        "componentId": 78,
                        "activation": {"activationType": 3},
                        "limitedUse": {"maxUses": 4, "numberUsed": 1},
                    },
                    {
                        "name": "Battle Wit",
                        "componentId": 78,
                        "activation": {"activationType": 3},
                        "limitedUse": {"maxUses": 4, "numberUsed": 0},
                    },
                ]
            },
        },
        Catalog(),
        (1, 0),
        "ddb-8",
    )
    mocking = next(item for item in hero.reactions if item.name == "Mocking Word")
    assert mocking.effect == "subtract"
    assert mocking.die == "1d6"
    assert mocking.range_feet == 30
    assert mocking.pool == "Battle Wit"
    assert "cutting_words" not in hero.traits
    assert hero.uses_max["Battle Wit"] == 4
    assert hero.uses_left["Battle Wit"] == 3
    refused = next(item for item in hero.reactions if item.name == "Hellish Rebuke")
    assert refused.effect == "unresolved"
    notes = " ".join(hero.traits["unresolved"])
    assert "Mocking Word" not in notes
    assert "Hellish Rebuke" in notes
    assert "Battle Wit" in notes

    encounter = _strike(hero, [8, 3], "ask")
    assert encounter.pending is not None
    assert [offer["label"] for offer in encounter.pending["offers"]] == ["Mocking Word"]
    encounter.answer_reaction(encounter.pending["offers"][0]["id"])
    assert hero.hp == hero.max_hp
    assert hero.uses_left["Battle Wit"] == 3
    assert hero.reaction_available is False
    assert "uses Mocking Word" in _logged(encounter)
    assert "Cutting Words" not in _logged(encounter)

    copy = creature(
        "lore",
        "Lore",
        "party",
        hero.ac,
        30,
        (1, 0),
        reactions=list(hero.reactions),
        uses_max={"Battle Wit": 4},
        uses_left={"Battle Wit": 4},
    )
    wide = _strike(copy, [18, 4])
    assert copy.uses_left["Battle Wit"] == 4
    assert copy.hp == copy.max_hp - 6
    assert "Mocking Word" not in _logged(wide)


def test_protective_magic_spends_its_listed_uses():
    raw = _block(
        name="Warden",
        armor_class=12,
        actions=[
            {
                "name": "Spellcasting",
                "desc": "The warden casts one of the following spells, using Intelligence as the spellcasting ability (spell save DC 15):",
            }
        ],
        reactions=[
            {
                "name": "Protective Magic",
                "desc": (
                    "The warden casts Counterspell or Shield in response to the spell's trigger, "
                    "using the same spellcasting ability as Spellcasting."
                ),
                "usage": {"type": "per day", "times": 3},
            }
        ],
    )
    record, warden = _loaded(raw, "warden", "Warden", ac=12)
    assert record.spell_save_dc == 15
    assert warden.spell_save_dc == 15
    assert {(item.name, item.spell_index) for item in warden.reactions} == {("Shield", "shield"), ("Counterspell", "counterspell")}
    assert warden.uses_max["Protective Magic"] == 3
    assert warden.uses_left["Protective Magic"] == 3
    assert "Protective Magic" not in record.excluded_actions

    fight = _strike(warden, [10])
    assert warden.hp == warden.max_hp
    assert warden.uses_left["Protective Magic"] == 2
    assert warden.traits["shield_ac"] == 5
    assert "casts Shield using Protective Magic" in _logged(fight)
    assert fight.pending is None

    _, empty = _loaded(raw, "warden", "Warden", ac=12)
    empty.uses_left["Protective Magic"] = 0
    empty.uses_max["Protective Magic"] = 0
    missed = _strike(empty, [10, 4], "ask")
    assert missed.pending is None
    assert empty.hp == empty.max_hp - 6
    assert empty.uses_left["Protective Magic"] == 0
    assert "casts Shield" not in _logged(missed)
