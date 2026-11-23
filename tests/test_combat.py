import json

import pandas as pd

from src.actions import GameAction, legal_actions, space_between, targets_for, weapon_reaches
from src.board import battle_payload
from src.targeting import (
    abilities_from,
    board_marks,
    facing_for_square,
    match_creature,
    match_facing,
    match_instant,
    match_square,
    preview_squares,
)
from toon_format import decode

from src.ai import (
    DECISION_MODEL,
    REPORT_MODEL,
    REPORT_SYSTEM,
    action_score,
    choose_action,
    heuristic_choice,
    parse_choice,
    render_prompt,
    response_kwargs,
    LLMClient,
)
from src.bestiary import AttackOption, SpellRecord, classify_monster, classify_spell, spell_damage_parts
from src.combatant import Combatant, blank_abilities, from_monster, save_bonuses_for
from src.map_view import PREVIEW, TARGET_LINE, board_figure
from src.eda import (
    concentration_summary,
    correlation_table,
    fight_factors,
    action_economy_points,
    describe_fights,
    latest_run_id,
    living_actions,
    luck_summary,
    pace_points,
    roster_table,
    slot_summary,
    outlier_fights,
    report_brief,
    split_win_rate,
    swing_summary,
    variation_notes,
    win_rate,
)
from src.encounter import Encounter, IllegalAction
from src.reactions import _spend_known_or_pool
from src.grid import area_squares, footprint
from src.paths import CACHE, FIXTURES
from src.rules import resolve_attack, resolve_save
from src.simulate import run_batch


class Scripted:
    def __init__(self, rolls):
        self.rolls = list(rolls)

    def randint(self, low, high):
        value = self.rolls.pop(0)
        if not low <= value <= high:
            raise AssertionError(f"{value} is outside {low}-{high}")
        return value


def load_spell(name: str) -> SpellRecord:
    payload = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    return classify_spell(payload)


def cache_spell(name: str) -> SpellRecord:
    payload = json.loads((CACHE / "spells" / f"{name}.json").read_text(encoding="utf-8"))
    return classify_spell(payload)


def _mat(encounter, actor):
    actions = legal_actions(encounter, actor, automated=False)
    return battle_payload(
        encounter,
        mode="combat",
        actor=actor,
        actions=actions,
        abilities=abilities_from(actions),
        armed=None,
        slot=None,
        selected_id=actor.id,
    )


def creature(creature_id, name, side, ac, hp, position, attacks=None, spells=None, slots=None, **kwargs):
    abilities = blank_abilities(kwargs.pop("abilities", None))
    return Combatant(
        id=creature_id,
        name=name,
        side=side,
        ac=ac,
        max_hp=hp,
        hp=hp,
        speed=kwargs.pop("speed", 30),
        abilities=abilities,
        save_bonuses=save_bonuses_for(abilities, kwargs.pop("proficient", []), 2),
        position=position,
        attacks=attacks or [],
        spells=spells or [],
        slot_max=dict(slots or {}),
        slots=dict(slots or {}),
        **kwargs,
    )


def sword():
    return AttackOption(
        name="Sword",
        resolution="attack",
        attack_bonus=4,
        reach_feet=5,
        damage=[{"type": "slashing", "expr": "1d6+2"}],
    )


def club():
    return AttackOption(
        name="Club",
        resolution="attack",
        attack_bonus=8,
        reach_feet=5,
        damage=[{"type": "bludgeoning", "expr": "2d10+6"}],
    )


def _aura_cells(caster, spell):
    return area_squares(tuple(caster.position), spell.area_type, int(spell.area_size or 0), None)


def _in_aura(creature, cells) -> bool:
    return bool(footprint(tuple(creature.position), creature.span) & cells)


def _guardian(spell, position, hit_points=40):
    caster = creature(
        "cleric", "Cleric", "party", 18, hit_points, position,
        spells=[spell], slots={3: 1}, spell_save_dc=15,
    )
    caster.concentrating_on = "spirit-guardians"
    caster.ongoing_slot = 3
    return caster


def test_attack_hit_miss_crit_and_immunity():
    attacker = creature("a", "A", "party", 10, 10, (0, 0), attacks=[sword()])
    target = creature("b", "B", "enemy", 15, 30, (1, 0))
    hit = resolve_attack(
        attacker, target, bonus=4, damage=[("slashing", "1d6+2")], rng=Scripted([11, 4]), melee=True, label="Sword"
    )
    assert hit.damage == 6
    assert target.hp == 24

    miss = resolve_attack(
        attacker, target, bonus=4, damage=[("slashing", "1d6+2")], rng=Scripted([1]), melee=True, label="Sword"
    )
    assert miss.damage == 0
    assert target.hp == 24

    crit_target = creature("c", "C", "enemy", 18, 30, (1, 0))
    crit = resolve_attack(
        attacker, crit_target, bonus=0, damage=[("slashing", "1d6+2")], rng=Scripted([20, 3, 3]), melee=True, label="Sword"
    )
    assert crit.damage == 8

    immune = creature("d", "D", "enemy", 10, 20, (1, 0), immunities=["fire"])
    saved = resolve_save(
        attacker,
        immune,
        dc=10,
        ability="dex",
        success="none",
        damage=[("fire", "2d6")],
        rng=Scripted([1, 6, 6]),
        label="breath",
    )
    assert saved.damage == 0
    assert immune.hp == 20


def test_half_damage_on_a_successful_save():
    source = creature("a", "A", "party", 10, 10, (0, 0))
    target = creature("b", "B", "enemy", 10, 40, (1, 0))
    effect = resolve_save(
        source,
        target,
        dc=12,
        ability="dex",
        success="half",
        damage=[("fire", "1d6")],
        rng=Scripted([15, 5]),
        label="Fireball",
    )
    assert effect.damage == 2
    assert target.hp == 38


def test_fireball_one_click_and_magic_missile():
    fireball = load_spell("fireball")
    missile = load_spell("magic-missile")
    wizard = creature(
        "wiz",
        "Wizard",
        "party",
        12,
        20,
        (2, 5),
        spells=[fireball, missile],
        slots={3: 1, 1: 1},
        spell_save_dc=14,
        spell_attack_bonus=6,
        caster_level=5,
    )
    first = creature("e1", "One", "enemy", 12, 30, (8, 5))
    second = creature("e2", "Two", "enemy", 12, 30, (8, 6))
    encounter = Encounter([wizard, first, second], seed=1, label="blast")
    encounter.start()
    encounter.rng = Scripted([1, 1, 1, 1, 1, 1, 1, 1, 1, 19, 1, 1, 1, 1, 1, 1, 1, 1])
    encounter.order = ["wiz", "e1", "e2"]
    encounter.turn_index = 0
    encounter._prepared = None
    encounter.begin_turn()
    assert encounter.current().id == "wiz"
    action = next(item for item in encounter.legal_actions() if item.spell_index == "fireball" and item.aim == (8, 5))
    encounter.apply(action, "test")
    assert first.hp == 22
    assert second.hp == 26
    assert wizard.slots[3] == 0

    encounter.rng = Scripted([1, 1, 1])
    # The enemies act before the wizard's next turn. Force the wizard back to the front.
    encounter.order = ["wiz", "e1", "e2"]
    encounter.turn_index = 0
    encounter.round = 2
    encounter._prepared = None
    encounter.over = False
    encounter.winner = None
    encounter.begin_turn()
    shot = next(item for item in encounter.legal_actions() if item.spell_index == "magic-missile" and item.target_id == "e1")
    before = first.hp
    encounter.apply(shot, "test")
    assert first.hp == before - 6


def test_heuristic_closes_then_attacks_and_llm_falls_back():
    enemy = creature(
        "gob",
        "Gob",
        "enemy",
        12,
        10,
        (0, 0),
        attacks=[sword()],
    )
    party = creature("pc", "PC", "party", 18, 20, (5, 0))
    encounter = Encounter([party, enemy], seed=1)
    choice = encounter and __import__("src.ai", fromlist=["heuristic_choice"]).heuristic_choice(encounter, enemy)
    assert choice.kind == "move"
    assert choice.destination is not None
    assert choice.destination[0] > 0

    enemy.position = (4, 0)
    attack = __import__("src.ai", fromlist=["heuristic_choice"]).heuristic_choice(encounter, enemy)
    assert attack.kind == "attack"
    assert attack.target_id == "pc"

    far = creature("far", "Far", "enemy", 12, 10, (0, 0), attacks=[sword()])
    near = creature("near", "Near", "party", 18, 20, (5, 0))
    started = Encounter([near, far], seed=1)
    _prepare(started, ["far", "near"], [])
    opening = __import__("src.ai", fromlist=["heuristic_choice"]).heuristic_choice(started, far)
    assert opening.kind == "move"
    started.apply(opening, "test")
    assert started.current().id == "far"
    assert far.action_available
    assert far.position[0] > 0

    class FakeClient:
        def __init__(self, replies):
            self.replies = list(replies)

        def complete(self, system, user, action_ids=None):
            return self.replies.pop(0)

    legal = encounter.legal_actions(enemy, automated=True)
    chosen, source, reason = choose_action(encounter, enemy, "llm", FakeClient(['{"action_id": 0, "reason": "press"}']))
    assert source == "llm"
    assert chosen.id == legal[0].id
    assert reason == "press"
    fallback, source, _reason = choose_action(encounter, enemy, "llm", FakeClient(["nope", "still nope"]))
    assert source == "fallback"
    assert fallback.kind == "attack"
    assert parse_choice('{"action_id": 99}', 3) is None


def test_the_model_sees_expected_damage_and_skips_a_lone_attack():
    foe = creature("pc", "PC", "party", 14, 30, (1, 0))
    downed = creature("down", "Downed", "party", 12, 10, (4, 4))
    downed.hp = 0
    enemy = creature("gob", "Gob", "enemy", 12, 20, (0, 0), attacks=[sword(), club()])
    encounter = Encounter([foe, downed, enemy], seed=1)
    prompt = render_prompt(encounter, enemy, encounter.legal_actions(enemy, automated=True))
    decoded = decode(prompt)
    assert decode(prompt) == decoded
    header = next(line for line in prompt.splitlines() if line.startswith("actions["))
    assert header.count("expected") == 1
    assert "fresh" in header and "opportunity" in header and "tick" in header
    assert all(row["name"] != "Downed" for row in decoded["combatants"])
    legal = encounter.legal_actions(enemy, automated=True)
    kinds = [row["kind"] for row in decoded["actions"]]
    assert kinds.index("attack") < kinds.index("move") < kinds.index("end")
    attacks = [row for row in decoded["actions"] if row["kind"] == "attack"]
    assert len(attacks) == 2
    by_id = {action.id: action for action in legal}
    for row in attacks:
        assert row["expected"] == round(action_score(encounter, enemy, by_id[row["id"]]), 1)
    moves = [row for row in decoded["actions"] if row["kind"] == "move"]
    assert moves
    assert moves[0]["fresh"] is not None
    assert moves[0]["opportunity"] is not None
    assert moves[0]["tick"] is not None

    sword_id = next(action.id for action in legal if action.attack_name == "Sword")
    chosen, source, reason = choose_action(
        encounter,
        enemy,
        "llm",
        _Replies([json.dumps({"action_id": sword_id, "reason": "focus the lighter hit"})]),
    )
    assert source == "llm"
    assert chosen.attack_name == "Sword"
    assert reason == "focus the lighter hit"

    enemy.movement_left = 0
    enemy.attacks = [sword()]

    class Silent:
        def complete(self, system, user, action_ids=None):
            raise AssertionError("the model was called for a lone attack")

    alone, source, reason = choose_action(encounter, enemy, "llm", Silent())
    assert source == "heuristic"
    assert alone.kind == "attack"
    assert reason == "Highest expected damage."

    enemy.movement_left = None
    enemy.attacks = [sword(), club()]
    enemy.uses_left["Rage"] = 1
    raging, source, reason = choose_action(encounter, enemy, "llm", Silent())
    assert source == "heuristic"
    assert raging.attack_name == "Rage"
    assert reason == "Rage is available."


class _Replies:
    def __init__(self, replies):
        self.replies = list(replies)

    def complete(self, system, user, action_ids=None):
        return self.replies.pop(0)


def test_decision_client_does_not_reason_and_report_stays_on_grok_4_7(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_REPORT_MODEL", raising=False)
    monkeypatch.setenv("XAI_API_KEY", "test-key")
    decision = LLMClient()
    report = LLMClient(report=True)
    assert decision.model == DECISION_MODEL
    assert report.model == REPORT_MODEL
    payload = response_kwargs(decision.model, "system", "user", action_ids=[2, 5])
    assert "reasoning" not in payload
    schema = payload["text_format"].model_json_schema()
    assert schema["properties"]["action_id"]["enum"] == [2, 5]
    monkeypatch.setenv("LLM_MODEL", "custom-decision")
    monkeypatch.setenv("LLM_REPORT_MODEL", "custom-report")
    assert LLMClient().model == "custom-decision"
    assert LLMClient(report=True).model == "custom-report"
    assert "two paragraphs" not in REPORT_SYSTEM.lower()
    assert "Do not restate the brief" in REPORT_SYSTEM
    assert response_kwargs(report.model, "system", "user", reasoning=True)["reasoning"] == {"effort": "medium"}


def test_a_fragile_enemy_stays_outside_spirit_guardians():
    spell = cache_spell("spirit-guardians")
    caster = _guardian(spell, (8, 0), 40)
    enemy = creature("gob", "Gob", "enemy", 12, 8, (0, 0), attacks=[sword()])
    encounter = Encounter([caster, enemy], seed=1)
    choice = heuristic_choice(encounter, enemy)
    cells = _aura_cells(caster, spell)
    if choice.kind == "move":
        assert choice.destination is not None
        assert not (footprint(choice.destination, enemy.span) & cells)
    else:
        assert choice.kind == "dodge"
    assert not _in_aura(enemy, cells)


def test_a_fragile_enemy_stays_when_the_opportunity_attack_costs_more_than_the_tick():
    spell = cache_spell("spirit-guardians")
    caster = _guardian(spell, (6, 6), 40)
    caster.attacks = [club()]
    enemy = creature("gob", "Gob", "enemy", 12, 8, (7, 6), attacks=[sword()])
    encounter = Encounter([caster, enemy], seed=1)
    opening = heuristic_choice(encounter, enemy)
    assert opening.kind == "attack"
    assert opening.target_id == "cleric"
    enemy.action_available = False
    enemy.attacks_made = 1
    assert heuristic_choice(encounter, enemy).kind == "end"


def test_a_small_opportunity_attack_does_not_keep_a_fragile_enemy_in_the_emanation():
    spell = cache_spell("spirit-guardians")
    caster = _guardian(spell, (6, 6), 40)
    caster.attacks = [sword()]
    enemy = creature("gob", "Gob", "enemy", 12, 8, (7, 6), attacks=[sword()])
    encounter = Encounter([caster, enemy], seed=1)
    cells = _aura_cells(caster, spell)
    opening = heuristic_choice(encounter, enemy)
    assert opening.kind == "attack"
    enemy.action_available = False
    enemy.attacks_made = 1
    leaving = heuristic_choice(encounter, enemy)
    assert leaving.kind == "move"
    assert leaving.destination is not None
    assert not (footprint(leaving.destination, enemy.span) & cells)


def test_nimble_escape_leaves_an_emanation_after_the_attack():
    spell = cache_spell("spirit-guardians")
    caster = _guardian(spell, (6, 6), 40)
    caster.attacks = [club()]
    escape = AttackOption(name="Nimble Escape", resolution="disengage", activation="bonus")
    enemy = creature("gob", "Gob", "enemy", 12, 8, (7, 6), attacks=[sword(), escape])
    encounter = Encounter([caster, enemy], seed=1)
    cells = _aura_cells(caster, spell)
    opening = heuristic_choice(encounter, enemy)
    assert opening.kind == "attack"
    assert opening.target_id == "cleric"
    enemy.action_available = False
    enemy.attacks_made = 1
    disengage = heuristic_choice(encounter, enemy)
    assert disengage.kind == "disengage"
    assert disengage.budget == "bonus"
    enemy.bonus_available = False
    enemy.traits["disengaged"] = True
    leaving = heuristic_choice(encounter, enemy)
    assert leaving.kind == "move"
    assert leaving.destination is not None
    assert not (footprint(leaving.destination, enemy.span) & cells)


def test_action_disengage_is_used_when_the_swing_and_the_tick_would_both_drop_them():
    spell = cache_spell("spirit-guardians")
    caster = _guardian(spell, (6, 6), 40)
    caster.attacks = [club()]
    enemy = creature("gob", "Gob", "enemy", 12, 8, (7, 6))
    encounter = Encounter([caster, enemy], seed=1)
    choice = heuristic_choice(encounter, enemy)
    assert choice.kind == "disengage"
    assert choice.budget == "action"


def test_a_fragile_enemy_attacks_then_leaves_spirit_guardians():
    spell = cache_spell("spirit-guardians")
    caster = _guardian(spell, (6, 6), 40)
    enemy = creature("gob", "Gob", "enemy", 12, 8, (7, 6), attacks=[sword()])
    encounter = Encounter([caster, enemy], seed=1)
    cells = _aura_cells(caster, spell)
    assert _in_aura(enemy, cells)
    opening = heuristic_choice(encounter, enemy)
    assert opening.kind == "attack"
    assert opening.target_id == "cleric"
    enemy.action_available = False
    enemy.attacks_made = 1
    leaving = heuristic_choice(encounter, enemy)
    assert leaving.kind == "move"
    assert leaving.destination is not None
    assert not (footprint(leaving.destination, enemy.span) & cells)


def test_a_slow_brute_stays_in_spirit_guardians_to_keep_the_melee():
    spell = cache_spell("spirit-guardians")
    caster = _guardian(spell, (6, 6), 30)
    enemy = creature("ogre", "Ogre", "enemy", 14, 90, (7, 6), attacks=[club()], speed=10)
    encounter = Encounter([caster, enemy], seed=1)
    opening = heuristic_choice(encounter, enemy)
    assert opening.kind == "attack"
    assert opening.target_id == "cleric"
    enemy.action_available = False
    enemy.attacks_made = 1
    assert heuristic_choice(encounter, enemy).kind == "end"


def test_a_tough_enemy_walks_into_spirit_guardians():
    spell = cache_spell("spirit-guardians")
    caster = _guardian(spell, (8, 0), 30)
    enemy = creature("ogre", "Ogre", "enemy", 14, 90, (2, 0), attacks=[club()], speed=30)
    encounter = Encounter([caster, enemy], seed=1)
    choice = heuristic_choice(encounter, enemy)
    assert choice.kind == "move"
    assert choice.destination is not None
    assert footprint(choice.destination, enemy.span) & _aura_cells(caster, spell)


def test_an_enemy_already_saved_against_the_emanation_does_not_retreat():
    spell = cache_spell("spirit-guardians")
    caster = _guardian(spell, (6, 6), 40)
    enemy = creature("gob", "Gob", "enemy", 12, 20, (7, 6), attacks=[sword()])
    encounter = Encounter([caster, enemy], seed=1)
    caster.traits["emanation_saved"] = {"turn": [encounter.round, encounter.turn_index], "ids": [enemy.id]}
    opening = heuristic_choice(encounter, enemy)
    assert opening.kind == "attack"
    enemy.action_available = False
    enemy.attacks_made = 1
    assert heuristic_choice(encounter, enemy).kind == "end"


def test_an_allied_emanation_does_not_stop_an_enemy_from_closing():
    spell = cache_spell("spirit-guardians")
    ally = creature(
        "druid", "Druid", "enemy", 14, 30, (0, 2),
        spells=[spell], slots={3: 1}, spell_save_dc=15,
    )
    ally.concentrating_on = "spirit-guardians"
    ally.ongoing_slot = 3
    foe = creature("pc", "PC", "party", 16, 20, (6, 0))
    enemy = creature("gob", "Gob", "enemy", 12, 12, (0, 0), attacks=[sword()])
    encounter = Encounter([ally, foe, enemy], seed=1)
    choice = heuristic_choice(encounter, enemy)
    assert choice.kind == "move"
    assert choice.destination is not None
    assert space_between(choice.destination, enemy, foe) < enemy.gap_to(foe)


def test_a_ranged_enemy_does_not_step_into_an_emanation_it_can_already_hit():
    spell = cache_spell("spirit-guardians")
    caster = _guardian(spell, (8, 0), 40)
    bow = AttackOption(
        name="Bow", resolution="attack", attack_bonus=6, range_feet=80,
        damage=[{"type": "piercing", "expr": "1d8+3"}],
    )
    enemy = creature("archer", "Archer", "enemy", 14, 40, (0, 0), attacks=[bow])
    encounter = Encounter([caster, enemy], seed=1)
    cells = _aura_cells(caster, spell)
    opening = heuristic_choice(encounter, enemy)
    assert opening.kind == "attack"
    enemy.action_available = False
    enemy.attacks_made = 1
    follow = heuristic_choice(encounter, enemy)
    if follow.kind == "move":
        assert follow.destination is not None
        assert not (footprint(follow.destination, enemy.span) & cells)
    else:
        assert follow.kind == "end"


def test_a_fragile_enemy_dashes_out_when_a_step_cannot():
    spell = cache_spell("spirit-guardians")
    caster = _guardian(spell, (6, 6), 40)
    enemy = creature("gob", "Gob", "enemy", 12, 8, (7, 6), attacks=[sword()], speed=10)
    encounter = Encounter([caster, enemy], seed=1)
    assert heuristic_choice(encounter, enemy).kind == "dash"


def test_illegal_move_is_rejected():
    party = creature("pc", "PC", "party", 18, 20, (1, 1), attacks=[sword()])
    enemy = creature("gob", "Gob", "enemy", 12, 10, (2, 1), attacks=[sword()])
    encounter = Encounter([party, enemy], seed=1)
    encounter.rng = Scripted([20, 1])
    encounter.start()
    encounter.begin_turn()
    bad = GameAction(id=0, kind="move", label="teleport", destination=(11, 11))
    try:
        encounter.apply(bad, "test")
        raised = False
    except IllegalAction:
        raised = True
    assert raised


def test_batch_is_reproducible_and_eda_flags_variation():
    party = creature(
        "pc",
        "PC",
        "party",
        20,
        40,
        (1, 1),
        attacks=[
            AttackOption(
                name="Sword",
                resolution="attack",
                attack_bonus=8,
                reach_feet=5,
                damage=[{"type": "slashing", "expr": "1d4+20"}],
            )
        ],
    )
    enemy = creature(
        "gob",
        "Gob",
        "enemy",
        10,
        8,
        (2, 1),
        attacks=[sword()],
        challenge_rating=0.25,
    )
    setup = Encounter([party, enemy], seed=3, label="duel", enemy_policy="heuristic").to_setup()
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        first = run_batch(setup, count=2, policy="heuristic", seed=3, run_dir=root / "a")
        second = run_batch(setup, count=2, policy="heuristic", seed=3, run_dir=root / "b")
        def body(path):
            rows = Path(path).read_text(encoding="utf-8").splitlines()
            return [",".join(line.split(",")[1:]) for line in rows]

        assert body(first["fights"]) == body(second["fights"])
        fights = pd.read_csv(first["fights"])
        assert len(fights) == 2
        assert set(fights["winner"]) <= {"party", "enemy", "draw"}

    frame = pd.DataFrame(
        [
            {"fight_id": "a", "label": "easy", "policy": "heuristic", "winner": "party", "rounds": 2, "damage_by_party": 10, "damage_by_enemy": 1, "party_win": 1, "enemy_count": 2, "mean_enemy_cr": 0.25, "party_size": 2, "mean_attack_bonus": 4, "start_distance": 15, "fallback_turns": 0, "llm_turns": 0},
            {"fight_id": "b", "label": "easy", "policy": "heuristic", "winner": "party", "rounds": 3, "damage_by_party": 12, "damage_by_enemy": 2, "party_win": 1, "enemy_count": 2, "mean_enemy_cr": 0.25, "party_size": 2, "mean_attack_bonus": 4, "start_distance": 15, "fallback_turns": 0, "llm_turns": 0},
            {"fight_id": "c", "label": "hard", "policy": "llm", "winner": "enemy", "rounds": 20, "damage_by_party": 30, "damage_by_enemy": 40, "party_win": 0, "enemy_count": 6, "mean_enemy_cr": 2, "party_size": 2, "mean_attack_bonus": 5, "start_distance": 20, "fallback_turns": 1, "llm_turns": 8},
            {"fight_id": "d", "label": "hard", "policy": "llm", "winner": "party", "rounds": 4, "damage_by_party": 18, "damage_by_enemy": 8, "party_win": 1, "enemy_count": 6, "mean_enemy_cr": 2, "party_size": 2, "mean_attack_bonus": 5, "start_distance": 20, "fallback_turns": 0, "llm_turns": 6},
        ]
    )
    assert win_rate(frame) == 0.75
    notes = variation_notes(frame)
    assert "party_size" in notes
    corr = correlation_table(frame)
    assert corr.loc["mean_enemy_cr", "party_win"] < 0
    flagged = outlier_fights(frame)
    assert "c" in set(flagged["fight_id"])


def test_describe_fights_splits_by_side_and_winner():
    fights = pd.DataFrame(
        [
            {
                "fight_id": "a",
                "winner": "party",
                "rounds": 2,
                "damage_by_party": 10,
                "damage_by_enemy": 4,
                "party_hp_remaining": 20,
                "enemy_hp_remaining": 0,
            },
            {
                "fight_id": "b",
                "winner": "enemy",
                "rounds": 4,
                "damage_by_party": 6,
                "damage_by_enemy": 12,
                "party_hp_remaining": 0,
                "enemy_hp_remaining": 8,
            },
        ]
    )
    described = describe_fights(fights)
    assert list(described.columns) == ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    assert described.index.names == ["Winner", "Measure", "Side"]
    measure_sides = [
        ("Rounds", "Fight"),
        ("Damage dealt", "Party"),
        ("Damage dealt", "Enemy"),
        ("Hit points remaining", "Party"),
        ("Hit points remaining", "Enemy"),
    ]
    assert list(described.index) == [
        (winner, measure, side)
        for winner in ("all", "party", "enemy")
        for measure, side in measure_sides
    ]
    assert set(described.xs("Rounds", level="Measure").index.get_level_values("Side")) == {"Fight"}
    assert described.loc[("all", "Rounds", "Fight"), "mean"] == 3
    assert described.loc[("party", "Rounds", "Fight"), "mean"] == 2
    assert described.loc[("enemy", "Rounds", "Fight"), "mean"] == 4
    party_damage = described.loc[("party", "Damage dealt", "Party")]
    assert party_damage["count"] == 1
    assert party_damage["mean"] == 10
    assert pd.isna(party_damage["std"])
    assert described.loc[("enemy", "Hit points remaining", "Enemy"), "mean"] == 8
    assert described.loc[("party", "Damage dealt", "Enemy"), "mean"] == 4
    assert described.loc[("enemy", "Damage dealt", "Party"), "mean"] == 6
    assert "draw" not in described.index.get_level_values("Winner")

    only_all = describe_fights(fights.drop(columns=["winner"]))
    assert set(only_all.index.get_level_values("Winner")) == {"all"}
    no_hp = describe_fights(fights.drop(columns=["party_hp_remaining", "enemy_hp_remaining"]))
    assert "Hit points remaining" not in set(no_hp.index.get_level_values("Measure"))
    drawn = describe_fights(pd.DataFrame([
        {
            "winner": "draw",
            "rounds": 5,
            "damage_by_party": 1,
            "damage_by_enemy": 1,
            "party_hp_remaining": 3,
            "enemy_hp_remaining": 4,
        },
    ]))
    assert list(drawn.index.get_level_values("Winner").unique()) == ["all", "draw"]
    assert describe_fights(pd.DataFrame()).empty
    assert describe_fights(pd.DataFrame([{"fight_id": "z", "winner": "party"}])).empty


def test_latest_run_id_prefers_the_session_batch():
    import os
    import tempfile
    import time
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        older = root / "older"
        newer = root / "newer"
        session = root / "session-batch"
        for folder in (older, newer, session):
            folder.mkdir()
            (folder / "fights.csv").write_text("fight_id\n1\n", encoding="utf-8")
        now = time.time()
        os.utime(older / "fights.csv", (now - 200, now - 200))
        os.utime(session / "fights.csv", (now - 100, now - 100))
        os.utime(newer / "fights.csv", (now, now))
        assert latest_run_id(root, session) == "session-batch"
        assert latest_run_id(root) == "newer"
        pending = root / "pending"
        pending.mkdir()
        assert latest_run_id(root, pending) is None


def test_box_plot_frames_keep_one_point_per_fight():
    fights = pd.DataFrame(
        [
            {"fight_id": "a", "winner": "party", "rounds": 2, "damage_by_party": 10, "damage_by_enemy": 4, "party_win": 1},
            {"fight_id": "b", "winner": "enemy", "rounds": 0, "damage_by_party": 3, "damage_by_enemy": 9, "party_win": 0},
        ]
    )
    factors = pd.DataFrame(
        [
            {"fight_id": "a", "party_turns": 4, "enemy_turns": 3, "party_hostile": 2, "enemy_hostile": 1},
            {"fight_id": "b", "party_turns": 1, "enemy_turns": 5, "party_hostile": 1, "enemy_hostile": 4},
        ]
    )
    points = action_economy_points(fights, factors)
    assert len(points) == 8
    party_actions = points[(points["winner"] == "party") & (points["measure"] == "Party actions")]
    enemy_actions = points[(points["winner"] == "enemy") & (points["measure"] == "Enemy actions")]
    assert float(party_actions["count"].iloc[0]) == 4
    assert float(enemy_actions["count"].iloc[0]) == 5
    assert set(points["measure"]) == {
        "Party actions",
        "Enemy actions",
        "Party attacks and spells",
        "Enemy attacks and spells",
    }
    pace = pace_points(fights)
    assert len(pace) == 2
    assert set(pace["side"]) == {"Party", "Enemy"}
    assert float(pace.loc[pace["side"] == "Party", "damage_per_round"].iloc[0]) == 5
    assert float(pace.loc[pace["side"] == "Enemy", "damage_per_round"].iloc[0]) == 2


def test_luck_summary_splits_dice_by_side_and_winner():
    turns = pd.DataFrame(
        [
            {
                "fight_id": "a",
                "side": "party",
                "actor_name": "Barbarian",
                "detail": (
                    "Attack  advantage 13 and 2, keeps 13 + 7 = 20 vs AC 15.\n"
                    "1d10+4: dice 9, modifier +4, total 13 slashing.\n"
                    "Goblin resists Topple (4+2=6 vs DC 13).\n"
                    "Sample Cleric casts Cure Wounds (2d8+3 -> 11). Sample Cleric regains 11 hit points.\n"
                    "Graze deals 4 slashing.\n"
                    "Initiative: Goblin 18, Barbarian 12."
                ),
            },
            {"fight_id": "a", "side": "enemy", "actor_name": "Goblin", "detail": "Goblin ends the turn."},
            {
                "fight_id": "b",
                "side": "enemy",
                "actor_name": "Goblin",
                "detail": (
                    "Attack  14 + 7 = 21 vs AC 15.\n"
                    "2d6: dice 3 + 4, modifier +2, total 9 slashing."
                ),
            },
        ]
    )
    fights = pd.DataFrame(
        [
            {"fight_id": "a", "winner": "party"},
            {"fight_id": "b", "winner": "enemy"},
        ]
    )
    summary = luck_summary(turns, fights)
    assert summary["d20_count"] == 4
    assert summary["d20_mean"] == 8.25
    assert summary["damage_count"] == 2
    assert summary["damage_percentile"] == 67.5
    groups = summary["groups"].set_index(["side", "winner"])
    assert float(groups.loc[("Party", "party"), "d20_mean"]) == 7.5
    assert int(groups.loc[("Party", "party"), "d20_count"]) == 2
    assert float(groups.loc[("Party", "party"), "damage_percentile"]) == 85
    assert int(groups.loc[("Party", "party"), "damage_count"]) == 1
    assert float(groups.loc[("Enemy", "party"), "d20_mean"]) == 4
    assert int(groups.loc[("Enemy", "party"), "d20_count"]) == 1
    assert int(groups.loc[("Enemy", "party"), "damage_count"]) == 0
    assert float(groups.loc[("Enemy", "enemy"), "d20_mean"]) == 14
    assert float(groups.loc[("Enemy", "enemy"), "damage_percentile"]) == 50
    assert ("Party", "enemy") not in groups.index
    text = report_brief(fights, turns=turns)
    assert "Mean d20: 8.25 over 4 faces" in text
    assert "Luck for Party in fights won by party" in text
    legacy = luck_summary(
        pd.DataFrame([{
            "fight_id": "c",
            "side": "party",
            "actor_name": "Wizard",
            "detail": "Goblin 1 fails a DC 14 DEX save against Wizard's Fireball L3 at (8, 5) (8+2=10). Dice 8d6 -> 30.",
        }, {
            "fight_id": "c",
            "side": "enemy",
            "actor_name": "Goblin 1",
            "detail": "Goblin 1 ends the turn.",
        }]),
        pd.DataFrame([{"fight_id": "c", "winner": "party"}]),
    )
    legacy_groups = legacy["groups"].set_index(["side", "winner"])
    assert int(legacy_groups.loc[("Enemy", "party"), "d20_count"]) == 1
    assert float(legacy_groups.loc[("Enemy", "party"), "d20_mean"]) == 8
    assert int(legacy_groups.loc[("Party", "party"), "damage_count"]) == 1
    savage = luck_summary(
        pd.DataFrame([{
            "fight_id": "d",
            "side": "party",
            "actor_name": "Barbarian",
            "detail": "1d6+2: dice 4, modifier +2, total 6 slashing.\nSavage Attacker keeps 4 (other roll 1).",
        }]),
        pd.DataFrame([{"fight_id": "d", "winner": "party"}]),
    )
    assert savage["damage_count"] == 2


def _spell(index: str, name: str, level: int) -> SpellRecord:
    return SpellRecord(
        index=index,
        name=name,
        level=level,
        kind="attack",
        casting_time="1 action",
        duration="instantaneous",
        concentration=False,
        range_mode="ranged",
        range_feet=120,
        area_type=None,
        area_size=None,
        attack_type="ranged",
        dc_ability=None,
        dc_success=None,
        damage=[],
        healing={},
        ongoing=False,
        exclude_reason=None,
        classes=[],
    )


def test_a_spent_slot_is_counted_and_a_free_casting_is_not():
    missile = _spell("magic-missile", "Magic Missile", 1)
    cantrip = _spell("fire-bolt", "Fire Bolt", 0)
    hero = creature(
        "hero",
        "Aria",
        "party",
        15,
        20,
        (1, 1),
        spells=[missile, cantrip],
        slots={1: 2, 3: 1},
    )
    hero.traits["free_spells"] = {"magic-missile": "Magic Missile"}
    hero.uses_max["Magic Missile"] = 1
    foe = creature("foe", "Goblin", "enemy", 12, 10, (4, 1))
    encounter = Encounter([hero, foe])
    encounter.note_slot(hero, 2)
    encounter.start()
    assert encounter.summary("gate")["party_slots"] == 0
    assert encounter._spend_spell_slot(hero, cantrip, 0) == ""
    assert encounter._spend_spell_slot(hero, missile, 1) == "Magic Missile"
    assert encounter._spend_spell_slot(hero, missile, 3) == ""
    summary = encounter.summary("gate")
    assert summary["party_slots"] == 1
    assert summary["enemy_slots"] == 0
    assert summary["party_slot_levels"] == 3
    assert summary["party_slots_by_level"] == '{"3": 1}'
    assert summary["enemy_slots_by_level"] == "{}"
    assert hero.slots[1] == 2
    assert hero.slots[3] == 0
    assert hero.uses_left["Magic Missile"] == 0


def test_a_reaction_spell_slot_is_counted():
    shield = _spell("shield", "Shield", 1)
    hero = creature("hero", "Aria", "party", 15, 20, (1, 1), spells=[shield], slots={1: 2})
    foe = creature("foe", "Goblin", "enemy", 12, 10, (4, 1))
    encounter = Encounter([hero, foe])
    encounter.start()
    assert _spend_known_or_pool(encounter, hero, "shield", 1) == ""
    assert encounter.summary("gate")["party_slots"] == 1
    assert encounter.summary("gate")["party_slot_levels"] == 1
    assert hero.slots[1] == 1
    hero.traits["free_spells"] = {"shield": "Shield"}
    hero.uses_left["Shield"] = 1
    assert _spend_known_or_pool(encounter, hero, "shield", 1) == "a free casting"
    assert encounter.summary("gate")["party_slots"] == 1
    assert hero.slots[1] == 1


def test_slot_summary_groups_by_side_and_winner():
    fights = pd.DataFrame(
        [
            {
                "fight_id": "a",
                "winner": "party",
                "party_slots": 2,
                "enemy_slots": 0,
                "party_slot_levels": 2,
                "enemy_slot_levels": 0,
                "party_slots_by_level": '{"1": 2}',
                "enemy_slots_by_level": "{}",
            },
            {
                "fight_id": "b",
                "winner": "party",
                "party_slots": 0,
                "enemy_slots": 1,
                "party_slot_levels": 0,
                "enemy_slot_levels": 3,
                "party_slots_by_level": "{}",
                "enemy_slots_by_level": '{"3": 1}',
            },
            {
                "fight_id": "c",
                "winner": "enemy",
                "party_slots": 0,
                "enemy_slots": 0,
                "party_slot_levels": 0,
                "enemy_slot_levels": 0,
                "party_slots_by_level": "{}",
                "enemy_slots_by_level": "{}",
            },
        ]
    )
    summary = slot_summary(fights).set_index(["winner", "side"])
    assert list(summary.index) == [("party", "party"), ("party", "enemy"), ("enemy", "party"), ("enemy", "enemy")]
    assert int(summary.loc[("party", "party"), "fights"]) == 2
    assert float(summary.loc[("party", "party"), "mean_slots"]) == 1
    assert float(summary.loc[("party", "party"), "mean_slot_levels"]) == 1
    assert float(summary.loc[("party", "party"), "level_1"]) == 1
    assert float(summary.loc[("party", "enemy"), "mean_slots"]) == 0.5
    assert float(summary.loc[("party", "enemy"), "mean_slot_levels"]) == 1.5
    assert float(summary.loc[("party", "enemy"), "level_3"]) == 0.5
    assert float(summary.loc[("enemy", "party"), "mean_slots"]) == 0
    assert "level_2" not in summary.columns
    assert slot_summary(pd.DataFrame([{"fight_id": "z", "winner": "party"}])).empty
    text = report_brief(fights)
    assert "Spell slots for the party in fights won by party: mean 1.00 slots" in text
    assert "3rd 0.50" in text


def test_roster_collapses_repeated_names(tmp_path):
    run = tmp_path / "batch"
    run.mkdir()
    setup = {
        "label": "gate",
        "combatants": [
            {"name": "Aria", "side": "party"},
            {"name": "Goblin", "side": "enemy"},
            {"name": "Goblin", "side": "enemy"},
            {"name": "Wolf", "side": "enemy"},
        ],
    }
    (run / "setup.json").write_text(json.dumps(setup), encoding="utf-8")
    fights = pd.DataFrame([{"fight_id": "batch-1", "run_id": "batch", "label": "gate", "winner": "party"}])
    table = roster_table(tmp_path, fights)
    assert table.iloc[0]["party"] == "Aria"
    assert table.iloc[0]["enemy"] == "Goblin ×2, Wolf"
    assert bool(table.iloc[0]["saved"])
    missing = roster_table(tmp_path, pd.DataFrame([{"run_id": "gone", "label": "elsewhere"}]))
    assert not bool(missing.iloc[0]["saved"])
    text = report_brief(fights, rosters=table)
    assert "Roster for gate: party Aria. enemy Goblin ×2, Wolf." in text
    assert "The roster was not saved." in report_brief(fights, rosters=missing)


def test_living_actions_drop_end_and_a_downed_skip():
    turns = pd.DataFrame(
        [
            {"side": "enemy", "action_kind": "attack", "source": "llm", "detail": "Shortbow. Reason: Highest expected damage."},
            {"side": "enemy", "action_kind": "move", "source": "llm", "detail": "Move to (3, 6). Reason: Step closer; the new attack is worth more than the opportunity attack."},
            {"side": "enemy", "action_kind": "move", "source": "heuristic", "detail": "Move to (4, 4)."},
            {"side": "enemy", "action_kind": "end", "source": "llm", "detail": "Goblin Warrior 1 ends the turn. Reason: End turn."},
            {"side": "enemy", "action_kind": "end", "source": "heuristic", "detail": "Goblin Warrior 2 ends the turn."},
            {"side": "enemy", "action_kind": "skip", "source": "engine", "detail": "Goblin Warrior 1 is at 0 hit points and skips."},
            {"side": "party", "action_kind": "feature", "source": "heuristic", "detail": "Sample Barbarian rages."},
        ]
    )
    kept = living_actions(turns)
    assert set(kept["action_kind"]) == {"attack", "move", "feature"}
    enemy = kept[kept["side"] == "enemy"]
    counts = enemy["source"].value_counts()
    assert int(counts["llm"]) == 2
    assert int(counts["heuristic"]) == 1
    assert "engine" not in set(enemy["source"])
    assert "skip" not in set(kept["action_kind"])
    assert "end" not in set(kept["action_kind"])


def test_fight_factors_read_initiative_first_blood_and_concentration():
    fights = pd.DataFrame(
        [
            {"fight_id": "a", "party_win": 1, "winner": "party", "rounds": 2, "damage_by_party": 20, "damage_by_enemy": 4, "party_size": 2, "enemy_count": 2, "party_down": 0, "enemy_down": 2, "label": "skirmish"},
            {"fight_id": "b", "party_win": 0, "winner": "enemy", "rounds": 1, "damage_by_party": 0, "damage_by_enemy": 12, "party_size": 2, "enemy_count": 1, "party_down": 1, "enemy_down": 0, "label": "skirmish"},
            {"fight_id": "c", "party_win": 1, "winner": "party", "rounds": 1, "damage_by_party": 16, "damage_by_enemy": 0, "party_size": 1, "enemy_count": 2, "party_down": 0, "enemy_down": 2, "label": "skirmish"},
        ]
    )
    turns = pd.DataFrame(
        [
            {"fight_id": "a", "round": 1, "actor_name": "Goblin", "side": "enemy", "action_kind": "end", "detail": "Goblin ends the turn. Goblin drops to 0 hit points. Goblin loses concentration on spirit-guardians."},
            {"fight_id": "a", "round": 1, "actor_name": "Cleric", "side": "party", "action_kind": "spell", "detail": "Goblin holds concentration (12+2=14 vs DC 10). Orc drops to 0 hit points. Wolf drops to 0 hit points."},
            {"fight_id": "a", "round": 1, "actor_name": "Wolf", "side": "enemy", "action_kind": "attack", "detail": "Cleric fails the concentration save (3+1=4 vs DC 10) and loses bless."},
            {"fight_id": "b", "round": 1, "actor_name": "Ogre", "side": "enemy", "action_kind": "attack", "detail": "Fighter drops to 0 hit points."},
            {"fight_id": "c", "round": 1, "actor_name": "Downed", "side": "enemy", "action_kind": "skip", "detail": "Downed is at 0 hit points and skips."},
            {"fight_id": "c", "round": 1, "actor_name": "Wizard", "side": "party", "action_kind": "spell", "detail": "Goblin 1 drops to 0 hit points. Goblin 2 drops to 0 hit points."},
        ]
    )
    factors = fight_factors(fights, turns).set_index("fight_id")
    assert factors.loc["a", "first_side"] == "enemy"
    assert factors.loc["a", "first_blood_victim"] == "enemy"
    assert factors.loc["a", "first_blood_scorer"] == "party"
    assert int(factors.loc["a", "first_blood_round"]) == 1
    assert int(factors.loc["a", "max_swing"]) == 2
    assert int(factors.loc["a", "concentration_fails"]) == 1
    assert int(factors.loc["a", "concentration_holds"]) == 1
    assert int(factors.loc["a", "concentration_drops"]) == 1
    assert int(factors.loc["a", "party_turns"]) == 1
    assert int(factors.loc["a", "enemy_turns"]) == 1
    assert factors.loc["b", "first_blood_victim"] == "party"
    assert factors.loc["b", "first_blood_scorer"] == "enemy"
    assert factors.loc["c", "first_side"] == "party"
    assert factors.loc["c", "first_blood_scorer"] == "party"
    assert int(factors.loc["c", "max_swing"]) == 2
    assert int(factors.loc["c", "enemy_turns"]) == 0
    rates = split_win_rate(fights, factors.reset_index(), "first_side")
    assert set(rates["group"]) == {"party", "enemy"}
    assert float(rates.loc[rates["group"] == "enemy", "party_win"].iloc[0]) == 0.5
    swing = swing_summary(factors.reset_index())
    assert swing == {"fights": 3, "area_fights": 2, "largest": 2}
    conc = concentration_summary(fights, factors.reset_index())
    assert (conc["fails"], conc["holds"], conc["drops"]) == (1, 1, 1)
    assert conc["broke_fights"] == 1
    assert conc["held_fights"] == 0
    text = report_brief(fights, turns=turns)
    assert "Pearson" not in text
    assert "First turn:" in text
    assert "Concentration saves failed: 1" in text


def finish_turn(encounter: Encounter) -> None:
    action = next(item for item in encounter.legal_actions() if item.kind == "end")
    encounter.apply(action, "test")


def _prepare(encounter: Encounter, order: list[str], rolls: list[int]) -> None:
    encounter.start()
    encounter.rng = Scripted(rolls)
    encounter.order = order
    encounter.turn_index = 0
    encounter.round = 1
    encounter._prepared = None


def test_recharge_blocks_then_returns():
    breath = AttackOption(
        name="Breath",
        resolution="save",
        damage=[{"type": "cold", "expr": "1d4"}],
        dc_ability="dex",
        dc_value=15,
        dc_success="none",
        recharge_min=5,
    )
    dragon = creature("dragon", "Dragon", "enemy", 18, 80, (0, 0), attacks=[breath])
    tank = creature("tank", "Tank", "party", 18, 80, (1, 0), attacks=[sword()])
    encounter = Encounter([dragon, tank], seed=1)
    _prepare(encounter, ["dragon", "tank"], [1, 2, 3, 5])
    opening = encounter.legal_actions(dragon)
    breath_action = next(action for action in opening if action.attack_name == "Breath")
    encounter.apply(breath_action, "test")
    assert "Breath" in dragon.recharging
    assert tank.hp == 78
    finish_turn(encounter)
    encounter.apply(next(action for action in encounter.legal_actions(tank) if action.kind == "dodge"), "test")
    finish_turn(encounter)
    encounter.begin_turn()
    assert dragon.recharging == ["Breath"]
    assert "still recharging" in encounter.log[-1]
    assert all(action.attack_name != "Breath" for action in encounter.legal_actions(dragon))
    encounter.apply(next(action for action in encounter.legal_actions(dragon) if action.kind == "dodge"), "test")
    finish_turn(encounter)
    encounter.begin_turn()
    encounter.apply(next(action for action in encounter.legal_actions(tank) if action.kind == "dodge"), "test")
    finish_turn(encounter)
    encounter.begin_turn()
    assert dragon.recharging == []
    assert any(action.attack_name == "Breath" for action in encounter.legal_actions(dragon))
    assert any("recharges Breath" in line for line in encounter.log)


def test_ongoing_aura_pulses_until_concentration_breaks():
    spell = SpellRecord(
        index="aura",
        name="Aura",
        level=1,
        kind="save",
        casting_time="1 action",
        duration="1 minute",
        concentration=True,
        range_mode="self",
        range_feet=0,
        area_type="sphere",
        area_size=5,
        attack_type=None,
        dc_ability="dex",
        dc_success="half",
        damage=[{"type": "radiant", "dice": "1d4"}],
        healing={},
        ongoing=True,
        exclude_reason=None,
        classes=[],
    )
    caster = creature(
        "wiz",
        "Wizard",
        "party",
        10,
        30,
        (0, 0),
        spells=[spell],
        slots={1: 1},
        spell_save_dc=13,
        immunities=["radiant"],
    )
    foe = creature(
        "foe",
        "Foe",
        "enemy",
        10,
        40,
        (1, 0),
        attacks=[
            AttackOption(
                name="Claw",
                resolution="attack",
                attack_bonus=10,
                reach_feet=5,
                damage=[{"type": "slashing", "expr": "1d6"}],
            )
        ],
    )
    encounter = Encounter([caster, foe], seed=1)
    # Cast hits both squares: caster save, caster die, foe save, foe die.
    # The pulse is a Dexterity save with advantage, because the foe is dodging,
    # then one damage die. The claw is a straight roll and then a concentration save.
    _prepare(encounter, ["wiz", "foe"], [10, 3, 1, 4, 1, 4, 4, 10, 6, 1])
    cast = next(action for action in encounter.legal_actions(caster) if action.spell_index == "aura")
    encounter.apply(cast, "player")
    assert caster.slots[1] == 0
    assert caster.concentrating_on == "aura"
    assert foe.hp == 36
    finish_turn(encounter)
    encounter.apply(next(action for action in encounter.legal_actions(foe) if action.kind == "dodge"), "player")
    finish_turn(encounter)
    move = next(action for action in encounter.legal_actions(caster) if action.kind == "move" and action.destination == (0, 1))
    encounter.apply(move, "player")
    assert foe.hp == 32
    assert any("ongoing" in line for line in encounter.log)
    finish_turn(encounter)
    encounter.begin_turn()
    encounter.apply(next(action for action in encounter.legal_actions(foe) if action.attack_name == "Claw"), "player")
    assert caster.concentrating_on is None
    assert "loses aura" in encounter.log[-1] or any("loses aura" in line for line in encounter.log)
    held = foe.hp
    encounter.begin_turn()
    assert foe.hp == held


def test_move_then_attack_and_budgets():
    hero = creature("hero", "Hero", "party", 16, 30, (0, 0), attacks=[sword()], speed=30)
    foe = creature("foe", "Foe", "enemy", 12, 40, (3, 0), attacks=[sword()])
    encounter = Encounter([hero, foe], seed=1)
    _prepare(encounter, ["hero", "foe"], [12, 4])
    move = next(action for action in encounter.legal_actions(hero) if action.destination == (2, 0))
    encounter.apply(move, "test")
    assert hero.position == (2, 0)
    assert hero.movement_left == 20
    assert encounter.current().id == "hero"
    strike = next(action for action in encounter.legal_actions(hero) if action.attack_name == "Sword")
    encounter.apply(strike, "test")
    assert hero.action_available is False
    assert encounter.current().id == "hero"
    assert foe.hp < 40
    again = GameAction(id=0, kind="attack", label="Sword again", attack_name="Sword", target_id="foe")
    try:
        encounter.apply(again, "test")
        rejected = False
    except IllegalAction:
        rejected = True
    assert rejected
    finish_turn(encounter)
    assert encounter.current().id == "foe"


def test_movement_cannot_exceed_remaining_speed():
    hero = creature("hero", "Hero", "party", 16, 20, (0, 0), attacks=[sword()], speed=30)
    foe = creature("foe", "Foe", "enemy", 12, 20, (0, 8), attacks=[sword()])
    encounter = Encounter([hero, foe], seed=1)
    _prepare(encounter, ["hero", "foe"], [])
    full = next(action for action in encounter.legal_actions(hero) if action.destination == (6, 0))
    encounter.apply(full, "test")
    assert hero.movement_left == 0
    too_far = GameAction(id=0, kind="move", label="too far", destination=(7, 0), budget="move")
    try:
        encounter.apply(too_far, "test")
        rejected = False
    except IllegalAction:
        rejected = True
    assert rejected
    assert hero.position == (6, 0)

    hero.position = (0, 0)
    hero.movement_left = None
    encounter._prepared = None
    encounter.begin_turn()
    first = next(action for action in encounter.legal_actions(hero) if action.destination == (2, 0))
    encounter.apply(first, "test")
    assert hero.movement_left == 20
    second = next(action for action in encounter.legal_actions(hero) if action.destination == (6, 0))
    encounter.apply(second, "test")
    assert hero.position == (6, 0)
    assert hero.movement_left == 0
    assert all(action.destination != (7, 0) for action in encounter.legal_actions(hero))


def test_healing_word_spends_the_bonus_action():
    word = load_spell("healing-word")
    caster = creature(
        "wiz",
        "Wizard",
        "party",
        12,
        20,
        (0, 0),
        spells=[word],
        slots={1: 1},
        spellcasting_mod=3,
    )
    ally = creature("ally", "Ally", "party", 16, 20, (1, 0))
    ally.hp = 8
    foe = creature("foe", "Foe", "enemy", 12, 20, (4, 0), attacks=[sword()])
    encounter = Encounter([caster, ally, foe], seed=1)
    _prepare(encounter, ["wiz", "ally", "foe"], [1, 1])
    ally.hp = 8
    heal = next(action for action in encounter.legal_actions(caster) if action.spell_index == "healing-word")
    assert heal.budget == "bonus"
    encounter.apply(heal, "test")
    assert caster.action_available is True
    assert caster.bonus_available is False
    assert caster.slots[1] == 0
    assert ally.hp == 8 + 5
    assert encounter.current().id == "wiz"


def test_nimble_escape_spends_the_bonus_action():
    record = classify_monster(json.loads((FIXTURES / "goblin-warrior.json").read_text(encoding="utf-8")))
    goblin = from_monster(record, creature_id="goblin", name="Goblin Warrior 1", side="enemy", position=(1, 0))
    hero = creature("hero", "Hero", "party", 16, 20, (0, 0), attacks=[sword()])
    encounter = Encounter([hero, goblin], seed=1)
    _prepare(encounter, ["goblin", "hero"], [])
    escape = next(action for action in encounter.legal_actions(goblin) if action.attack_name == "Nimble Escape")
    assert escape.budget == "bonus"
    encounter.apply(escape, "test")
    assert goblin.bonus_available is False
    assert goblin.action_available is True
    assert goblin.position == (1, 0)
    assert any("uses Nimble Escape and disengages." in line for line in encounter.log)
    assert encounter.current().id == "goblin"


def _aura() -> SpellRecord:
    return SpellRecord(
        index="test-aura",
        name="Test Aura",
        level=1,
        kind="save",
        casting_time="1 action",
        duration="1 minute",
        concentration=False,
        range_mode="self",
        range_feet=0,
        area_type="sphere",
        area_size=10,
        attack_type=None,
        dc_ability="dex",
        dc_success="half",
        damage=[{"damage_at_slot_level": {"1": "1d6"}}],
        healing={},
        ongoing=False,
        exclude_reason=None,
        classes=[],
    )


def test_abilities_collapse_targets_into_one_button():
    wizard = creature(
        "wiz",
        "Wizard",
        "party",
        12,
        32,
        (2, 2),
        spells=[
            load_spell("fire-bolt"),
            load_spell("magic-missile"),
            load_spell("burning-hands"),
            load_spell("fireball"),
            load_spell("healing-word"),
        ],
        slots={1: 4, 2: 3, 3: 2},
        spellcasting_mod=3,
        caster_level=5,
    )
    fighter = creature("ally", "Fighter", "party", 18, 44, (2, 3), attacks=[sword()])
    fighter.hp = 8
    east = creature("east", "Goblin East", "enemy", 15, 10, (4, 2))
    north = creature("north", "Goblin North", "enemy", 15, 10, (2, 5))
    encounter = Encounter([wizard, fighter, east, north], width=12, height=12)
    actions = legal_actions(encounter, wizard, automated=False)
    abilities = abilities_from(actions)
    labels = [ability.label for ability in abilities]

    fireballs = [ability for ability in abilities if ability.label == "Fireball"]
    assert len(fireballs) == 1
    assert fireballs[0].needs == "square"
    assert fireballs[0].slots == (3,)
    assert len([action for action in actions if action.spell_index == "fireball"]) > 1
    assert not any(ability.kind == "move" for ability in abilities)

    bolt = next(ability for ability in abilities if ability.label == "Fire Bolt")
    assert bolt.needs == "creature"
    aimed_bolt = match_creature(actions, bolt, 0, "east")
    assert aimed_bolt is not None and aimed_bolt.target_id == "east"
    assert match_creature(actions, bolt, 0, "missing") is None

    missile = next(ability for ability in abilities if ability.label == "Magic Missile")
    assert missile.needs == "creature"
    assert missile.slots == (1, 2, 3)
    second = match_creature(actions, missile, 2, "north")
    assert second is not None and second.slot_level == 2

    hands = next(ability for ability in abilities if ability.label == "Burning Hands")
    assert hands.needs == "facing"
    eastward = match_facing(actions, hands, 1, (2, 2), (6, 2))
    assert eastward is not None and eastward.facing == (1, 0)
    assert match_facing(actions, hands, 1, (2, 2), (2, 2)) is None

    word = next(ability for ability in abilities if ability.label == "Healing Word")
    assert word.budget == "bonus"
    assert word.needs == "creature"
    heal = match_creature(actions, word, 1, "ally")
    assert heal is not None and heal.target_id == "ally"

    dash = next(ability for ability in abilities if ability.label == "Dash")
    assert dash.needs == "none"
    assert match_instant(actions, dash, None).kind == "dash"

    burst = match_square(actions, fireballs[0], 3, (4, 3))
    assert burst is not None and burst.aim == (4, 3)
    cells = preview_squares(encounter, wizard, burst)
    assert (4, 2) in cells
    assert match_square(actions, fireballs[0], 3, (11, 11)) is None

    highlight, aim, targets, shown = board_marks(encounter, wizard, actions, fireballs[0], 3, (4, 3))
    assert highlight == set()
    assert (4, 2) in targets
    assert (4, 3) in aim
    assert (4, 2) in shown
    assert "Fireball" in labels


def test_self_centered_spell_confirms_without_an_aim_click():
    caster = creature(
        "wiz",
        "Wizard",
        "party",
        12,
        20,
        (2, 2),
        spells=[_aura()],
        slots={1: 2},
    )
    foe = creature("foe", "Goblin", "enemy", 12, 10, (3, 2))
    encounter = Encounter([caster, foe], width=8, height=8)
    actions = legal_actions(encounter, caster, automated=False)
    aura = next(ability for ability in abilities_from(actions) if ability.label == "Test Aura")
    assert aura.needs == "confirm"
    action = match_instant(actions, aura, 1)
    assert action is not None and action.aim == (2, 2)
    _movement, _aim, _targets, shown = board_marks(encounter, caster, actions, aura, 1, None)
    assert (3, 2) in shown


def test_map_labels_include_name_and_hp():
    hero = creature("hero", "Fighter", "party", 18, 44, (2, 5))
    foe = creature("foe", "Goblin Warrior 1", "enemy", 15, 10, (8, 4))
    figure = board_figure(Encounter([hero, foe], width=12, height=12), {(3, 5)}, current_id="hero")
    text = " ".join(str(cell) for row in figure.data[0].text for cell in row)
    assert "Fighter" in text
    assert "44/44" in text
    assert "Goblin Warrior 1" in text
    assert "10/10" in text
    assert "P Fighter" in text
    assert "E Goblin Warrior 1" in text


def test_map_outlines_targets_and_previews_the_area():
    hero = creature("hero", "Fighter", "party", 18, 44, (2, 5))
    foe = creature("foe", "Goblin Warrior 1", "enemy", 15, 10, (8, 4))
    figure = board_figure(
        Encounter([hero, foe], width=12, height=12),
        {(3, 5)},
        current_id="hero",
        aim={(4, 4)},
        targets={(8, 4)},
        preview={(8, 4), (8, 5)},
    )
    text = " ".join(str(cell) for row in figure.data[0].text for cell in row)
    assert "P Fighter" in text
    assert "E Goblin Warrior 1" in text
    grid = figure.data[0].z
    assert grid[4][4] == 4
    assert grid[5][3] == 1
    assert grid[4][8] == 3
    assert grid[5][2] == 2
    lines = [shape.line.color for shape in figure.layout.shapes]
    fills = [shape.fillcolor for shape in figure.layout.shapes]
    assert TARGET_LINE in lines
    assert PREVIEW in fills


def test_roster_cr_column_is_numeric_for_arrow():
    import pyarrow as pa

    from src.ui import _roster_frame

    hero = creature("hero", "Fighter", "party", 18, 44, (2, 5))
    foe = creature("foe", "Goblin", "enemy", 15, 10, (8, 4), challenge_rating=0.25)
    frame = _roster_frame(Encounter([hero, foe], width=12, height=12))
    assert frame["CR"].dtype.kind == "f"
    assert frame.loc[frame["name"] == "Goblin", "CR"].iloc[0] == 0.25
    pa.Table.from_pandas(frame)


def test_battle_payload_marks_moves_and_spell_templates():
    wizard = creature(
        "wiz",
        "Wizard",
        "party",
        12,
        32,
        (2, 2),
        spells=[
            load_spell("fire-bolt"),
            load_spell("burning-hands"),
            load_spell("fireball"),
        ],
        slots={1: 4, 3: 2},
        spellcasting_mod=3,
        caster_level=5,
    )
    goblin = creature("east", "Goblin Warrior 1", "enemy", 15, 10, (4, 2))
    encounter = Encounter([wizard, goblin], width=12, height=12)
    actions = legal_actions(encounter, wizard, automated=False)
    abilities = abilities_from(actions)
    payload = battle_payload(
        encounter,
        mode="combat",
        actor=wizard,
        actions=actions,
        abilities=abilities,
        armed=None,
        slot=None,
        selected_id=wizard.id,
    )
    json.dumps(payload)
    sides = {token["side"] for token in payload["tokens"]}
    assert sides == {"party", "enemy"}
    wizard_token = next(token for token in payload["tokens"] if token["id"] == "wiz")
    goblin_token = next(token for token in payload["tokens"] if token["id"] == "east")
    assert wizard_token["initials"] == "W"
    assert wizard_token["acting"] is True
    assert goblin_token["initials"] == "GW1"
    assert goblin_token["acting"] is False
    assert [3, 2] in payload["move"]
    assert [4, 2] not in payload["move"]
    fireballs = [item for item in payload["abilities"] if item["label"] == "Fireball"]
    assert len(fireballs) == 1
    assert all(item["budget"] in {"action", "bonus"} for item in payload["abilities"])
    assert all(item["label"] != "End turn" for item in payload["abilities"])
    assert "Green squares" in payload["hint"]

    fireball = next(ability for ability in abilities if ability.label == "Fireball")
    armed = battle_payload(
        encounter,
        mode="combat",
        actor=wizard,
        actions=actions,
        abilities=abilities,
        armed=fireball,
        slot=3,
        selected_id=wizard.id,
    )
    assert armed["move"] == []
    assert armed["armed"]["slot"] == 3
    assert [4, 2] in armed["hovers"]["0,0"]
    assert any([4, 2] in cells for cells in armed["hovers"].values())
    assert "pointer" in armed["hint"]

    hands = next(ability for ability in abilities if ability.label == "Burning Hands")
    cone = battle_payload(
        encounter,
        mode="combat",
        actor=wizard,
        actions=actions,
        abilities=abilities,
        armed=hands,
        slot=hands.slots[0],
        selected_id=wizard.id,
    )
    assert facing_for_square((2, 2), (6, 2)) == (1, 0)
    assert [4, 2] in cone["hovers"]["6,2"]
    north = cone["hovers"].get("2,5")
    assert north != cone["hovers"]["6,2"]

    placed = battle_payload(
        encounter,
        mode="place",
        actor=None,
        actions=[],
        abilities=abilities,
        armed=fireball,
        slot=3,
        selected_id=wizard.id,
    )
    assert placed["abilities"] == []
    assert placed["move"] == []
    assert placed["armed"] is None
    assert wizard_token["id"] == "wiz"
    assert next(token for token in placed["tokens"] if token["id"] == "wiz")["selected"] is True
    assert "select" in placed["hint"].lower()


def test_armed_melee_names_the_adjacent_creature():
    hero = creature("hero", "Hero", "party", 16, 30, (2, 2), attacks=[sword()])
    goblin = creature("east", "Goblin Warrior 1", "enemy", 12, 10, (3, 2))
    encounter = Encounter([hero, goblin], width=12, height=12)
    actions = legal_actions(encounter, hero, automated=False)
    abilities = abilities_from(actions)
    attack = next(ability for ability in abilities if ability.label == "Sword")
    payload = battle_payload(
        encounter,
        mode="combat",
        actor=hero,
        actions=actions,
        abilities=abilities,
        armed=attack,
        slot=None,
        selected_id=hero.id,
    )
    token = next(item for item in payload["tokens"] if item["id"] == "east")
    assert token["target"] is True
    assert payload["hint"] == "Click a gold-ringed creature: Goblin Warrior 1."


def test_heuristic_takes_reckless_then_a_sheet_heal():
    hero = creature("hero", "Hero", "party", 15, 40, (0, 0), attacks=[sword()])
    hero.traits["reckless_feature"] = True
    foe = creature("foe", "Foe", "enemy", 10, 30, (1, 0))
    encounter = Encounter([hero, foe], seed=1)
    opening = heuristic_choice(encounter, hero)
    assert opening.attack_name == "Reckless Attack"
    hero.traits["reckless"] = True
    follow = heuristic_choice(encounter, hero)
    assert follow.kind == "attack"
    assert follow.attack_name == "Sword"

    hurt = creature("hurt", "Hurt", "party", 15, 40, (0, 0), attacks=[sword()])
    hurt.hp = 10
    hurt.action_available = False
    hurt.traits["bonus_heals"] = [{"name": "Warrior of the Gods", "die": "1d12"}]
    hurt.uses_left["Warrior of the Gods"] = 2
    hurt.uses_max["Warrior of the Gods"] = 4
    wounded = Encounter([hurt, foe], seed=1)
    heal = heuristic_choice(wounded, hurt)
    assert heal.attack_name == "Warrior of the Gods"


def test_nick_is_one_extra_swing_after_a_different_light_weapon():
    dagger = AttackOption(
        name="Dagger", resolution="attack", attack_bonus=5, reach_feet=5,
        damage=[{"type": "piercing", "expr": "1d4"}], light=True, finesse=True, ability="dexterity",
    )
    axe = AttackOption(
        name="Handaxe", resolution="attack", attack_bonus=5, reach_feet=5,
        damage=[{"type": "slashing", "expr": "1d6"}], mastery="nick", light=True, ability="strength",
    )
    hero = creature("hero", "Hero", "party", 15, 20, (0, 0), attacks=[dagger, axe], attacks_per_action=1)
    foe = creature("foe", "Foe", "enemy", 10, 30, (1, 0))
    encounter = Encounter([hero, foe], seed=1)
    encounter.start()
    encounter.order = ["hero", "foe"]
    encounter.turn_index = 0
    encounter.round = 1
    encounter._prepared = None
    encounter.apply(next(action for action in encounter.legal_actions() if action.attack_name == "Dagger"), "test")
    assert hero.action_available is True
    assert hero.attacks_made == 1
    nick = [action for action in encounter.legal_actions() if action.kind == "attack"]
    assert {action.attack_name for action in nick} == {"Handaxe"}
    encounter.apply(nick[0], "test")
    assert hero.attacks_made == 1
    assert hero.action_available is False


def test_seed_changes_later_rolls_and_keeps_the_log():
    import random

    hero = creature("hero", "Hero", "party", 15, 30, (0, 0), attacks=[sword()])
    foe = creature("foe", "Foe", "enemy", 12, 20, (1, 0))
    encounter = Encounter([hero, foe], seed=1)
    assert encounter.log == []
    encounter.use_seed(4)
    assert encounter.seed == 4
    assert encounter.log == []
    assert encounter.rng_seed == 1
    encounter.start()
    assert encounter.log[0] == "Dice seed 4."
    assert encounter.log[1].startswith("Initiative:")
    assert encounter.rng_seed == 4
    initiative = encounter.log[1]
    hit_points = [body.hp for body in encounter.combatants]

    seen = encounter.rng.randint(1, 20)
    replay = random.Random(4)
    for _ in encounter.combatants:
        replay.randint(1, 20)
    assert seen == replay.randint(1, 20)
    before = list(encounter.log)
    encounter.use_seed(4)
    assert encounter.log == before
    assert encounter.rng.randint(1, 20) == replay.randint(1, 20)

    encounter.use_seed(2)
    assert encounter.log[0] == "Dice seed 4."
    assert encounter.log[1] == initiative
    assert [body.hp for body in encounter.combatants] == hit_points
    note = "Dice seed is now 2. Rolls from here use that seed."
    assert [line for line in encounter.log if line.startswith("Dice seed is now")] == [note]
    fresh = random.Random(2)
    assert encounter.rng.randint(1, 1_000_000_000) == fresh.randint(1, 1_000_000_000)
    encounter.use_seed(2)
    assert [line for line in encounter.log if line.startswith("Dice seed is now")] == [note]
    continued = [fresh.randint(1, 1_000_000_000) for _ in range(4)]
    restarted = [random.Random(2).randint(1, 1_000_000_000) for _ in range(4)]
    actual = [encounter.rng.randint(1, 1_000_000_000) for _ in range(4)]
    assert actual == continued
    assert actual != restarted


def test_starting_seeds_change_initiative():
    def fight(seed):
        hero = creature("hero", "Hero", "party", 15, 30, (0, 0), attacks=[sword()])
        foe = creature("foe", "Foe", "enemy", 12, 20, (1, 0))
        encounter = Encounter([hero, foe], seed=seed)
        encounter.start()
        return encounter

    one = fight(1)
    two = fight(2)
    assert one.log[0] == "Dice seed 1."
    assert two.log[0] == "Dice seed 2."
    assert one.log[1].startswith("Initiative:")
    assert two.log[1].startswith("Initiative:")
    assert one.log[1] != two.log[1]


def test_a_large_body_fills_its_squares_and_threatens_the_next_one():
    from src.ui import _overlap

    brute = creature("brute", "Ogre", "enemy", 12, 40, (0, 0), attacks=[sword()], size="Large")
    near = creature("near", "Near", "party", 14, 20, (2, 0), attacks=[sword()])
    far = creature("far", "Far", "party", 14, 20, (3, 0), attacks=[sword()])
    assert brute.span == 2
    assert brute.squares() == {(0, 0), (1, 0), (0, 1), (1, 1)}
    assert creature("tiny", "Tiny", "party", 10, 4, (0, 0), size="Tiny").span == 1
    assert creature("huge", "Huge", "enemy", 18, 80, (0, 0), size="Huge").span == 3
    assert creature("titan", "Titan", "enemy", 20, 100, (0, 0), size="Gargantuan").span == 4
    assert weapon_reaches(brute, near, sword())
    assert not weapon_reaches(brute, far, sword())
    reach = AttackOption(
        name="Glaive",
        resolution="attack",
        attack_bonus=4,
        reach_feet=10,
        damage=[{"type": "slashing", "expr": "1d10"}],
    )
    assert weapon_reaches(brute, far, reach)
    inside = creature("inside", "Inside", "party", 10, 10, (1, 1))
    assert _overlap(Encounter([brute, inside], width=8, height=8))
    assert not _overlap(Encounter([brute, near], width=8, height=8))


def test_a_sphere_hits_a_huge_creature_through_one_square():
    from src.grid import area_squares

    dragon = creature("dragon", "Dragon", "enemy", 18, 80, (0, 0), size="Huge")
    wizard = creature(
        "wiz",
        "Wizard",
        "party",
        12,
        20,
        (8, 2),
        spells=[load_spell("fireball")],
        slots={3: 1},
        spell_save_dc=14,
        caster_level=5,
    )
    cells = set(area_squares((6, 2), "sphere", 20, None))
    assert (2, 2) in cells
    assert (1, 2) not in cells
    assert dragon.in_cells(cells)
    encounter = Encounter([wizard, dragon], width=12, height=12, seed=1)
    _prepare(encounter, ["wiz", "dragon"], [])
    blast = next(action for action in encounter.legal_actions(wizard) if action.aim == (6, 2))
    assert dragon in targets_for(encounter, wizard, blast)


def test_a_large_move_keeps_the_body_off_an_enemy_and_costs_five_feet():
    brute = creature("brute", "Ogre", "enemy", 12, 40, (0, 0), speed=30, size="Large")
    foe = creature("foe", "Foe", "party", 14, 20, (2, 0), speed=30)
    encounter = Encounter([brute, foe], width=8, height=8, seed=1)
    _prepare(encounter, ["brute", "foe"], [])
    destinations = {action.destination for action in encounter.legal_actions(brute) if action.kind == "move"}
    assert (1, 0) not in destinations
    assert (0, 1) in destinations
    step = next(action for action in encounter.legal_actions(brute) if action.destination == (0, 1))
    encounter.apply(step, "test")
    assert brute.position == (0, 1)
    assert brute.squares() == {(0, 1), (1, 1), (0, 2), (1, 2)}
    assert brute.movement_left == 25


def test_a_2024_emanation_is_a_self_centered_sphere():
    guardians = cache_spell("spirit-guardians")
    assert guardians.kind == "save"
    assert guardians.activation == "action"
    assert guardians.ongoing is True
    assert guardians.area_type == "sphere"
    assert guardians.area_size == 15
    assert guardians.range_mode == "self"
    assert guardians.dc_ability == "wisdom"
    assert guardians.dc_success == "half"
    assert spell_damage_parts(guardians, 3, 5) == [("radiant", "3d8")]
    assert spell_damage_parts(guardians, 4, 5) == [("radiant", "4d8")]

    woods = cache_spell("conjure-woodland-beings")
    assert woods.kind == "save"
    assert woods.ongoing is True
    assert woods.area_type == "sphere"
    assert woods.area_size == 10
    assert woods.range_mode == "self"

    darkness = cache_spell("darkness")
    assert darkness.area_type == "sphere"
    assert darkness.area_size == 15
    daylight = cache_spell("daylight")
    assert daylight.area_type == "sphere"
    assert daylight.area_size == 60
    globe = cache_spell("globe-of-invulnerability")
    assert globe.area_type is None
    assert globe.kind == "excluded"

    weapon = cache_spell("spiritual-weapon")
    assert weapon.kind == "attack"
    assert weapon.activation == "bonus"
    assert weapon.attack_type == "melee"
    assert weapon.concentration is True
    assert weapon.ongoing is False
    assert weapon.range_feet == 60
    assert spell_damage_parts(weapon, 2, 5) == [("force", "1d8+MOD")]
    assert spell_damage_parts(weapon, 3, 5) == [("force", "2d8+MOD")]


def test_spirit_guardians_is_castable_and_skips_the_casters_side():
    spell = cache_spell("spirit-guardians")
    caster = creature(
        "cleric",
        "Cleric",
        "party",
        16,
        40,
        (0, 0),
        spells=[spell],
        slots={3: 1},
        spell_save_dc=15,
    )
    ally = creature("ally", "Ally", "party", 12, 30, (0, 1))
    foe = creature(
        "foe",
        "Foe",
        "enemy",
        12,
        40,
        (1, 0),
        attacks=[
            AttackOption(
                name="Claw",
                resolution="attack",
                attack_bonus=10,
                reach_feet=5,
                damage=[{"type": "slashing", "expr": "1d6"}],
            )
        ],
    )
    far = creature("far", "Far", "enemy", 12, 20, (4, 0))
    wide = creature("wide", "Wide", "enemy", 12, 20, (0, 4))
    outside = Encounter(
        [
            creature("cleric", "Cleric", "party", 16, 40, (0, 0), spells=[spell], slots={3: 1}, spell_save_dc=15),
            creature("ally", "Ally", "party", 12, 30, (1, 0)),
            creature("far", "Far", "enemy", 12, 20, (4, 0)),
        ]
    )
    _prepare(outside, ["cleric", "ally", "far"], [])
    assert all(action.spell_index != "spirit-guardians" for action in outside.legal_actions(outside.by_id("cleric")))
    edge = Encounter(
        [
            creature("cleric", "Cleric", "party", 16, 40, (0, 0), spells=[spell], slots={3: 1}, spell_save_dc=15),
            creature("foe", "Foe", "enemy", 12, 20, (3, 0)),
        ]
    )
    _prepare(edge, ["cleric", "foe"], [])
    assert any(action.spell_index == "spirit-guardians" for action in edge.legal_actions(edge.by_id("cleric")))

    encounter = Encounter([caster, ally, foe, far, wide])
    # Each failed Wisdom save is 1, then 3d8. The claw is a hit, its damage, and a failed concentration save.
    _prepare(
        encounter,
        ["cleric", "foe", "ally", "far", "wide"],
        [1, 4, 4, 4, 1, 4, 4, 4, 1, 4, 4, 4, 1, 4, 4, 4, 10, 6, 1],
    )
    cast = next(action for action in encounter.legal_actions(caster) if action.spell_index == "spirit-guardians")
    assert cast.budget == "action"
    assert foe in targets_for(encounter, caster, cast)
    assert ally not in targets_for(encounter, caster, cast)
    assert caster not in targets_for(encounter, caster, cast)
    assert far not in targets_for(encounter, caster, cast)
    assert wide not in targets_for(encounter, caster, cast)
    encounter.apply(cast, "test")
    assert caster.slots[3] == 0
    assert caster.concentrating_on == "spirit-guardians"
    assert foe.hp == 28
    assert ally.hp == 30
    assert far.hp == 20
    assert wide.hp == 20
    assert caster.speed == 30
    assert foe.speed == 30
    caster.traits["emanation_saved"] = json.loads(json.dumps(caster.traits["emanation_saved"]))
    opened = {tuple(cell) for cell in _mat(encounter, caster)["aura"]}
    assert (0, 0) in opened
    assert (1, 0) in opened
    assert (4, 0) not in opened
    assert (0, 4) not in opened
    assert _mat(encounter, caster)["preview"] == []

    finish_turn(encounter)
    assert foe.hp == 28
    finish_turn(encounter)
    assert foe.hp == 16
    assert ally.hp == 30
    assert any("Spirit Guardians (emanation)" in line for line in encounter.log)
    finish_turn(encounter)
    assert foe.hp == 16

    step_in = next(action for action in encounter.legal_actions(far) if action.destination == (3, 0))
    encounter.apply(step_in, "test")
    assert far.position == (3, 0)
    assert far.hp == 8
    finish_turn(encounter)
    assert far.hp == 8
    finish_turn(encounter)
    assert wide.hp == 20

    encounter.begin_turn()
    assert foe.hp == 16
    assert far.hp == 8
    assert wide.hp == 20
    assert not any("(ongoing)" in line for line in encounter.log)

    step = next(action for action in encounter.legal_actions(caster) if action.destination == (1, 1))
    encounter.apply(step, "test")
    assert caster.position == (1, 1)
    assert wide.hp == 8
    assert foe.hp == 16
    assert far.hp == 8
    assert ally.hp == 30
    followed = {tuple(cell) for cell in _mat(encounter, caster)["aura"]}
    assert (1, 1) in followed
    assert (0, 4) in followed
    assert (4, 0) in followed
    assert (4, 0) not in opened
    assert encounter.damage_dealt == {"party": 48, "enemy": 0}

    finish_turn(encounter)
    assert wide.hp == 8
    claw = next(
        action
        for action in encounter.legal_actions(foe)
        if action.attack_name == "Claw" and action.target_id == caster.id
    )
    encounter.apply(claw, "test")
    assert caster.concentrating_on is None
    assert any("loses spirit-guardians" in line for line in encounter.log)
    held = (foe.hp, far.hp, wide.hp)
    finish_turn(encounter)
    finish_turn(encounter)
    finish_turn(encounter)
    encounter.begin_turn()
    assert (foe.hp, far.hp, wide.hp) == held
    assert caster.speed == 30
    assert foe.speed == 30


def test_a_push_into_spirit_guardians_forces_a_save():
    spell = cache_spell("spirit-guardians")
    caster = creature(
        "cleric",
        "Cleric",
        "party",
        16,
        40,
        (0, 0),
        spells=[spell],
        spell_save_dc=15,
    )
    foe = creature("foe", "Foe", "enemy", 12, 40, (4, 0))
    pusher = creature("push", "Push", "party", 12, 20, (5, 0))
    pusher.traits["telekinetic"] = 15
    encounter = Encounter([pusher, caster, foe])
    _prepare(encounter, ["push", "foe", "cleric"], [1, 1, 4, 4, 4])
    caster.concentrating_on = "spirit-guardians"
    caster.ongoing_slot = 3
    shove = next(
        action
        for action in encounter.legal_actions(pusher)
        if action.attack_name == "Telekinetic Shove" and action.target_id == foe.id
    )
    encounter.apply(shove, "test")
    assert foe.position == (3, 0)
    assert foe.hp == 28
    assert caster.hp == 40
    finish_turn(encounter)
    assert foe.hp == 28


def _seat(encounter: Encounter, order: list[str]) -> None:
    """Start the fight, then force a turn order. Later rolls keep the encounter seed."""
    encounter.start()
    encounter.order = list(order)
    encounter.turn_index = 0
    encounter.round = 1
    encounter._prepared = None


def _emanation_lines(encounter: Encounter) -> list[str]:
    return [line for line in encounter.log if "Spirit Guardians (emanation)" in line]


def test_spirit_guardians_on_a_20_by_20_mat_damages_the_whole_emanation():
    spell = cache_spell("spirit-guardians")
    cleric = creature(
        "cleric",
        "Cleric",
        "party",
        17,
        80,
        (10, 10),
        spells=[spell],
        slots={3: 1},
        spell_save_dc=14,
        traits={"war_caster": True},
    )
    near = creature("near", "Near", "enemy", 12, 80, (9, 9))
    mid = creature("mid", "Mid", "enemy", 12, 80, (12, 9))
    edge = creature("edge", "Edge", "enemy", 12, 80, (13, 11))
    bystander = creature("bystander", "Bystander", "enemy", 12, 80, (14, 10))
    wide = creature("wide", "Wide", "enemy", 12, 80, (10, 14))
    encounter = Encounter([cleric, near, mid, edge, bystander, wide], width=20, height=20)
    _seat(encounter, ["cleric", "bystander", "edge", "near", "mid", "wide"])

    cast = next(action for action in encounter.legal_actions(cleric) if action.spell_index == "spirit-guardians")
    encounter.apply(cast, "test")
    assert near.hp < 80
    assert mid.hp < 80
    assert edge.hp < 80
    assert bystander.hp == 80
    assert wide.hp == 80
    assert cleric.hp == 80
    assert _emanation_lines(encounter) == []
    aura = {tuple(cell) for cell in _mat(encounter, cleric)["aura"]}
    assert (10, 10) in aura
    assert (13, 11) in aura
    assert (14, 10) not in aura
    assert (10, 14) not in aura
    assert len(aura) == 49
    held = (near.hp, mid.hp, edge.hp)

    cover = next(action for action in encounter.legal_actions(cleric) if action.destination == (10, 11))
    encounter.apply(cover, "test")
    assert cleric.position == (10, 11)
    assert wide.hp < 80
    assert (near.hp, mid.hp, edge.hp) == held
    assert bystander.hp == 80
    assert len(_emanation_lines(encounter)) == 1

    finish_turn(encounter)
    assert edge.hp == held[2]

    inward = next(action for action in encounter.legal_actions(bystander) if action.destination == (13, 10))
    encounter.apply(inward, "test")
    assert bystander.position == (13, 10)
    assert bystander.hp < 80
    assert edge.hp == held[2]
    assert len(_emanation_lines(encounter)) == 2

    finish_turn(encounter)
    assert edge.hp == held[2]
    finish_turn(encounter)
    assert edge.hp < held[2]
    assert len(_emanation_lines(encounter)) == 3

    hammer = AttackOption(
        name="Light Hammer",
        resolution="attack",
        attack_bonus=4,
        reach_feet=10,
        damage=[{"type": "bludgeoning", "expr": "1d4+2"}],
    )
    caster = creature(
        "cleric",
        "Cleric",
        "party",
        17,
        80,
        (10, 10),
        spells=[spell],
        slots={3: 1},
        spell_save_dc=14,
        traits={"war_caster": True},
    )
    bugbear = creature("bugbear", "Bugbear", "enemy", 14, 40, (12, 10), attacks=[hammer])
    outsider = creature("outsider", "Outsider", "enemy", 12, 80, (10, 14))
    paused = Encounter([caster, bugbear, outsider], width=20, height=20)
    _seat(paused, ["cleric", "bugbear", "outsider"])
    opening = next(action for action in paused.legal_actions(caster) if action.spell_index == "spirit-guardians")
    paused.apply(opening, "test")
    assert 0 < bugbear.hp < 40
    assert outsider.hp == 80
    reach = next(action for action in paused.legal_actions(caster) if action.destination == (10, 13))
    paused.apply(reach, "test")
    assert paused.pending is not None
    assert paused.pending["phase"] == "move"
    assert caster.position == (10, 12)
    assert outsider.hp == 80
    paused.answer_reaction(None)
    assert paused.pending is None
    assert caster.position == (10, 13)
    assert outsider.hp < 80
    assert _emanation_lines(paused)



    spell = cache_spell("spiritual-weapon")
    caster = creature(
        "cleric",
        "Cleric",
        "party",
        16,
        40,
        (0, 0),
        spells=[spell],
        slots={2: 2},
        spell_attack_bonus=6,
        spellcasting_mod=3,
    )
    foe = creature("foe", "Foe", "enemy", 10, 40, (2, 0))
    encounter = Encounter([caster, foe], width=14, height=2)
    _prepare(encounter, ["cleric", "foe"], [10, 4, 10, 4])
    player = legal_actions(encounter, caster, automated=False)
    auto = legal_actions(encounter, caster, automated=True)
    player_aims = {action.aim for action in player if action.spell_index == "spiritual-weapon"}
    auto_aims = {action.aim for action in auto if action.spell_index == "spiritual-weapon"}
    assert (1, 0) in player_aims
    assert (12, 0) in player_aims
    assert (13, 0) not in player_aims
    assert (0, 0) not in player_aims
    assert (2, 0) not in player_aims
    assert (1, 0) in auto_aims
    assert (12, 0) not in auto_aims
    assert all(action.target_id is None for action in player if action.spell_index == "spiritual-weapon")
    armed = next(item for item in abilities_from(player) if item.spell_index == "spiritual-weapon")
    assert armed.needs == "square"
    assert armed.budget == "bonus"

    place = next(action for action in player if action.spell_index == "spiritual-weapon" and action.aim == (1, 0))
    encounter.apply(place, "test")
    assert caster.slots[2] == 1
    assert caster.bonus_available is False
    assert caster.action_available is True
    assert caster.concentrating_on == "spiritual-weapon"
    assert caster.traits["spiritual_weapon"] == {"x": 1, "y": 0, "slot": 2}
    assert foe.hp == 33
    assert "summons a Spiritual Weapon at (1, 0)" in " ".join(encounter.log)
    assert all(creature.name != "Spiritual Weapon" for creature in encounter.combatants)
    assert any(action.destination == (1, 0) for action in encounter.legal_actions(caster) if action.kind == "move")
    markers = [token for token in _mat(encounter, caster)["tokens"] if token.get("marker")]
    assert len(markers) == 1
    assert markers[0]["name"] == "Spiritual Weapon"
    assert markers[0]["x"] == 1
    assert markers[0]["y"] == 0
    assert markers[0]["defeated"] is False
    assert markers[0]["target"] is False

    finish_turn(encounter)
    finish_turn(encounter)
    encounter.begin_turn()
    again = encounter.legal_actions(caster)
    labels = [item.label for item in abilities_from(again)]
    assert "Move Spiritual Weapon" in labels
    assert "Spiritual Weapon" in labels
    repeat = next(
        action
        for action in again
        if action.kind == "feature" and action.attack_name == "Spiritual Weapon" and action.aim == (3, 0)
    )
    stayed = caster.slots[2]
    encounter.apply(repeat, "test")
    assert caster.slots[2] == stayed
    assert caster.traits.get("leveled_cast") is None
    assert caster.bonus_available is False
    assert caster.traits["spiritual_weapon"] == {"x": 3, "y": 0, "slot": 2}
    assert foe.hp == 26
    assert "moves the Spiritual Weapon to (3, 0)" in " ".join(encounter.log)
    moved = [token for token in _mat(encounter, caster)["tokens"] if token.get("marker")]
    assert len(moved) == 1
    assert moved[0]["x"] == 3


def test_recasting_spiritual_weapon_replaces_the_token():
    spell = cache_spell("spiritual-weapon")
    guardians = cache_spell("spirit-guardians")
    caster = creature(
        "cleric",
        "Cleric",
        "party",
        16,
        40,
        (0, 0),
        spells=[spell, guardians],
        slots={2: 2, 3: 1},
        spell_attack_bonus=6,
        spellcasting_mod=3,
        spell_save_dc=15,
    )
    foe = creature("foe", "Foe", "enemy", 10, 40, (2, 0))
    encounter = Encounter([caster, foe])
    _prepare(encounter, ["cleric", "foe"], [10, 4, 10, 4, 1, 4, 4, 4])
    first = next(
        action
        for action in encounter.legal_actions(caster)
        if action.spell_index == "spiritual-weapon" and action.aim == (1, 0)
    )
    encounter.apply(first, "test")
    finish_turn(encounter)
    finish_turn(encounter)
    encounter.begin_turn()
    second = next(
        action
        for action in encounter.legal_actions(caster)
        if action.spell_index == "spiritual-weapon" and action.aim == (3, 0)
    )
    encounter.apply(second, "test")
    assert caster.slots[2] == 0
    assert caster.traits["spiritual_weapon"] == {"x": 3, "y": 0, "slot": 2}
    assert len([token for token in _mat(encounter, caster)["tokens"] if token.get("marker")]) == 1
    finish_turn(encounter)
    finish_turn(encounter)
    encounter.begin_turn()
    aura = next(action for action in encounter.legal_actions(caster) if action.spell_index == "spirit-guardians")
    encounter.apply(aura, "test")
    assert caster.concentrating_on == "spirit-guardians"
    assert "spiritual_weapon" not in caster.traits
    assert [token for token in _mat(encounter, caster)["tokens"] if token.get("marker")] == []


def test_breaking_concentration_removes_the_spiritual_weapon():
    spell = cache_spell("spiritual-weapon")
    caster = creature(
        "cleric",
        "Cleric",
        "party",
        16,
        40,
        (0, 0),
        spells=[spell],
        slots={2: 1},
        spell_attack_bonus=6,
        spellcasting_mod=3,
    )
    foe = creature(
        "foe",
        "Foe",
        "enemy",
        10,
        40,
        (1, 0),
        attacks=[
            AttackOption(
                name="Claw",
                resolution="attack",
                attack_bonus=6,
                reach_feet=5,
                damage=[{"type": "slashing", "expr": "1d6"}],
            )
        ],
    )
    encounter = Encounter([caster, foe])
    _prepare(encounter, ["cleric", "foe"], [10, 4, 10, 4, 1])
    place = next(
        action
        for action in encounter.legal_actions(caster)
        if action.spell_index == "spiritual-weapon" and action.aim == (2, 0)
    )
    encounter.apply(place, "test")
    assert foe.hp == 33
    assert caster.traits["spiritual_weapon"]["x"] == 2
    finish_turn(encounter)
    encounter.apply(next(action for action in encounter.legal_actions(foe) if action.attack_name == "Claw"), "test")
    assert caster.concentrating_on is None
    assert "spiritual_weapon" not in caster.traits
    assert any("loses spiritual-weapon" in line for line in encounter.log)
    assert [token for token in _mat(encounter, caster)["tokens"] if token.get("marker")] == []


def test_spiritual_weapon_can_be_placed_with_nobody_in_reach():
    spell = cache_spell("spiritual-weapon")
    caster = creature(
        "cleric",
        "Cleric",
        "party",
        16,
        40,
        (0, 0),
        spells=[spell],
        slots={2: 1},
        spell_attack_bonus=6,
        spellcasting_mod=3,
    )
    near = creature("gob-a", "Goblin A", "enemy", 10, 20, (2, 1))
    far = creature("gob-b", "Goblin B", "enemy", 10, 20, (2, 0))
    encounter = Encounter([caster, near, far])
    _prepare(encounter, ["cleric", "gob-a", "gob-b"], [10, 4])
    empty = next(
        action
        for action in encounter.legal_actions(caster)
        if action.spell_index == "spiritual-weapon" and action.aim == (4, 4)
    )
    encounter.apply(empty, "test")
    assert near.hp == 20
    assert far.hp == 20
    assert caster.traits["spiritual_weapon"] == {"x": 4, "y": 4, "slot": 2}
    assert "summons a Spiritual Weapon at (4, 4)" in " ".join(encounter.log)


def test_spiritual_weapon_hits_the_nearer_creature_with_the_lower_id():
    spell = cache_spell("spiritual-weapon")
    caster = creature(
        "cleric",
        "Cleric",
        "party",
        16,
        40,
        (0, 0),
        spells=[spell],
        slots={2: 1},
        spell_attack_bonus=6,
        spellcasting_mod=3,
    )
    first = creature("gob-a", "Goblin A", "enemy", 10, 20, (2, 1))
    second = creature("gob-b", "Goblin B", "enemy", 10, 20, (2, 0))
    encounter = Encounter([caster, second, first])
    _prepare(encounter, ["cleric", "gob-b", "gob-a"], [10, 4])
    place = next(
        action
        for action in encounter.legal_actions(caster)
        if action.spell_index == "spiritual-weapon" and action.aim == (1, 0)
    )
    encounter.apply(place, "test")
    assert first.hp == 13
    assert second.hp == 20


def test_the_heuristic_uses_the_spiritual_weapon_it_already_has():
    spell = cache_spell("spiritual-weapon")
    caster = creature(
        "cleric",
        "Cleric",
        "party",
        16,
        40,
        (0, 0),
        spells=[spell],
        slots={2: 1},
        spell_attack_bonus=6,
        spellcasting_mod=3,
    )
    foe = creature("foe", "Foe", "enemy", 10, 40, (2, 0))
    encounter = Encounter([caster, foe])
    _prepare(encounter, ["cleric", "foe"], [])
    caster.action_available = False
    caster.movement_left = 0
    fresh = heuristic_choice(encounter, caster)
    assert fresh.spell_index == "spiritual-weapon"
    assert fresh.aim == (1, 0)

    caster.traits["spiritual_weapon"] = {"x": 1, "y": 0, "slot": 2}
    caster.concentrating_on = "spiritual-weapon"
    caster.ongoing_slot = 2
    held = heuristic_choice(encounter, caster)
    assert held.kind == "feature"
    assert held.attack_name == "Spiritual Weapon"
    assert held.aim == (1, 0)
    prompt = render_prompt(encounter, caster, encounter.legal_actions(caster))
    decoded = decode(prompt)
    assert decoded["actor"]["weapon_x"] == 1
    assert decoded["actor"]["weapon_y"] == 0


def test_custom_weapon_damage_must_be_dice():
    from src.ui import _append_custom

    encounter = Encounter([], width=12, height=12)
    try:
        _append_custom(encounter, None, "Hero", "party", 16, 20, 30, 5, "a lot", [], 13, 5, 5, "")
        raised = False
    except ValueError as exc:
        raised = True
        assert "1d8+3" in str(exc)
    assert raised
    assert encounter.combatants == []


def _zone_cells(encounter, actor):
    return {tuple(cell) for cell in _mat(encounter, actor)["aura"]}


def test_spike_growth_persists_and_damages_each_square_moved():
    spell = cache_spell("spike-growth")
    assert spell.kind == "zone"
    assert spell.ongoing is False
    assert spell.area_type == "sphere"
    assert spell.area_size == 20
    caster = creature(
        "druid", "Druid", "party", 14, 30, (0, 0),
        spells=[spell], slots={2: 1}, spell_save_dc=13,
    )
    victim = creature("victim", "Victim", "enemy", 12, 40, (6, 5))
    walker = creature("walker", "Walker", "enemy", 12, 40, (5, 0), speed=30)
    encounter = Encounter([caster, walker, victim], width=12, height=12, seed=1)
    _prepare(encounter, ["druid", "walker", "victim"], [1, 1, 1, 1, 1, 1])
    cast = next(
        action for action in encounter.legal_actions(caster)
        if action.spell_index == "spike-growth" and action.aim == (5, 5)
    )
    encounter.apply(cast, "test")
    assert caster.concentrating_on == "spike-growth"
    assert (caster.traits["zone"]["x"], caster.traits["zone"]["y"]) == (5, 5)
    assert victim.hp == 40
    assert walker.hp == 40
    finish_turn(encounter)
    painted = _zone_cells(encounter, walker)
    assert (5, 5) in painted
    assert (5, 1) in painted
    assert (6, 5) in painted
    assert (5, 0) not in painted
    step = next(action for action in encounter.legal_actions(walker) if action.destination == (5, 3))
    encounter.apply(step, "test")
    assert walker.position == (5, 3)
    assert walker.movement_left == 0
    assert walker.hp == 34
    assert encounter.damage_dealt["party"] == 6
    assert len([line for line in encounter.log if "automatically" in line]) == 3
    finish_turn(encounter)
    encounter.apply(next(action for action in encounter.legal_actions(victim) if action.kind == "end"), "test")
    assert victim.hp == 40
    assert walker.hp == 34
    assert len([line for line in encounter.log if "automatically" in line]) == 3
    still = _zone_cells(encounter, victim)
    assert (5, 1) in still and (5, 5) in still
    from src.rules import apply_damage

    apply_damage(caster, [("slashing", 100)], encounter.rng)
    assert caster.hp == 0
    assert caster.concentrating_on is None
    assert "zone" not in caster.traits
    assert _zone_cells(encounter, victim) == set()


def test_insect_plague_stays_where_it_was_cast():
    from src.grid import sphere

    spell = cache_spell("insect-plague")
    assert spell.kind == "zone"
    assert spell.ongoing is False
    assert spell.area_type == "sphere"
    assert spell.area_size == 20
    caster = creature(
        "druid", "Druid", "party", 14, 40, (0, 0),
        spells=[spell], slots={5: 1}, spell_save_dc=15,
    )
    foe = creature("foe", "Foe", "enemy", 12, 80, (8, 8))
    encounter = Encounter([caster, foe], width=12, height=12, seed=1)
    _prepare(encounter, ["druid", "foe"], [20, 1, 1, 1, 1])
    cast = next(
        action for action in encounter.legal_actions(caster)
        if action.spell_index == "insect-plague" and action.aim == (8, 8)
    )
    encounter.apply(cast, "test")
    step = next(action for action in encounter.legal_actions(caster) if action.destination == (0, 1))
    encounter.apply(step, "test")
    assert caster.position == (0, 1)
    assert (caster.traits["zone"]["x"], caster.traits["zone"]["y"]) == (8, 8)
    aura = _zone_cells(encounter, caster)
    board = {(x, y) for x in range(12) for y in range(12)}
    assert aura == sphere((8, 8), 20) & board
    assert (0, 1) not in aura
    assert (8, 8) in aura


def test_moonbeam_is_a_narrow_cylinder_and_flaming_sphere_is_one_square():
    from src.grid import sphere
    from src.zones import cells_of, zone_spec

    moon = cache_spell("moonbeam")
    assert moon.kind == "zone"
    assert moon.ongoing is False
    assert moon.area_type == "cylinder"
    assert moon.area_size == 5
    assert cells_of(zone_spec("moonbeam"), {"x": 4, "y": 4}) == sphere((4, 4), 5)

    ball = cache_spell("flaming-sphere")
    assert ball.kind == "zone"
    assert ball.ongoing is False
    assert ball.area_type is None
    assert ball.area_size is None
    assert cells_of(zone_spec("flaming-sphere"), {"x": 2, "y": 2}) == {(2, 2)}


def test_entangle_restrains_without_a_damage_roll():
    spell = cache_spell("entangle")
    assert spell.kind == "zone"
    assert spell.dc_ability == "strength"
    assert spell.dc_success == "none"
    caster = creature(
        "druid", "Druid", "party", 14, 30, (0, 0),
        spells=[spell], slots={1: 1}, spell_save_dc=15,
    )
    foe = creature("foe", "Foe", "enemy", 12, 20, (4, 0))
    encounter = Encounter([caster, foe], width=12, height=12, seed=1)
    _prepare(encounter, ["druid", "foe"], [1])
    cast = next(
        action for action in encounter.legal_actions(caster)
        if action.spell_index == "entangle" and action.aim == (4, 0)
    )
    encounter.apply(cast, "test")
    assert "restrained" in foe.conditions
    text = " ".join(encounter.log)
    assert "Failed save" in text
    assert "restrained" in text.lower()
    assert not any(token in text for token in ("d4", "d6", "d8", "d10", "d12"))


def test_web_restrains_on_the_creatures_turn_and_does_not_burn():
    spell = cache_spell("web")
    assert spell.kind == "zone"
    assert spell.ongoing is False
    caster = creature(
        "wizard", "Wizard", "party", 14, 30, (0, 0),
        spells=[spell], slots={2: 1}, spell_save_dc=15,
    )
    foe = creature("foe", "Foe", "enemy", 12, 20, (4, 0))
    encounter = Encounter([caster, foe], width=12, height=12, seed=1)
    _prepare(encounter, ["wizard", "foe"], [1])
    cast = next(
        action for action in encounter.legal_actions(caster)
        if action.spell_index == "web" and action.aim == (4, 0)
    )
    encounter.apply(cast, "test")
    assert "restrained" not in foe.conditions
    assert "Failed save" not in " ".join(encounter.log)
    finish_turn(encounter)
    encounter.apply(next(action for action in encounter.legal_actions(foe) if action.kind == "end"), "test")
    assert "restrained" in foe.conditions
    text = " ".join(encounter.log).lower()
    assert "2d4" not in text
    assert "fire" not in text


def test_faerie_fire_outlines_and_grants_advantage_after_the_creature_leaves():
    from src.rules import attack_mode

    spell = cache_spell("faerie-fire")
    assert spell.kind == "zone"
    assert spell.area_type == "cube"
    assert spell.area_size == 20
    caster = creature(
        "druid", "Druid", "party", 14, 30, (0, 0),
        spells=[spell], slots={1: 1}, spell_save_dc=15,
    )
    foe = creature("foe", "Foe", "enemy", 12, 20, (4, 0), speed=30)
    encounter = Encounter([caster, foe], width=12, height=12, seed=1)
    _prepare(encounter, ["druid", "foe"], [1])
    cast = next(
        action for action in encounter.legal_actions(caster)
        if action.spell_index == "faerie-fire" and action.aim == (4, 0)
    )
    encounter.apply(cast, "test")
    assert "outlined" in foe.conditions
    assert attack_mode(caster, foe) == 1
    finish_turn(encounter)
    step = next(action for action in encounter.legal_actions(foe) if action.destination == (0, 4))
    encounter.apply(step, "test")
    assert foe.position == (0, 4)
    assert "outlined" in foe.conditions
    assert attack_mode(caster, foe) == 1


def test_flame_blade_and_shining_smite_are_not_auras():
    blade = cache_spell("flame-blade")
    smite = cache_spell("shining-smite")
    assert blade.ongoing is False
    assert blade.area_type is None
    assert smite.ongoing is False
    assert smite.area_type is None


def _monster(name: str):
    payload = json.loads((CACHE / "monsters" / f"{name}.json").read_text(encoding="utf-8"))
    return classify_monster(payload)


def _spell_action(encounter, actor, index: str, **wanted):
    for action in encounter.legal_actions(actor):
        if action.spell_index != index:
            continue
        if any(getattr(action, key) != value for key, value in wanted.items()):
            continue
        return action
    raise AssertionError(index)


def test_hold_person_paralyzes_a_humanoid_and_clears_when_concentration_ends():
    from src.rules import attack_mode

    hold = cache_spell("hold-person")
    brave = cache_spell("heroism")
    assert hold.kind == "afflict"
    caster = creature(
        "wizard", "Wizard", "party", 14, 30, (0, 0),
        spells=[hold, brave], slots={1: 1, 2: 1}, spell_save_dc=15,
    )
    foe = creature("foe", "Foe", "enemy", 12, 20, (1, 0), attacks=[sword()])
    undead = creature("bones", "Bones", "enemy", 12, 20, (1, 0), creature_type="Undead")
    blocked = Encounter([caster, undead], width=12, height=12, seed=1)
    _prepare(blocked, ["wizard", "bones"], [])
    assert all(action.spell_index != "hold-person" for action in blocked.legal_actions(caster))
    encounter = Encounter([caster, foe], width=12, height=12, seed=1)
    _prepare(encounter, ["wizard", "foe"], [1, 1])
    offered = [action.target_id for action in encounter.legal_actions(caster) if action.spell_index == "hold-person"]
    assert offered == ["foe"]
    cast = _spell_action(encounter, caster, "hold-person", target_id="foe")
    assert action_score(encounter, caster, cast) > 3
    encounter.apply(cast, "test")
    assert encounter.current().id == "wizard"
    assert "paralyzed" in foe.conditions
    assert encounter.rng.rolls == [1]
    text = " ".join(encounter.log)
    assert "Failed save" in text
    assert not any(token in text for token in ("d4", "d6", "d8", "d10", "d12"))
    finish_turn(encounter)
    assert encounter.current().id == "foe"
    assert foe.movement_left is None
    encounter.begin_turn()
    assert all(action.kind != "move" for action in encounter.legal_actions(foe))
    finish_turn(encounter)
    assert "paralyzed" in foe.conditions
    assert not encounter.rng.rolls
    encounter.begin_turn()
    hero = _spell_action(encounter, caster, "heroism", target_id="wizard")
    encounter.apply(hero, "test")
    assert caster.concentrating_on == "heroism"
    assert "paralyzed" not in foe.conditions
    assert attack_mode(caster, foe) == 0


def test_ghoul_claw_paralyzes_until_the_targets_next_end():
    ghoul = from_monster(_monster("ghoul"), creature_id="ghoul", name="Ghoul", side="enemy", position=(0, 0))
    hero = creature("hero", "Hero", "party", 12, 30, (1, 0), attacks=[sword()])
    encounter = Encounter([ghoul, hero], width=12, height=12, seed=1)
    _prepare(encounter, ["ghoul", "hero"], [18, 2, 1])
    claw = next(action for action in encounter.legal_actions(ghoul) if action.attack_name == "Claw")
    encounter.apply(claw, "test")
    assert hero.hp == 26
    assert "paralyzed" in hero.conditions
    assert not encounter.rng.rolls
    finish_turn(encounter)
    encounter.begin_turn()
    assert "paralyzed" in hero.conditions
    finish_turn(encounter)
    assert "paralyzed" not in hero.conditions


def test_paralyzed_immunity_skips_the_ghoul_save():
    ghoul = from_monster(_monster("ghoul"), creature_id="ghoul", name="Ghoul", side="enemy", position=(0, 0))
    hero = creature(
        "hero", "Hero", "party", 12, 30, (1, 0),
        traits={"condition_immunities": ["paralyzed"]},
    )
    encounter = Encounter([ghoul, hero], width=12, height=12, seed=1)
    _prepare(encounter, ["ghoul", "hero"], [18, 2])
    claw = next(action for action in encounter.legal_actions(ghoul) if action.attack_name == "Claw")
    encounter.apply(claw, "test")
    assert hero.hp == 26
    assert "paralyzed" not in hero.conditions
    assert not encounter.rng.rolls
    assert "immune" in " ".join(encounter.log).lower()


def test_wolf_bite_knocks_prone_and_standing_costs_half_speed():
    from src.rules import attack_mode

    wolf = from_monster(_monster("wolf"), creature_id="wolf", name="Wolf", side="enemy", position=(0, 0))
    hero = creature("hero", "Hero", "party", 14, 30, (1, 0), attacks=[sword()], speed=30)
    encounter = Encounter([wolf, hero], width=12, height=12, seed=1)
    _prepare(encounter, ["wolf", "hero"], [16, 3])
    bite = next(action for action in encounter.legal_actions(wolf) if action.attack_name == "Bite")
    encounter.apply(bite, "test")
    assert "prone" in hero.conditions
    assert attack_mode(hero, wolf) == -1
    assert not encounter.rng.rolls
    finish_turn(encounter)
    encounter.begin_turn()
    stand = next(action for action in encounter.legal_actions(hero) if action.kind == "stand")
    encounter.apply(stand, "test")
    assert "prone" not in hero.conditions
    assert hero.movement_left == 15


def test_blindness_deafness_uses_the_chosen_condition():
    from src.rules import attack_mode

    spell = cache_spell("blindness-deafness")
    caster = creature(
        "wizard", "Wizard", "party", 14, 30, (0, 0),
        spells=[spell], slots={2: 1}, spell_save_dc=15,
    )
    foe = creature("foe", "Foe", "enemy", 12, 20, (1, 0), attacks=[sword()])
    encounter = Encounter([caster, foe], width=12, height=12, seed=1)
    _prepare(encounter, ["wizard", "foe"], [1])
    names = {
        action.attack_name
        for action in encounter.legal_actions(caster)
        if action.spell_index == "blindness-deafness"
    }
    assert names == {"blinded", "deafened"}
    encounter.apply(_spell_action(encounter, caster, "blindness-deafness", attack_name="blinded"), "test")
    assert "blinded" in foe.conditions
    assert "deafened" not in foe.conditions
    assert attack_mode(foe, caster) == -1
    assert attack_mode(caster, foe) == 1


def test_stunned_can_move_and_paralyzed_cannot_and_a_near_hit_is_a_critical():
    stun = cache_spell("power-word-stun")
    caster = creature(
        "wizard", "Wizard", "party", 14, 30, (0, 0),
        spells=[stun], slots={8: 1},
    )
    foe = creature("foe", "Foe", "enemy", 12, 40, (1, 0), speed=30)
    encounter = Encounter([caster, foe], width=12, height=12, seed=1)
    _prepare(encounter, ["wizard", "foe"], [])
    encounter.apply(_spell_action(encounter, caster, "power-word-stun", target_id="foe"), "test")
    assert "stunned" in foe.conditions
    assert not encounter.rng.rolls
    finish_turn(encounter)
    step = next(action for action in encounter.legal_actions(foe) if action.kind == "move" and action.destination == (2, 0))
    encounter.apply(step, "test")
    assert foe.position == (2, 0)
    assert "stunned" in foe.conditions

    hold = cache_spell("hold-person")
    fighter = creature(
        "fighter", "Fighter", "party", 16, 30, (0, 0),
        attacks=[sword()], spells=[hold], slots={2: 1}, spell_save_dc=15,
    )
    locked = creature("locked", "Locked", "enemy", 12, 40, (1, 0), speed=30)
    fight = Encounter([fighter, locked], width=12, height=12, seed=1)
    _prepare(fight, ["fighter", "locked"], [1, 1, 12, 8, 4, 5])
    fight.apply(_spell_action(fight, fighter, "hold-person", target_id="locked"), "test")
    finish_turn(fight)
    fight.begin_turn()
    assert "paralyzed" in locked.conditions
    assert all(action.kind != "move" for action in fight.legal_actions(locked))
    finish_turn(fight)
    fight.begin_turn()
    swing = next(action for action in fight.legal_actions(fighter) if action.attack_name == "Sword")
    fight.apply(swing, "test")
    assert not fight.rng.rolls
    assert "Dice are doubled" in " ".join(fight.log)
    assert locked.hp == 29


def test_grease_lingers_after_a_new_concentration_spell():
    from src.zones import centered_block

    grease = cache_spell("grease")
    brave = cache_spell("heroism")
    caster = creature(
        "wizard", "Wizard", "party", 14, 30, (0, 0),
        spells=[grease, brave], slots={1: 2}, spell_save_dc=15,
    )
    foe = creature("foe", "Foe", "enemy", 12, 20, (2, 0), speed=30)
    encounter = Encounter([caster, foe], width=12, height=12, seed=1)
    _prepare(encounter, ["wizard", "foe"], [20, 1])
    aims = [action.aim for action in encounter.legal_actions(caster) if action.spell_index == "grease"]
    assert (2, 0) in aims
    assert (0, 0) not in aims
    assert all(foe.in_cells(centered_block(aim, 10)) for aim in aims)
    encounter.apply(_spell_action(encounter, caster, "grease", aim=(2, 0)), "test")
    aura = {tuple(pair) for pair in _mat(encounter, caster)["aura"]}
    assert {(2, 0), (2, 1), (3, 0), (3, 1)} <= aura
    assert "prone" not in foe.conditions
    assert caster.traits["lingers"][0]["index"] == "grease"
    assert caster.concentrating_on is None
    finish_turn(encounter)
    encounter.begin_turn()
    assert "prone" not in foe.conditions
    finish_turn(encounter)
    assert "prone" in foe.conditions
    encounter.begin_turn()
    encounter.apply(_spell_action(encounter, caster, "heroism", target_id="wizard"), "test")
    assert caster.concentrating_on == "heroism"
    assert caster.traits["lingers"][0]["index"] == "grease"
    assert "prone" in foe.conditions


def test_sleep_second_failure_is_unconscious_and_an_elf_skips_the_die():
    spell = cache_spell("sleep")
    caster = creature(
        "wizard", "Wizard", "party", 14, 30, (0, 0),
        spells=[spell], slots={1: 1}, spell_save_dc=15,
    )
    foe = creature("foe", "Foe", "enemy", 12, 20, (3, 0))
    encounter = Encounter([caster, foe], width=12, height=12, seed=1)
    _prepare(encounter, ["wizard", "foe"], [1, 1])
    encounter.apply(_spell_action(encounter, caster, "sleep", aim=(3, 0)), "test")
    assert "incapacitated" in foe.conditions
    assert "unconscious" not in foe.conditions
    assert encounter.current().id == "wizard"
    finish_turn(encounter)
    encounter.begin_turn()
    finish_turn(encounter)
    assert "unconscious" in foe.conditions
    assert not encounter.rng.rolls

    elf = creature("elf", "Elf", "enemy", 12, 20, (3, 0), traits={"species": "High Elf"})
    quiet = Encounter(
        [creature("wizard", "Wizard", "party", 14, 30, (0, 0), spells=[spell], slots={1: 1}, spell_save_dc=15), elf],
        width=12, height=12, seed=1,
    )
    _prepare(quiet, ["wizard", "elf"], [4])
    quiet.apply(_spell_action(quiet, quiet.by_id("wizard"), "sleep", aim=(3, 0)), "test")
    assert quiet.rng.rolls == [4]
    assert "unaffected" in " ".join(quiet.log).lower()
    assert "incapacitated" not in elf.conditions


def test_command_grovel_knocks_prone_on_the_targets_next_turn():
    spell = cache_spell("command")
    caster = creature(
        "cleric", "Cleric", "party", 14, 30, (0, 0),
        spells=[spell], slots={1: 1}, spell_save_dc=15,
    )
    foe = creature("foe", "Foe", "enemy", 12, 20, (4, 0), speed=30)
    encounter = Encounter([caster, foe], width=12, height=12, seed=1)
    _prepare(encounter, ["cleric", "foe"], [1])
    encounter.apply(_spell_action(encounter, caster, "command", attack_name="grovel", target_id="foe"), "test")
    assert "prone" not in foe.conditions
    assert encounter.current().id == "cleric"
    finish_turn(encounter)
    assert encounter.current().id == "foe"
    assert foe.movement_left is None
    encounter.begin_turn()
    actions = encounter.legal_actions(foe)
    assert [action.kind for action in actions] == ["order"]
    assert actions[0].attack_name == "grovel"
    from src.ai import heuristic_advice
    chosen, reason = heuristic_advice(encounter, foe)
    assert chosen.kind == "order"
    assert reason == "The order is the only action."
    encounter.apply(actions[0], "test")
    assert "prone" in foe.conditions
    assert encounter.current().id == "cleric"


def _sword_action(encounter, actor, target_id=None):
    for action in encounter.legal_actions(actor):
        if action.attack_name != "Sword":
            continue
        if target_id is not None and action.target_id != target_id:
            continue
        return action
    raise AssertionError("Sword")


def test_a_healthy_brute_is_locked_and_a_dying_brute_is_struck():
    from src.ai import heuristic_advice

    hold = cache_spell("hold-person")

    def fight(hit_points):
        caster = creature(
            "wizard", "Wizard", "party", 16, 40, (0, 0),
            attacks=[sword()], spells=[hold], slots={2: 1}, spell_save_dc=15,
        )
        brute = creature(
            "brute", "Brute", "enemy", 14, hit_points, (1, 0),
            attacks=[club()], multiattack=[("Club", 3)],
        )
        return Encounter([caster, brute], width=12, height=12, seed=1), caster

    healthy, caster = fight(80)
    locked = _spell_action(healthy, caster, "hold-person", target_id="brute")
    swing = _sword_action(healthy, caster)
    assert action_score(healthy, caster, locked) > action_score(healthy, caster, swing)
    assert action_score(healthy, caster, locked) > 3
    chosen, reason = heuristic_advice(healthy, caster)
    assert chosen.spell_index == "hold-person"
    assert reason == "The lock removes more pressure than the hit."

    dying, caster = fight(6)
    locked = _spell_action(dying, caster, "hold-person", target_id="brute")
    swing = _sword_action(dying, caster)
    assert action_score(dying, caster, swing) > action_score(dying, caster, locked)
    chosen, reason = heuristic_advice(dying, caster)
    assert chosen.attack_name == "Sword"
    assert reason == "Highest expected damage."


def test_blinded_beats_deafened_and_charm_loses_to_a_sword():
    from src.ai import heuristic_advice

    blind = cache_spell("blindness-deafness")
    caster = creature(
        "wizard", "Wizard", "party", 14, 30, (0, 0),
        spells=[blind], slots={2: 1}, spell_save_dc=15,
    )
    foe = creature("foe", "Foe", "enemy", 14, 40, (1, 0), attacks=[sword()])
    encounter = Encounter([caster, foe], width=12, height=12, seed=1)
    blinded = _spell_action(encounter, caster, "blindness-deafness", attack_name="blinded")
    deafened = _spell_action(encounter, caster, "blindness-deafness", attack_name="deafened")
    assert action_score(encounter, caster, blinded) > action_score(encounter, caster, deafened)
    chosen, reason = heuristic_advice(encounter, caster)
    assert chosen.attack_name == "blinded"
    assert reason == "The lock removes more pressure than the hit."

    charm = cache_spell("charm-person")
    charmer = creature(
        "bard", "Bard", "party", 14, 30, (0, 0),
        attacks=[sword()], spells=[charm], slots={1: 1}, spell_save_dc=15,
    )
    victim = creature("victim", "Victim", "enemy", 14, 40, (1, 0), attacks=[sword()])
    charmed = Encounter([charmer, victim], width=12, height=12, seed=1)
    charm_action = _spell_action(charmed, charmer, "charm-person", target_id="victim")
    swing = _sword_action(charmed, charmer)
    assert action_score(charmed, charmer, charm_action) < action_score(charmed, charmer, swing)
    chosen, reason = heuristic_advice(charmed, charmer)
    assert chosen.attack_name == "Sword"
    assert reason == "Highest expected damage."


def test_lone_sleep_loses_to_hold_and_a_second_foe_makes_hypnotic_worth_it():
    sleep = cache_spell("sleep")
    hold = cache_spell("hold-person")
    caster = creature(
        "wizard", "Wizard", "party", 14, 30, (0, 0),
        attacks=[sword()], spells=[sleep, hold], slots={1: 1, 2: 1}, spell_save_dc=15,
    )
    foe = creature("foe", "Foe", "enemy", 14, 40, (1, 0), attacks=[sword()])
    alone = Encounter([caster, foe], width=12, height=12, seed=1)
    sleep_action = _spell_action(alone, caster, "sleep", aim=(1, 0))
    hold_action = _spell_action(alone, caster, "hold-person", target_id="foe")
    swing = _sword_action(alone, caster)
    assert action_score(alone, caster, sleep_action) < action_score(alone, caster, hold_action)
    assert action_score(alone, caster, sleep_action) < action_score(alone, caster, swing)

    pattern = cache_spell("hypnotic-pattern")
    caster = creature(
        "wizard", "Wizard", "party", 14, 30, (0, 0),
        attacks=[sword()], spells=[pattern], slots={3: 1}, spell_save_dc=15,
    )
    one = creature("one", "One", "enemy", 14, 40, (1, 0), attacks=[sword()])
    two = creature("two", "Two", "enemy", 14, 40, (6, 0), attacks=[sword()])
    crowd = Encounter([caster, one, two], width=12, height=12, seed=1)
    hypnotic = _spell_action(crowd, caster, "hypnotic-pattern", aim=(1, 0))
    swing = _sword_action(crowd, caster, "one")
    assert action_score(crowd, caster, hypnotic) > action_score(crowd, caster, swing)


def test_a_slept_foe_is_left_alone_while_another_enemy_fights():
    from src.ai import heuristic_advice
    from src.conditions import add_mark

    hero = creature("hero", "Hero", "party", 14, 30, (0, 0), attacks=[sword()])
    sleeper = creature("sleeper", "Sleeper", "enemy", 14, 40, (1, 0), attacks=[sword()])
    awake = creature("awake", "Awake", "enemy", 14, 40, (0, 1), attacks=[sword()])
    encounter = Encounter([hero, sleeper, awake], width=12, height=12, seed=1)
    add_mark(
        sleeper, hero, condition="incapacitated", until="concentration",
        ends_on_damage=True, spell="hypnotic-pattern",
    )
    sleeping = _sword_action(encounter, hero, "sleeper")
    waking = _sword_action(encounter, hero, "awake")
    assert action_score(encounter, hero, waking) > action_score(encounter, hero, sleeping)
    chosen, reason = heuristic_advice(encounter, hero)
    assert chosen.attack_name == "Sword"
    assert chosen.target_id == "awake"
    assert reason == "Highest expected damage."


def test_hold_person_on_someone_already_paralyzed_loses_to_the_sword():
    from src.conditions import add_mark

    hold = cache_spell("hold-person")
    caster = creature(
        "wizard", "Wizard", "party", 14, 30, (0, 0),
        attacks=[sword()], spells=[hold], slots={2: 1}, spell_save_dc=15,
    )
    foe = creature("foe", "Foe", "enemy", 14, 40, (1, 0), attacks=[sword()])
    encounter = Encounter([caster, foe], width=12, height=12, seed=1)
    add_mark(foe, caster, condition="paralyzed", until="concentration", spell="hold-person")
    locked = _spell_action(encounter, caster, "hold-person", target_id="foe")
    swing = _sword_action(encounter, caster)
    assert action_score(encounter, caster, locked) < action_score(encounter, caster, swing)


def test_a_plain_sword_matches_expected_damage_when_it_cannot_kill():
    from src.ai import _expected_parts

    attacker = creature("hero", "Hero", "party", 14, 30, (0, 0), attacks=[sword()])
    foe = creature("foe", "Foe", "enemy", 14, 40, (1, 0))
    encounter = Encounter([attacker, foe], width=12, height=12, seed=1)
    swing = _sword_action(encounter, attacker)
    assert action_score(encounter, attacker, swing) == _expected_parts(
        [("slashing", "1d6+2")], foe, bonus=4
    )


def test_a_prone_creature_stands_before_it_attacks():
    from src.ai import heuristic_advice
    from src.conditions import add_mark

    hero = creature("hero", "Hero", "party", 14, 30, (0, 0), attacks=[sword()], speed=30)
    foe = creature("foe", "Foe", "enemy", 14, 40, (1, 0))
    encounter = Encounter([hero, foe], width=12, height=12, seed=1)
    _prepare(encounter, ["hero", "foe"], [])
    add_mark(hero, foe, condition="prone", until="stand")
    chosen, reason = heuristic_advice(encounter, hero)
    assert chosen.kind == "stand"
    assert reason == "Stand before attacking."
    encounter.apply(chosen, "test")
    assert "prone" not in hero.conditions
    follow = heuristic_choice(encounter, hero)
    assert follow.kind == "attack"
    assert follow.attack_name == "Sword"


def test_an_incapacitated_target_drops_command_at_the_end_of_the_turn():
    from src.conditions import add_mark, pending_order

    spell = cache_spell("command")
    caster = creature(
        "cleric", "Cleric", "party", 14, 30, (0, 0),
        spells=[spell], slots={1: 1}, spell_save_dc=15,
    )
    foe = creature("foe", "Foe", "enemy", 12, 20, (4, 0), speed=30)
    encounter = Encounter([caster, foe], width=12, height=12, seed=1)
    _prepare(encounter, ["cleric", "foe"], [1])
    encounter.apply(_spell_action(encounter, caster, "command", attack_name="grovel", target_id="foe"), "test")
    finish_turn(encounter)
    encounter.begin_turn()
    assert pending_order(foe) is not None
    add_mark(foe, caster, condition="paralyzed", until="fight")
    assert all(action.kind != "order" for action in encounter.legal_actions(foe))
    finish_turn(encounter)
    assert pending_order(foe) is None


def test_one_square_of_grease_costs_ten_feet():
    from src.actions import GameAction

    grease = cache_spell("grease")
    caster = creature(
        "wizard", "Wizard", "party", 14, 30, (0, 0),
        spells=[grease], slots={1: 1}, spell_save_dc=15,
    )
    foe = creature("foe", "Foe", "enemy", 12, 20, (2, 0), speed=30)
    encounter = Encounter([caster, foe], width=12, height=12, seed=1)
    _prepare(encounter, ["wizard", "foe"], [20])
    encounter.begin_turn()
    # The menu only offers a patch that already covers a foe, so this cast is placed directly.
    encounter._place_zone(
        caster,
        grease,
        1,
        GameAction(id=0, kind="spell", label="Grease", spell_index="grease", aim=(3, 0), slot_level=1),
    )
    assert encounter.rng.rolls == [20]
    assert "prone" not in foe.conditions
    finish_turn(encounter)
    encounter.begin_turn()
    step = next(action for action in encounter.legal_actions(foe) if action.destination == (3, 0))
    encounter.apply(step, "test")
    assert foe.position == (3, 0)
    assert foe.movement_left == 20
    assert "prone" not in foe.conditions
    assert encounter.rng.rolls == []


def test_poison_ward_gives_advantage_on_the_contagion_save():
    from src.conditions import add_mark

    spell = cache_spell("contagion")
    assert spell.range_mode == "touch"
    assert spell.range_feet == 5
    caster = creature(
        "cleric", "Cleric", "party", 14, 30, (0, 0),
        spells=[spell], slots={5: 1}, spell_save_dc=15,
    )
    foe = creature("foe", "Foe", "enemy", 12, 80, (1, 0))
    encounter = Encounter([caster, foe], width=12, height=12, seed=1)
    _prepare(encounter, ["cleric", "foe"], [10, 19])
    add_mark(foe, caster, condition="", until="fight", poison_ward=True)
    assert foe.traits.get("poison_ward") is True
    encounter.apply(_spell_action(encounter, caster, "contagion", target_id="foe"), "test")
    assert "poisoned" not in foe.conditions
    assert foe.hp == 80
    assert encounter.rng.rolls == []
    assert "Saved" in " ".join(encounter.log)


def test_fear_with_nowhere_farther_keeps_the_normal_menu():
    from src.conditions import add_mark

    caster = creature("wizard", "Wizard", "party", 14, 30, (0, 0))
    foe = creature("foe", "Foe", "enemy", 14, 30, (11, 11), attacks=[sword()], speed=30)
    encounter = Encounter([caster, foe], width=12, height=12, seed=1)
    add_mark(foe, caster, condition="frightened", until="concentration", spell="fear")
    add_mark(
        foe, caster, condition="", order="away", repeat_order=True,
        until="concentration", tied="concentration", spell="fear",
    )
    actions = encounter.legal_actions(foe)
    assert actions
    assert all(action.kind != "order" for action in actions)
    chosen = heuristic_choice(encounter, foe)
    assert chosen.kind != "order"


def test_fireball_does_not_pay_for_a_held_lock():
    fireball = cache_spell("fireball")
    caster = creature(
        "wizard", "Wizard", "party", 14, 40, (0, 0),
        spells=[fireball], slots={3: 1}, spell_save_dc=15, concentrating_on="hold-person",
    )
    foe = creature("foe", "Foe", "enemy", 14, 40, (4, 0), attacks=[sword()])
    encounter = Encounter([caster, foe], width=12, height=12, seed=1)
    from src.conditions import add_mark
    add_mark(
        foe, caster, condition="paralyzed", until="concentration", tied="concentration",
        spell="hold-person", repeat=True, save="wisdom", dc=15,
    )
    blast = _spell_action(encounter, caster, "fireball", aim=(4, 0))
    holding = action_score(encounter, caster, blast)
    caster.concentrating_on = None
    assert action_score(encounter, caster, blast) == holding


def test_the_action_prompt_names_pressure_and_lists_conditions():
    from src.ai import ACTION_SYSTEM, heuristic_advice

    assert "pressure removed" in ACTION_SYSTEM
    assert "Blinded beats Deafened" in ACTION_SYSTEM
    foe = creature("pc", "PC", "party", 14, 30, (1, 0))
    foe.conditions["poisoned"] = 99
    enemy = creature("gob", "Gob", "enemy", 12, 20, (0, 0), attacks=[sword()])
    enemy.concentrating_on = "hold-person"
    encounter = Encounter([foe, enemy], seed=1)
    prompt = render_prompt(encounter, enemy, encounter.legal_actions(enemy, automated=True))
    decoded = decode(prompt)
    from toon_format import encode
    assert decode(encode(decoded, delimiter="\t")) == decoded
    assert decoded["actor"]["concentrating"] == "hold-person"
    assert decoded["actor"]["conditions"] == ""
    rows = {row["name"]: row for row in decoded["combatants"]}
    assert rows["PC"]["conditions"] == "Poisoned"
    assert "expected" in next(line for line in prompt.splitlines() if line.startswith("actions["))
    legal = encounter.legal_actions(enemy, automated=True)
    sword_id = next(action.id for action in legal if action.attack_name == "Sword")
    chosen, source, reason = choose_action(
        encounter,
        enemy,
        "llm",
        _Replies([json.dumps({"action_id": sword_id, "reason": "press"})]),
    )
    assert source == "llm"
    assert chosen.id == sword_id
    assert reason == "press"
    advice, why = heuristic_advice(encounter, enemy)
    assert advice.kind == "attack"
    assert why == "Highest expected damage."
