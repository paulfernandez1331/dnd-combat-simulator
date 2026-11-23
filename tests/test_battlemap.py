"""Click the battle mat in Edge. Run this when a mat change needs a browser.

    python -m pytest -q tests/test_battlemap.py
"""

import json

import pytest

from src.actions import GameAction, legal_actions
from src.bestiary import AttackOption, classify_spell
from src.board import _frame_height, battle_payload
from src.combatant import Combatant, blank_abilities, save_bonuses_for
from src.encounter import Encounter
from src.paths import CACHE
from src.targeting import abilities_from
from board_browser import open_board

pytestmark = pytest.mark.battlemap


def _weapon(name: str) -> AttackOption:
    return AttackOption(
        name=name,
        resolution="attack",
        attack_bonus=5,
        reach_feet=5,
        damage=[{"type": "slashing", "expr": "1d8"}],
    )


def _creature(creature_id, name, side, position, attacks) -> Combatant:
    abilities = blank_abilities()
    return Combatant(
        id=creature_id,
        name=name,
        side=side,
        ac=14,
        max_hp=20,
        hp=20,
        speed=30,
        abilities=abilities,
        save_bonuses=save_bonuses_for(abilities, [], 2),
        position=position,
        attacks=attacks,
    )


def _melee_spec(armed: bool) -> tuple[dict, str]:
    """Hero stands on the enemy's left, which is where the action list used to land."""
    hero = _creature("hero", "Hero", "party", (2, 3), [_weapon("Sword"), _weapon("Axe")])
    goblin = _creature("goblin", "Goblin Warrior 1", "enemy", (3, 3), [])
    encounter = Encounter([hero, goblin], width=8, height=8)
    actions = legal_actions(encounter, hero, automated=False)
    abilities = abilities_from(actions)
    sword = next(ability for ability in abilities if ability.label == "Sword")
    spec = battle_payload(
        encounter,
        mode="combat",
        actor=hero,
        actions=actions,
        abilities=abilities,
        armed=sword if armed else None,
        slot=None,
        selected_id=hero.id,
    )
    return spec, sword.key


@pytest.fixture(scope="module")
def board():
    with open_board() as page:
        yield page


def test_circles_fill_their_space_and_names_sit_outside(board):
    hero = _creature("hero", "Hero", "party", (5, 3), [_weapon("Sword")])
    dragon = _creature("dragon", "Ancient Dragon", "enemy", (0, 0), [])
    dragon.size = "Huge"
    encounter = Encounter([hero, dragon], width=8, height=8)
    spec = battle_payload(
        encounter,
        mode="place",
        actor=None,
        actions=[],
        abilities=[],
        armed=None,
        slot=None,
        selected_id=hero.id,
    )
    board.show(spec)
    metrics = board.page.evaluate(
        """() => {
            function box(node) {
                const rect = node.getBoundingClientRect();
                return { w: rect.width, h: rect.height, l: rect.left, t: rect.top, r: rect.right, b: rect.bottom };
            }
            function separated(name, body) {
                return name.b <= body.t + 1 || name.t >= body.b - 1 || name.r <= body.l || name.l >= body.r;
            }
            const medium = document.querySelector(".token.party");
            const body = box(medium.querySelector(".body"));
            const name = box(medium.querySelector(".name"));
            const cell = box(document.querySelector('button.cell[data-x="5"][data-y="3"]'));
            const centerX = body.l + body.w / 2;
            const centerY = body.t + body.h / 2;
            const huge = box(document.querySelector(".token.enemy .body"));
            const mid = box(document.querySelector('button.cell[data-x="1"][data-y="1"]'));
            return {
                bw: body.w,
                bh: body.h,
                separate: separated(name, body),
                dx: Math.abs(centerX - (cell.l + cell.w / 2)),
                dy: Math.abs(centerY - (cell.t + cell.h / 2)),
                cw: cell.w,
                hw: huge.w,
                hh: huge.h,
                hdx: Math.abs((huge.l + huge.w / 2) - (mid.l + mid.w / 2)),
                hdy: Math.abs((huge.t + huge.h / 2) - (mid.t + mid.h / 2)),
            };
        }"""
    )
    assert metrics["bw"] == pytest.approx(metrics["bh"])
    assert metrics["bh"] > 48
    assert metrics["separate"] is True
    assert metrics["dx"] < 2
    assert metrics["dy"] < 2
    assert metrics["cw"] == pytest.approx(64, abs=0.5)
    assert metrics["hw"] == pytest.approx(3 * 64 - 8, abs=1.5)
    assert metrics["hh"] == pytest.approx(metrics["hw"])
    assert metrics["hdx"] < 2
    assert metrics["hdy"] < 2
    assert board.hit_at(6, 3)["covered"] is False
    board.clear()
    board.click_cell(1, 1)
    square = board.values()[-1]
    assert square["kind"] == "square"
    assert (square["x"], square["y"]) == (1, 1)


def test_a_token_image_fills_the_circle_and_initials_remain_without_one(board):
    hero = _creature("hero", "Hero", "party", (2, 2), [])
    goblin = _creature("goblin", "Goblin", "enemy", (4, 2), [])
    encounter = Encounter([hero, goblin], width=8, height=8)
    spec = battle_payload(
        encounter,
        mode="place",
        actor=None,
        actions=[],
        abilities=[],
        armed=None,
        slot=None,
        selected_id=None,
    )
    picture = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    for token in spec["tokens"]:
        if token["id"] == "hero":
            token["image"] = picture
    board.show(spec)
    painted = board.page.evaluate(
        """() => {
            const party = document.querySelector(".token.party .body");
            const enemy = document.querySelector(".token.enemy .body");
            const face = party.querySelector("img.face");
            return {
                face: face ? face.getAttribute("src").slice(0, 5) : "",
                initials: enemy.textContent,
                round: getComputedStyle(party).borderRadius,
            };
        }"""
    )
    assert painted["face"] == "data:"
    assert painted["initials"] == "G"
    assert painted["round"] == "50%"


def test_a_twenty_by_twenty_mat_stays_inside_the_window(board):
    hero = _creature("hero", "Hero", "party", (2, 5), [_weapon("Sword")])
    encounter = Encounter([hero], width=20, height=20)
    spec = battle_payload(
        encounter,
        mode="place",
        actor=None,
        actions=[],
        abilities=[],
        armed=None,
        slot=None,
        selected_id=hero.id,
    )
    board.show(spec)
    metrics = board.page.evaluate(
        """() => {
            const cells = Array.from(document.querySelectorAll("button.cell"));
            const rects = cells.map((node) => node.getBoundingClientRect());
            const grid = document.querySelector("#grid").getBoundingClientRect();
            const width = rects[0].width;
            const height = rects[0].height;
            return {
                n: cells.length,
                w: width,
                h: height,
                same: rects.every((rect) => Math.abs(rect.width - width) < 0.5 && Math.abs(rect.height - height) < 0.5),
                right: grid.right,
                bottom: grid.bottom,
                viewW: window.innerWidth,
                viewH: window.innerHeight,
            };
        }"""
    )
    assert metrics["n"] == 20 * 20
    assert metrics["same"] is True
    assert metrics["w"] == pytest.approx(metrics["h"])
    assert metrics["w"] < 64
    assert metrics["w"] >= 20
    assert metrics["right"] <= metrics["viewW"] + 1
    assert metrics["bottom"] <= metrics["viewH"] + 1
    board.click_cell(19, 19)
    square = board.values()[-1]
    assert square["kind"] == "square"
    assert (square["x"], square["y"]) == (19, 19)


def test_the_first_iframe_is_capped_before_the_mat_can_shrink():
    assert _frame_height({"height": 8}) == 8 * 64 + 96 + 32
    assert _frame_height({"height": 20}) == 720


def test_open_attack_list_leaves_the_adjacent_enemy_clickable(board):
    spec, sword_key = _melee_spec(armed=False)
    board.show(spec)
    board.click("Action")
    board.click("Attack")
    assert board.hit_at(3, 3)["covered"] is False

    board.clear()
    board.click_cell(3, 3)
    square = board.values()[-1]
    assert square["kind"] == "square"
    assert (square["x"], square["y"]) == (3, 3)

    board.clear()
    board.click("Sword")
    chosen = board.values()[-1]
    assert chosen["kind"] == "ability"
    assert chosen["key"] == sword_key


def test_armed_melee_names_the_enemy_and_the_click_selects_that_square(board):
    spec, _sword_key = _melee_spec(armed=True)
    board.show(spec)
    assert board.hint() == "Click a gold-ringed creature: Goblin Warrior 1."
    assert board.page.locator(".callout").inner_text() == "Goblin Warrior 1"
    assert board.page.locator(".menu").count() == 0
    assert "Sword" in board.page.locator(".chip").inner_text()
    assert board.hit_at(3, 3)["covered"] is False

    board.clear()
    board.click_cell(3, 3)
    square = board.values()[-1]
    assert square["kind"] == "square"
    assert (square["x"], square["y"]) == (3, 3)

    board.clear()
    board.click("Cancel")
    assert board.values()[-1]["kind"] == "clear"


def test_a_spiritual_weapon_marker_shows_no_hit_points(board):
    hero = _creature("hero", "Hero", "party", (2, 3), [_weapon("Sword")])
    hero.concentrating_on = "spiritual-weapon"
    hero.traits["spiritual_weapon"] = {"x": 4, "y": 3, "slot": 2}
    goblin = _creature("goblin", "Goblin Warrior 1", "enemy", (6, 3), [])
    encounter = Encounter([hero, goblin], width=8, height=8)
    spec = battle_payload(
        encounter,
        mode="combat",
        actor=hero,
        actions=[],
        abilities=[],
        armed=None,
        slot=None,
        selected_id=hero.id,
    )
    board.show(spec)
    marker = board.page.locator(".token.marker")
    assert marker.count() == 1
    assert marker.locator(".name").inner_text() == "Spiritual Weapon"
    assert marker.locator(".hp-text").count() == 0
    assert marker.locator(".hp").count() == 0
    weapon_label = board.page.locator('button.cell[data-x="4"][data-y="3"]').get_attribute("aria-label")
    assert weapon_label.startswith("Spiritual Weapon")
    assert "hit points" not in weapon_label
    hero_label = board.page.locator('button.cell[data-x="2"][data-y="3"]').get_attribute("aria-label")
    assert "20 of 20 hit points" in hero_label


def test_a_token_label_lists_a_condition(board):
    from src.conditions import add_mark

    hero = _creature("hero", "Hero", "party", (2, 3), [_weapon("Sword")])
    goblin = _creature("goblin", "Goblin Warrior 1", "enemy", (6, 3), [])
    add_mark(goblin, hero, condition="blinded", until="fight")
    encounter = Encounter([hero, goblin], width=8, height=8)
    spec = battle_payload(
        encounter,
        mode="combat",
        actor=hero,
        actions=[],
        abilities=[],
        armed=None,
        slot=None,
        selected_id=hero.id,
    )
    board.show(spec)
    assert board.page.locator(".token.enemy .status").inner_text() == "Blinded"
    label = board.page.locator('button.cell[data-x="6"][data-y="3"]').get_attribute("aria-label")
    assert "20 of 20 hit points" in label
    assert "Blinded" in label
    hero_label = board.page.locator('button.cell[data-x="2"][data-y="3"]').get_attribute("aria-label")
    assert "20 of 20 hit points" in hero_label
    assert "Blinded" not in hero_label


def test_an_emanation_stays_drawn_apart_from_the_aim_preview(board):
    spell = classify_spell(json.loads((CACHE / "spells" / "spirit-guardians.json").read_text(encoding="utf-8")))
    hero = _creature("hero", "Hero", "party", (2, 3), [_weapon("Sword")])
    hero.spells = [spell]
    hero.concentrating_on = "spirit-guardians"
    hero.ongoing_slot = 3
    goblin = _creature("goblin", "Goblin Warrior 1", "enemy", (6, 3), [])
    encounter = Encounter([hero, goblin], width=8, height=8)
    spec = battle_payload(
        encounter,
        mode="combat",
        actor=hero,
        actions=[],
        abilities=[],
        armed=None,
        slot=None,
        selected_id=hero.id,
    )
    assert spec["aura"]
    assert spec["preview"] == []
    assert [2, 3] in spec["aura"]
    assert [6, 3] not in spec["aura"]
    board.show(spec)
    before = board.page.evaluate(
        """() => ({
            aura: document.querySelectorAll("#aura .aura").length,
            amber: document.querySelectorAll("#amber .amber").length,
            auraInAmber: document.querySelectorAll("#amber .aura").length
        })"""
    )
    assert before["aura"] == len(spec["aura"])
    assert before["amber"] == 0
    assert before["auraInAmber"] == 0
    # The caster's own square is the center of the emanation.
    aligned = board.page.evaluate(
        """() => {
            const button = document.querySelector('button.cell[data-x="2"][data-y="3"]');
            const box = button.getBoundingClientRect();
            return [...document.querySelectorAll("#aura .aura")].some((mark) => {
                const rect = mark.getBoundingClientRect();
                return Math.abs(rect.left - box.left) < 1.5
                    && Math.abs(rect.top - box.top) < 1.5
                    && Math.abs(rect.width - box.width) < 1.5
                    && Math.abs(rect.height - box.height) < 1.5;
            });
        }"""
    )
    assert aligned is True
    board.page.locator('button.cell[data-x="4"][data-y="3"]').hover()
    after = board.page.evaluate(
        """() => ({
            aura: document.querySelectorAll("#aura .aura").length,
            amber: document.querySelectorAll("#amber .amber").length
        })"""
    )
    assert after["aura"] == before["aura"]
    assert after["amber"] == 0


def test_the_action_menu_scrolls_and_keeps_names_off_the_slot_chips(board):
    hero = _creature("hero", "Hero", "party", (2, 3), [_weapon("Sword")])
    goblin = _creature("goblin", "Goblin Warrior 1", "enemy", (3, 3), [])
    actions = []
    number = 1
    for name in (
        "Entangle", "Faerie Fire", "Spike Growth", "Poison Spray", "Guiding Bolt",
        "Hold Person", "Moonbeam", "Call Lightning", "Healing Word", "Cure Wounds",
        "Thunderwave", "Barkskin",
    ):
        for slot in (1, 2, 3):
            actions.append(GameAction(
                id=number,
                kind="spell",
                label=f"{name} L{slot}",
                spell_index=name.lower().replace(" ", "-"),
                target_id="goblin",
                slot_level=slot,
                budget="action",
            ))
            number += 1
    for kind, label in (("dash", "Dash"), ("disengage", "Disengage"), ("dodge", "Dodge")):
        actions.append(GameAction(id=number, kind=kind, label=label, budget="action"))
        number += 1
    encounter = Encounter([hero, goblin], width=8, height=8)
    abilities = abilities_from(actions)
    spec = battle_payload(
        encounter,
        mode="combat",
        actor=hero,
        actions=actions,
        abilities=abilities,
        armed=None,
        slot=None,
        selected_id=hero.id,
    )
    board.show(spec)
    board.click("Action")
    metrics = board.page.evaluate(
        """() => {
            function box(node) {
                const rect = node.getBoundingClientRect();
                return { top: rect.top, bottom: rect.bottom, left: rect.left, right: rect.right, height: rect.height };
            }
            function hits(a, b) {
                return a.left < b.right - 1 && a.right > b.left + 1 && a.top < b.bottom - 1 && a.bottom > b.top + 1;
            }
            const scroll = document.querySelector(".menu-scroll");
            const row = document.querySelector(".ability-row");
            const name = box(row.querySelector(".ability-name"));
            const chips = [...row.querySelectorAll(".slot")].map(box);
            scroll.scrollTop = scroll.scrollHeight;
            const dash = box([...scroll.querySelectorAll("button")].find((button) => button.textContent === "Dash"));
            const cancel = box(document.querySelector(".menu > .cancel"));
            return {
                scrolls: scroll.scrollHeight > scroll.clientHeight + 1,
                chips: chips.length,
                clear: chips.every((chip) => !hits(name, chip)),
                dashH: dash.height,
                cancelH: cancel.height,
                apart: dash.bottom <= cancel.top + 1,
            };
        }"""
    )
    assert metrics["scrolls"] is True
    assert metrics["chips"] == 3
    assert metrics["clear"] is True
    assert metrics["dashH"] >= 20
    assert metrics["cancelH"] >= 20
    assert metrics["apart"] is True
    assert board.hit_at(3, 3)["covered"] is False
