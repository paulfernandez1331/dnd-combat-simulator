"""Streamlit pages for the encounter simulator."""

from __future__ import annotations

import html
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.ai import LLMError, play_turn, write_report
from src.api_client import CatalogError, load_catalog, refresh_catalog
from src.bestiary import ability_mod
from src.ddb import PASTE, fetch_public_character, parse_character_id, public_character, sample_characters
from src.sheets import sheet_for
from src.combatant import Combatant, blank_abilities, from_monster, save_bonuses_for
from src.conditions import condition_labels
from src.eda import (
    action_economy_points,
    concentration_summary,
    damage_per_round,
    describe_fights,
    dropped_fraction,
    fight_factors,
    filter_run,
    headcount_sentence,
    latest_run_id,
    layout_summary,
    living_actions,
    load_runs,
    luck_summary,
    outlier_fights,
    pace_points,
    report_brief,
    split_win_rate,
    swing_summary,
    win_rate,
)
from src.encounter import Encounter, IllegalAction
from src.grid import footprint, in_bounds, span_for
from src.paths import ROOT, RUNS
from src.board import battle_payload, render_board
from src.presets import PLAY_HEIGHT, PLAY_WIDTH, bugbear_patrol, four_goblins
from src.targeting import (
    abilities_from,
    chosen_slot,
    match_creature,
    match_facing,
    match_instant,
    match_square,
    offer_weapons,
    preview_squares,
)

PAGES = ["Bestiary", "Encounter", "Combat", "Simulate", "Analysis"]


def main() -> None:
    st.set_page_config(page_title="D&D Encounter Simulator", layout="wide")
    st.sidebar.title("Encounter simulator")
    st.sidebar.caption("SRD data, one-click resolution, and repeated fights.")
    if st.session_state.get("goto"):
        st.session_state.page = st.session_state.pop("goto")
    page = st.sidebar.radio("Page", PAGES, key="page")
    catalog = _catalog()
    if page == "Bestiary":
        bestiary_page(catalog)
    elif page == "Encounter":
        encounter_page(catalog)
    elif page == "Combat":
        combat_page(catalog)
    elif page == "Simulate":
        simulate_page()
    else:
        analysis_page(catalog)


@st.cache_resource
def _load_cached_catalog():
    return load_catalog()


def _catalog():
    try:
        return _load_cached_catalog()
    except CatalogError as exc:
        st.sidebar.warning(str(exc))
        return None


def _encounter() -> Encounter | None:
    return st.session_state.get("encounter")


def _set_encounter(encounter: Encounter) -> None:
    st.session_state.encounter = encounter


def bestiary_page(catalog) -> None:
    st.header("Bestiary")
    st.write(
        "Monster and spell records come from the 2024 D&D SRD API "
        "(https://www.dnd5eapi.co/api/2024). The app stores the JSON on disk and "
        "resolves combat from that cache, so a fight does not call the API."
    )
    if st.button("Download or refresh the catalog", width="stretch"):
        with st.spinner("Checking the SRD catalog..."):
            try:
                counts = refresh_catalog()
            except CatalogError as exc:
                st.error(str(exc))
            else:
                _load_cached_catalog.clear()
                st.success(
                    f"New monsters: {counts['monsters']}. New spells: {counts['spells']}. "
                    f"Already on disk: {counts['skipped']}."
                )
                st.rerun()
    if catalog is None:
        st.info("Download the catalog to browse it. The sample fixtures are not a full catalog.")
        return
    counts = st.columns(2)
    counts[0].metric("Monsters", len(catalog.monsters))
    counts[1].metric("Spells", len(catalog.spells))
    kind = st.radio("Browse", ["Monsters", "Spells"], horizontal=True)
    query = st.text_input("Search", placeholder="goblin, fireball, cone...")
    if kind == "Monsters":
        names = [name for name, _index in catalog.monster_names() if query.lower() in name.lower()]
        choice = st.selectbox("Monster", names)
        index = next(index for name, index in catalog.monster_names() if name == choice)
        record = catalog.monsters[index]
        st.subheader(record.name)
        st.write(
            f"AC {record.ac}, HP {record.hp}, speed {record.speed} ft, "
            f"CR {record.challenge_rating}, type {record.creature_type}."
        )
        attack_rows = []
        for attack in record.attacks:
            attack_rows.append(
                {
                    "name": attack.name,
                    "resolution": attack.resolution,
                    "bonus": attack.attack_bonus,
                    "reach": attack.reach_feet,
                    "range": attack.range_feet,
                    "damage": ", ".join(f"{part['expr']} {part['type']}" for part in attack.damage),
                    "save": f"DC {attack.dc_value} {attack.dc_ability} ({attack.dc_success})" if attack.dc_value else "",
                    "area": f"{attack.area_size}-foot {attack.area_type}" if attack.area_type else "",
                    "recharge": attack.recharge_min,
                    "budget": attack.activation,
                }
            )
        st.write("Resolvable actions")
        attacks = pd.DataFrame(attack_rows)
        for column in ("bonus", "reach", "range", "recharge"):
            if column in attacks.columns:
                attacks[column] = pd.to_numeric(attacks[column], errors="coerce")
        st.dataframe(attacks, hide_index=True)
        if record.multiattack:
            st.write("Multiattack: " + ", ".join(f"{name} x{count}" for name, count in record.multiattack))
    else:
        names = [name for name, _index in catalog.spell_names() if query.lower() in name.lower()]
        choice = st.selectbox("Spell", names or ["(none)"])
        if not names:
            return
        index = next(index for name, index in catalog.spell_names() if name == choice)
        spell = catalog.spells[index]
        st.subheader(spell.name)
        st.write(
            f"Level {spell.level}. Classified as **{spell.kind}** ({spell.activation}). "
            f"Casting time {spell.casting_time}. Range {spell.range_mode} {spell.range_feet} ft. "
            f"Duration {spell.duration}."
        )
        if spell.area_type:
            st.write(f"Area: {spell.area_size}-foot {spell.area_type}.")
        if spell.damage:
            st.json(spell.damage)
        if spell.healing:
            st.write("Healing: " + ", ".join(f"slot {level}: {expr}" for level, expr in spell.healing.items()))


def encounter_page(catalog) -> None:
    st.header("Encounter")
    if catalog is None:
        st.info("Download the catalog on the Bestiary page before loading a preset.")
        return
    encounter = _encounter()
    if encounter is None:
        _set_encounter(Encounter([], width=PLAY_WIDTH, height=PLAY_HEIGHT, label="Custom encounter"))
        encounter = _encounter()
    if "encounter_label" not in st.session_state:
        st.session_state.encounter_label = encounter.label
    board_column, setup_column = st.columns([5, 2], vertical_alignment="top")
    with board_column:
        if encounter.combatants:
            labels = {
                creature.id: f"{creature.name} ({creature.side}) at {tuple(creature.position)}"
                for creature in encounter.combatants
            }
            if st.session_state.get("selected_id") not in labels:
                st.session_state.selected_id = next(iter(labels))
            spec = battle_payload(
                encounter,
                mode="place",
                actor=None,
                actions=[],
                abilities=[],
                armed=None,
                slot=None,
                selected_id=st.session_state.selected_id,
            )
            placed = _fresh_board_event(render_board(spec, key="place-board"), "place-board")
            if placed and placed.get("kind") == "square":
                if _place(encounter, (int(placed["x"]), int(placed["y"]))):
                    st.rerun()
        else:
            st.info("Add at least one party member and one enemy.")
    with setup_column:
        st.write(
            "Place a party and some monsters, then start the fight. "
            "Presets use the public sheets in data/samples. "
            "Opening squares are what a later simulation repeats."
        )
        if st.button("Load four goblin warriors", width="stretch"):
            try:
                loaded = four_goblins(catalog)
                _set_encounter(loaded)
                st.session_state.encounter_label = loaded.label
                st.session_state.dice_seed = int(loaded.seed)
                st.rerun()
            except CatalogError as exc:
                st.error(str(exc))
        if st.button("Load bugbear patrol", width="stretch"):
            try:
                loaded = bugbear_patrol(catalog)
                _set_encounter(loaded)
                st.session_state.encounter_label = loaded.label
                st.session_state.dice_seed = int(loaded.seed)
                st.rerun()
            except CatalogError as exc:
                st.error(str(exc))
        st.text_input("Encounter name", key="encounter_label")
        encounter.label = st.session_state.encounter_label
        _dice_seed_box(encounter)
        if encounter.combatants:
            labels = {
                creature.id: f"{creature.name} ({creature.side}) at {tuple(creature.position)}"
                for creature in encounter.combatants
            }
            selected_id = st.selectbox(
                "Selected creature",
                list(labels),
                format_func=lambda creature_id: labels[creature_id],
                key="selected_id",
            )
            selected = encounter.by_id(selected_id)
            party = sum(1 for creature in encounter.combatants if creature.side == "party")
            enemy = sum(1 for creature in encounter.combatants if creature.side == "enemy")
            if st.button("Restore opening layout", width="stretch", disabled=not encounter.started):
                encounter.restore_opening_layout()
                st.rerun()
            if st.button("Remove selected", width="stretch"):
                encounter.combatants = [creature for creature in encounter.combatants if creature.id != selected.id]
                encounter.blueprint = None
                st.rerun()
            ready = party > 0 and enemy > 0 and not _overlap(encounter)
            if st.button("Start fight", type="primary", width="stretch", disabled=not ready):
                encounter.start()
                st.session_state.goto = "Combat"
                st.rerun()
            if _overlap(encounter):
                st.error("Two creatures share a square. Move one of them.")
        else:
            selected = None
        samples = sample_characters()
        with st.expander("Add a sample character"):
            if not samples:
                st.caption("Put a public character JSON file in data/samples.")
            else:
                picked = st.selectbox(
                    "Sample character",
                    [stem for stem, _character in samples],
                    format_func=lambda stem: next(
                        character.get("name") or item for item, character in samples if item == stem
                    ),
                )
                if st.button("Add sample", width="stretch"):
                    current = _encounter()
                    if current is None:
                        current = Encounter([], width=PLAY_WIDTH, height=PLAY_HEIGHT, label="Custom encounter")
                        _set_encounter(current)
                    if current.started:
                        st.warning("Restore the opening layout before changing who is in the fight.")
                    else:
                        try:
                            _append_sample(current, catalog, picked)
                        except (CatalogError, ValueError) as exc:
                            st.error(str(exc))
                        else:
                            st.rerun()
        with st.expander("Add a public D&D Beyond character"):
            st.caption(
                "Paste the public character link or the numeric id. "
                "D&D Beyond will not let this page embed the sheet, so the stat block is drawn here "
                "and the link opens the real page."
            )
            st.text_input("Character link or id", key="ddb_character")
            if st.button("Import character", width="stretch"):
                current = _encounter()
                if current is None:
                    current = Encounter([], width=PLAY_WIDTH, height=PLAY_HEIGHT, label="Custom encounter")
                    _set_encounter(current)
                if current.started:
                    st.warning("Restore the opening layout before changing who is in the fight.")
                else:
                    try:
                        _import_character(current, catalog, st.session_state.get("ddb_character") or "")
                    except (CatalogError, ValueError) as exc:
                        st.error(str(exc))
                    else:
                        st.rerun()
        _add_creatures(catalog, encounter)
    if not encounter.combatants:
        return
    roster_column, sheet_column = st.columns([3, 2], vertical_alignment="top")
    with roster_column:
        _roster_table(encounter)
    with sheet_column:
        _render_sheet(selected, catalog)


def _add_creatures(catalog, encounter: Encounter) -> None:
    with st.expander("Add a monster from the API"):
        names = catalog.monster_names()
        label = st.selectbox("Monster record", [name for name, _index in names])
        index = next(item for name, item in names if name == label)
        count = st.number_input("How many", min_value=1, max_value=8, value=1, step=1)
        side = st.selectbox("Side", ["enemy", "party"], key="monster_side")
        if st.button("Add monster", width="stretch"):
            if encounter.started:
                st.warning("Restore the opening layout before changing who is in the fight.")
            else:
                record = catalog.monsters[index]
                span = span_for(record.size)
                added = False
                for _ in range(int(count)):
                    number = 1 + sum(1 for creature in encounter.combatants if creature.creature_index == index)
                    creature_id = f"{index}-{number}"
                    position = _free_square(encounter, span)
                    if not _anchor_fits(encounter, position, span):
                        st.warning(f"{record.name} does not fit on the mat.")
                        break
                    encounter.combatants.append(
                        from_monster(
                            record,
                            creature_id=creature_id,
                            name=f"{record.name} {number}",
                            side=side,
                            position=position,
                        )
                    )
                    added = True
                if added:
                    st.rerun()
    with st.expander("Add a custom creature"):
        name = st.text_input("Name", value="Cleric")
        side = st.selectbox("Custom side", ["party", "enemy"])
        ac = st.number_input("Armor class", min_value=1, max_value=30, value=16)
        hp = st.number_input("Hit points", min_value=1, max_value=400, value=28)
        speed = st.number_input("Speed (feet)", min_value=0, max_value=120, value=30, step=5)
        size = st.selectbox("Size", ["Tiny", "Small", "Medium", "Large", "Huge", "Gargantuan"], index=2)
        bonus = st.number_input("Weapon attack bonus", min_value=0, max_value=15, value=5)
        damage = st.text_input("Weapon damage", value="1d8+3")
        spell_names = catalog.spell_names(resolvable_only=True)
        chosen = st.multiselect("Spells the engine can cast", [name for name, _index in spell_names])
        dc = st.number_input("Spell save DC", min_value=8, max_value=22, value=13)
        spell_bonus = st.number_input("Spell attack bonus", min_value=0, max_value=15, value=5)
        caster_level = st.number_input("Caster level", min_value=1, max_value=20, value=5)
        slot_text = st.text_input("Spell slots", value="1:4, 2:3, 3:2", help="Level:count pairs. Cantrips do not use slots.")
        if st.button("Add custom creature", width="stretch"):
            if encounter.started:
                st.warning("Restore the opening layout before changing who is in the fight.")
            else:
                try:
                    _append_custom(
                        encounter, catalog, name, side, int(ac), int(hp), int(speed),
                        int(bonus), damage, chosen, int(dc), int(spell_bonus), int(caster_level), slot_text,
                        size,
                    )
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    st.rerun()


def _append_custom(encounter, catalog, name, side, ac, hp, speed, bonus, damage, chosen, dc, spell_bonus, caster_level, slot_text, size="Medium"):
    from src.bestiary import AttackOption
    from src.dice import parse_dice

    cleaned = damage.replace(" ", "")
    if cleaned:
        try:
            parse_dice(cleaned)
        except ValueError as exc:
            raise ValueError("Weapon damage must look like 1d8+3.") from exc
    abilities = blank_abilities({"strength": 14, "dexterity": 14, "constitution": 14, "intelligence": 14, "wisdom": 14, "charisma": 10})
    slots = _parse_slots(slot_text)
    spells = [catalog.spells[index] for label, index in catalog.spell_names() if label in chosen]
    span = span_for(size)
    anchor = _free_square(encounter, span)
    if not _anchor_fits(encounter, anchor, span):
        raise ValueError(f"A {size} creature does not fit on the mat.")
    creature_id = f"custom-{1 + sum(1 for creature in encounter.combatants if creature.id.startswith('custom-'))}"
    encounter.combatants.append(
        Combatant(
            id=creature_id,
            name=name or "Custom",
            side=side,
            ac=ac,
            max_hp=hp,
            hp=hp,
            speed=speed,
            abilities=abilities,
            save_bonuses=save_bonuses_for(abilities, ["constitution", "wisdom"], 2),
            position=anchor,
            size=str(size or "Medium"),
            attacks=[
                AttackOption(
                    name="Weapon",
                    resolution="attack",
                    attack_bonus=bonus,
                    reach_feet=5,
                    damage=[{"type": "slashing", "expr": cleaned}],
                )
            ] if cleaned else [],
            spells=spells,
            spell_attack_bonus=spell_bonus,
            spell_save_dc=dc,
            spellcasting_mod=ability_mod(14),
            caster_level=caster_level,
            slot_max=slots,
            slots=dict(slots),
        )
    )


def _parse_slots(text: str) -> dict[int, int]:
    slots = {}
    for piece in text.split(","):
        if ":" not in piece:
            continue
        level, count = piece.split(":", 1)
        try:
            slots[int(level.strip())] = int(count.strip())
        except ValueError:
            continue
    return slots


def _replace_character(encounter: Encounter, creature) -> None:
    ddb_id = creature.traits.get("ddb_id")
    encounter.combatants = [
        existing
        for existing in encounter.combatants
        if existing.id != creature.id and (not ddb_id or existing.traits.get("ddb_id") != ddb_id)
    ]
    encounter.combatants.append(creature)
    encounter.blueprint = None


def _import_character(encounter: Encounter, catalog, pasted: str) -> None:
    character_id = parse_character_id(pasted)
    if character_id is None:
        raise CatalogError(PASTE)
    character = fetch_public_character(character_id)
    creature = public_character(character, catalog, _free_square(encounter), f"ddb-{character_id}")
    _replace_character(encounter, creature)


def _append_sample(encounter: Encounter, catalog, stem: str) -> None:
    character = next(character for item, character in sample_characters() if item == stem)
    creature_id = f"ddb-{int(character.get('id') or stem)}"
    creature = public_character(character, catalog, _free_square(encounter), creature_id)
    _replace_character(encounter, creature)


def _render_sheet(creature, catalog) -> None:
    sheet = sheet_for(creature, catalog)
    st.markdown(f"**{sheet.title}**")
    st.markdown(sheet.kind_label)
    for line in sheet.lines:
        st.markdown(line)
    if sheet.url:
        st.markdown(f"[Open on D&D Beyond]({sheet.url})")


def _anchor_fits(encounter: Encounter, anchor: tuple[int, int], span: int, ignore_id: str | None = None) -> bool:
    taken: set[tuple[int, int]] = set()
    for creature in encounter.combatants:
        if creature.id == ignore_id:
            continue
        taken.update(creature.squares())
    for cell in footprint(anchor, span):
        if not in_bounds(cell, encounter.width, encounter.height) or cell in taken:
            return False
    return True


def _free_square(encounter: Encounter, span: int = 1) -> tuple[int, int]:
    for y in range(encounter.height):
        for x in range(encounter.width):
            if _anchor_fits(encounter, (x, y), span):
                return (x, y)
    return (0, 0)


def _overlap(encounter: Encounter) -> bool:
    seen: set[tuple[int, int]] = set()
    for creature in encounter.combatants:
        body = creature.squares()
        if any(not in_bounds(cell, encounter.width, encounter.height) for cell in body):
            return True
        if seen & body:
            return True
        seen.update(body)
    return False


def _place(encounter: Encounter, square: tuple[int, int]) -> bool:
    occupant = next((creature for creature in encounter.combatants if square in creature.squares()), None)
    if occupant:
        if st.session_state.get("selected_id") == occupant.id:
            return False
        st.session_state.selected_id = occupant.id
        return True
    selected = next((creature for creature in encounter.combatants if creature.id == st.session_state.get("selected_id")), None)
    if selected is None or tuple(selected.position) == square:
        return False
    if encounter.started:
        st.warning("Restore the opening layout before moving tokens for a new fight.")
        return False
    if not _anchor_fits(encounter, square, selected.span, ignore_id=selected.id):
        st.warning("That creature does not fit there.")
        return False
    selected.position = square
    encounter.blueprint = None
    return True


def _roster_frame(encounter: Encounter) -> pd.DataFrame:
    rows = []
    for creature in encounter.combatants:
        rows.append(
            {
                "name": creature.name,
                "side": creature.side,
                "AC": creature.ac,
                "HP": creature.max_hp,
                "speed": creature.speed,
                "position": f"({creature.position[0]}, {creature.position[1]})",
                "attacks": ", ".join(attack.name for attack in creature.attacks) or "—",
                "spells": ", ".join(spell.name for spell in creature.spells) or "—",
                "CR": creature.challenge_rating,
            }
        )
    frame = pd.DataFrame(rows)
    frame["CR"] = pd.to_numeric(frame["CR"], errors="coerce")
    return frame


def _roster_table(encounter: Encounter) -> None:
    st.dataframe(_roster_frame(encounter), hide_index=True)


def _fresh_board_event(event, key: str) -> dict | None:
    """Ignore the component value Streamlit repeats after the click was used."""
    if not isinstance(event, dict) or "id" not in event:
        return None
    stamp = f"{key}:{event['id']}"
    if st.session_state.get("board_event_id") == stamp:
        return None
    st.session_state.board_event_id = stamp
    return event


def combat_page(catalog=None) -> None:
    st.header("Combat")
    encounter = _encounter()
    if encounter is None or not encounter.combatants:
        st.info("Build an encounter first.")
        return
    if not encounter.started:
        st.info("The layout is set. Start the fight from the Encounter page.")
        return
    if encounter.over:
        st.success(f"Fight over. Winner: {encounter.winner}.")
        if st.button("Fight this layout again", width="stretch"):
            _clear_arm()
            encounter.start()
            st.rerun()
        _log(encounter)
        return
    board_column, panel = st.columns([5, 2], vertical_alignment="top")
    with panel:
        _dice_seed_box(encounter)
        if st.button("Restart with this seed", width="stretch"):
            _clear_arm()
            encounter.start()
            st.rerun()
        if not encounter.pending:
            encounter.begin_turn()
        actor = encounter.current()
        waiting = encounter.pending is not None
        if waiting:
            actions = []
            abilities = []
            armed = None
            slot = None
            window = encounter.pending or {}
            choice = window.get("phase") == "rider" or window.get("spends_reaction") is False
            waiting_line = "A choice is waiting." if choice else "A reaction is waiting."
            st.markdown(f"**Round {min(encounter.round, encounter.round_cap)}.** {waiting_line}")
        else:
            _sync_arm(actor.id)
            actions = encounter.legal_actions(actor, automated=False)
            abilities = offer_weapons(encounter, actor, abilities_from(actions))
            armed = next((item for item in abilities if item.key == st.session_state.get("combat_ability")), None)
            slot = chosen_slot(armed, st.session_state.get("combat_slot")) if armed else None
            _turn_strip(encounter, actions)
        with st.expander("How this page works"):
            st.write(
                "The gold ring is the creature whose turn it is. Click a green square to move. "
                "Open Action or Bonus beside that token. Attack lists every weapon, even when nobody is in reach yet. "
                "Choosing an attack closes that list and names each creature you can reach. "
                "Then click a creature or a square. An area follows the pointer, and the click casts it. "
                "End turn passes initiative. Let the model choose, or Use the heuristic instead, plays the whole current turn "
                "for a party member or an enemy. Shield, Counterspell, Cutting Words, and a feat reaction pause the fight and ask. "
                "The heuristic answers those in a batch, and for the side whose turn is being played automatically. "
                "The model is not asked. The sheet beside the mat is the current creature's stat block."
            )
        if waiting:
            _reaction_prompt(encounter)
        else:
            _actor_card(actor)
            _render_sheet(actor, catalog)
            turn_buttons = st.columns(2)
            if turn_buttons[0].button("Let the model choose", type="primary", width="stretch"):
                _play_current_turn(encounter, "llm")
            if turn_buttons[1].button("Use the heuristic instead", width="stretch"):
                _play_current_turn(encounter, "heuristic")
        _log(encounter)
    with board_column:
        spec = battle_payload(
            encounter,
            mode="combat",
            actor=actor,
            actions=actions,
            abilities=abilities,
            armed=armed,
            slot=slot,
            selected_id=actor.id,
        )
        event = _fresh_board_event(render_board(spec, key="combat-board"), "combat-board")
        if event and not waiting:
            _apply_board_event(encounter, actor, actions, abilities, armed, slot, event)


def _reaction_prompt(encounter: Encounter) -> None:
    window = encounter.pending or {}
    reactor = encounter.by_id(window["reactor_id"])
    choice = window.get("phase") == "rider" or window.get("spends_reaction") is False
    st.subheader("Choice" if choice else "Reaction")
    st.write(window.get("prompt") or "A reaction is available.")
    if choice:
        st.caption(f"{reactor.name} can choose.")
        decline = str(window.get("decline") or "Don't")
    else:
        st.caption(f"{reactor.name} can use one reaction.")
        decline = "Don't use a reaction"
    for offer in window.get("offers") or []:
        if st.button(offer["label"], key=f"reaction-{offer['id']}", width="stretch"):
            _answer_pending(encounter, offer["id"])
    if st.button(decline, key="reaction-decline", width="stretch"):
        _answer_pending(encounter, None)


def _answer_pending(encounter: Encounter, choice: str | None) -> None:
    try:
        encounter.answer_reaction(choice)
        if not encounter.pending and encounter.auto_policy:
            play_turn(encounter, encounter.auto_policy)
    except (IllegalAction, LLMError) as exc:
        st.error(str(exc))
    else:
        st.rerun()


def _turn_strip(encounter: Encounter, actions) -> None:
    current_id = encounter.order[encounter.turn_index]
    bits = []
    for creature_id in encounter.order:
        creature = encounter.by_id(creature_id)
        mark = "now" if creature_id == current_id else creature.side
        labels = condition_labels(creature)
        status = f" {', '.join(labels)}" if labels else ""
        bits.append(f"{creature.name} {creature.hp}/{creature.max_hp} ({mark}){status}")
    st.markdown(f"**Round {min(encounter.round, encounter.round_cap)}.** " + " · ".join(bits))
    if st.button("End turn", type="primary", key="end-turn", width="stretch"):
        ending = next((action for action in actions if action.kind == "end"), None)
        _resolve(encounter, ending)


def _sync_arm(actor_id: str) -> None:
    if st.session_state.get("combat_actor") != actor_id:
        st.session_state.combat_actor = actor_id
        _clear_arm()


def _clear_arm() -> None:
    st.session_state.combat_ability = None
    st.session_state.combat_slot = None
    st.session_state.combat_aim = None


def _actor_card(actor) -> None:
    st.subheader(actor.name)
    ratio = 0.0 if actor.max_hp <= 0 else max(0.0, min(1.0, actor.hp / actor.max_hp))
    st.progress(ratio, text=f"{actor.hp}/{actor.max_hp} HP")
    movement = actor.movement_left if actor.movement_left is not None else actor.speed
    action = "ready" if actor.action_available else "spent"
    bonus = "ready" if actor.bonus_available else "spent"
    reaction = "ready" if actor.reaction_available else "spent"
    shield = int(actor.traits.get("shield_ac") or 0)
    ac_text = f"AC {actor.ac + shield}" if shield else f"AC {actor.ac}"
    shield_note = f" · Shield +{shield}" if shield else ""
    st.caption(
        f"{actor.side} · {ac_text}{shield_note} · square {tuple(actor.position)} · "
        f"movement {movement} ft · action {action} · bonus {bonus} · reaction {reaction}"
    )
    labels = condition_labels(actor)
    if labels:
        st.caption("Conditions: " + ", ".join(labels))
    if actor.temp_hp:
        st.write(f"Temporary hit points: {actor.temp_hp}.")
    class_name = actor.traits.get("class_name")
    if class_name:
        subclass = actor.traits.get("subclass")
        st.write(f"{class_name}" + (f", {subclass}." if subclass else "."))
    if actor.uses_max:
        bits = []
        for name, maximum in actor.uses_max.items():
            bits.append(f"{name} {actor.uses_left.get(name, 0)}/{maximum}")
        st.write("Uses: " + ", ".join(bits) + ".")
    excluded = actor.traits.get("excluded_features") or []
    if excluded:
        st.write("Not rolled this fight: " + ", ".join(str(name) for name in excluded) + ".")
    if actor.spells:
        st.write("Prepared spells: " + ", ".join(spell.name for spell in actor.spells) + ".")
    if actor.concentrating_on:
        st.write(f"Concentrating on {actor.concentrating_on}.")
    if actor.recharging:
        st.write("Recharging: " + ", ".join(actor.recharging))


def _apply_board_event(encounter, actor, actions, abilities, armed, slot, event: dict) -> None:
    kind = event.get("kind")
    if kind == "clear":
        _clear_arm()
        st.rerun()
        return
    if kind == "ability":
        ability = next((item for item in abilities if item.key == event.get("key")), None)
        if ability is None:
            return
        picked = event.get("slot")
        picked = int(picked) if isinstance(picked, (int, float)) else None
        same = armed is not None and armed.key == ability.key and chosen_slot(ability, picked) == slot
        if same:
            _clear_arm()
            st.rerun()
            return
        if ability.needs == "none":
            _resolve(encounter, match_instant(actions, ability, None))
            return
        st.session_state.combat_ability = ability.key
        st.session_state.combat_slot = chosen_slot(ability, picked)
        st.session_state.combat_aim = None
        st.rerun()
        return
    if kind != "square":
        return
    square = (int(event["x"]), int(event["y"]))
    if armed is None:
        move = next((action for action in actions if action.kind == "move" and action.destination == square), None)
        if move is not None:
            _resolve(encounter, move)
        return
    if armed.needs == "creature":
        occupant = next(
            (creature for creature in encounter.combatants if creature.hp > 0 and square in creature.squares()),
            None,
        )
        action = match_creature(actions, armed, slot, occupant.id) if occupant is not None else None
        if action is None:
            _clear_arm()
            st.rerun()
            return
        _resolve(encounter, action)
        return
    if armed.needs == "square":
        action = match_square(actions, armed, slot, square)
        if action is None:
            _clear_arm()
            st.rerun()
            return
        _resolve(encounter, action)
        return
    if armed.needs == "facing":
        action = match_facing(actions, armed, slot, tuple(actor.position), square)
        if action is None:
            _clear_arm()
            st.rerun()
            return
        _resolve(encounter, action)
        return
    if armed.needs == "confirm":
        action = match_instant(actions, armed, slot)
        cells = preview_squares(encounter, actor, action) if action is not None else set()
        if action is not None and square in cells:
            _resolve(encounter, action)
            return
        _clear_arm()
        st.rerun()


def _resolve(encounter, action) -> None:
    if action is None:
        return
    try:
        encounter.apply(action, "player")
    except IllegalAction as exc:
        st.error(str(exc))
    else:
        _clear_arm()
        st.rerun()


def _play_current_turn(encounter, policy: str) -> None:
    try:
        play_turn(encounter, policy)
    except LLMError as exc:
        st.error(str(exc))
    except IllegalAction as exc:
        st.error(str(exc))
    else:
        st.rerun()


def _dice_seed_box(encounter: Encounter) -> None:
    if "dice_seed" not in st.session_state:
        st.session_state.dice_seed = int(encounter.seed)
    seed = st.number_input(
        "Dice seed",
        min_value=0,
        max_value=1_000_000,
        step=1,
        key="dice_seed",
        help="Rolls that have not happened yet use this number. Restart with this seed rerolls initiative too.",
    )
    encounter.use_seed(int(seed))


def _log(encounter: Encounter) -> None:
    st.subheader("Combat log")
    if not encounter.log:
        st.caption("No rolls yet.")
        return
    st.caption(
        "Oldest at the top. On a critical hit, every doubled die is listed and flat bonuses are added once. "
        "Divine Fury is highlighted."
    )
    st.markdown(_log_html(encounter.log), unsafe_allow_html=True)
    with st.expander("Full log"):
        st.code("\n\n".join(encounter.log))


def _log_html(entries: list[str]) -> str:
    cards = []
    for entry in entries:
        kind = _log_kind(entry)
        border, wash = _LOG_COLORS[kind]
        rows = [_log_row(row, index == 0, kind) for index, row in enumerate(entry.split("\n")) if row]
        cards.append(
            "<div style=\"margin:0 0 8px;padding:8px 10px;border-radius:8px;"
            f"background:{wash};border-left:4px solid {border};line-height:1.45\">"
            + "".join(rows)
            + "</div>"
        )
    return (
        "<div style=\"max-height:440px;overflow:auto;padding:8px;border:1px solid #4a3b28;"
        "border-radius:8px;background:#1a1410\">"
        + "".join(cards)
        + "</div>"
    )


_LOG_COLORS = {
    "crit": ("#e0a84a", "#3a2a16"),
    "hit": ("#d0af70", "#2a2118"),
    "miss": ("#6e6256", "#241c16"),
    "save": ("#c47b4a", "#2e2218"),
    "saved": ("#3f8f4e", "#1c2820"),
    "heal": ("#3f8f4e", "#1c2820"),
    "meta": ("#3d8f8a", "#1a2424"),
    "turn": ("#4a3b28", "#211a15"),
    "note": ("#8c6a32", "#2a2118"),
}


def _log_kind(entry: str) -> str:
    first = entry.split("\n", 1)[0]
    if first.startswith("Critical hit"):
        return "crit"
    if first.startswith("Hit"):
        return "hit"
    if first.startswith("Miss"):
        return "miss"
    if first.startswith("Failed save"):
        return "save"
    if first.startswith("Saved"):
        return "saved"
    if "regains" in first or "uses " in first:
        return "heal"
    if first.startswith(("Initiative", "Dice seed", "Fight over")) or "round cap" in first:
        return "meta"
    if "ends the turn" in first:
        return "turn"
    return "note"


def _log_row(row: str, lead: bool, kind: str) -> str:
    safe = html.escape(row)
    if "Divine Fury" in row:
        return (
            "<div style=\"margin-top:6px;padding:4px 8px;border-radius:6px;"
            "background:#4a3718;color:#ffe7b0;font-weight:650\">"
            f"{safe}</div>"
        )
    color = "#fff8ea" if lead and kind == "crit" else ("#f4efe4" if lead else "#e7dccb")
    weight = "650" if lead else "450"
    gap = "0" if lead else "3px"
    return f"<div style=\"margin-top:{gap};font-weight:{weight};color:{color}\">{safe}</div>"


def simulate_page() -> None:
    st.header("Simulate")
    st.write(
        "The same opening layout is fought again from a dice seed. "
        "The party uses the heuristic. Enemies use the model, or the heuristic if you want a fast comparison cohort. "
        "Each finished fight is written to disk immediately, so a stopped run still counts."
    )
    encounter = _encounter()
    if encounter is None or not encounter.combatants:
        st.info("Build an encounter first.")
        return
    party = sum(1 for creature in encounter.combatants if creature.side == "party")
    enemies = sum(1 for creature in encounter.combatants if creature.side == "enemy")
    if party == 0 or enemies == 0:
        st.info("The layout needs a party and at least one enemy.")
        return
    count = st.slider("Fights", min_value=1, max_value=100, value=30)
    policy = st.radio("Enemy policy", ["llm", "heuristic"], horizontal=True)
    if "sim_seed" not in st.session_state:
        st.session_state.sim_seed = int(encounter.seed)
    seed = st.number_input("First seed", min_value=0, max_value=1_000_000, step=1, key="sim_seed")
    if policy == "llm":
        st.warning(
            f"A rough upper bound is {enemies * 4 * 6 * count} model calls "
            f"({enemies} enemies, about 4 rounds, up to 6 choices on a turn, {count} fights). "
            "Calls stop early when a side is defeated. This spends API credit and can take a long time."
        )
    else:
        st.caption("Heuristic fights are local and repeatable from the seed.")
    if st.button("Start simulation", type="primary", width="stretch"):
        _launch_sim(encounter, int(count), policy, int(seed))
    _watch_sim()


def _launch_sim(encounter: Encounter, count: int, policy: str, seed: int) -> None:
    run_dir = RUNS / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    setup = encounter.layout_setup()
    setup["enemy_policy"] = policy
    setup_path = run_dir / "setup.json"
    setup_path.write_text(json.dumps(setup), encoding="utf-8")
    log_handle = (run_dir / "worker.log").open("w", encoding="utf-8")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "src.worker",
            "--setup",
            str(setup_path),
            "--run-dir",
            str(run_dir),
            "--count",
            str(count),
            "--policy",
            policy,
            "--seed",
            str(seed),
        ],
        cwd=str(ROOT),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    st.session_state.sim_proc = process
    st.session_state.sim_log = log_handle
    st.session_state.sim_dir = str(run_dir)


def _watch_sim() -> None:
    run_dir = st.session_state.get("sim_dir")
    if not run_dir:
        return
    path = Path(run_dir)
    status_path = path / "status.json"
    if st.button("Stop after the current turn", width="stretch"):
        (path / "STOP").write_text("stop", encoding="utf-8")
    status = {}
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
    completed = int(status.get("completed") or 0)
    total = int(status.get("total") or 0)
    state = status.get("state") or "starting"
    if total:
        st.progress(min(1.0, completed / total), text=f"{completed} of {total} fights · {state}")
    else:
        st.write(f"Simulation {state}.")
    process = st.session_state.get("sim_proc")
    if process is not None and process.poll() is not None and state == "running":
        st.error("The simulation process stopped. See the worker log below.")
    log_path = path / "worker.log"
    if log_path.exists():
        text = log_path.read_text(encoding="utf-8", errors="replace").strip()
        if text:
            st.code(text[-2000:])
    if state in {"running", "starting"} and process is not None and process.poll() is None:
        st.caption("This page does not refresh on its own. Click the button to update the count.")
        if st.button("Refresh progress", width="stretch"):
            st.rerun()


def _percent(value) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.0%}"


def _chart(figure, alt: str) -> None:
    st.plotly_chart(figure, width="stretch", alt=alt)


def _split_sentence(frame: pd.DataFrame, labels: dict[str, str]) -> str:
    if frame.empty:
        return "None of these fights recorded it."
    parts = []
    for row in frame.itertuples(index=False):
        count = int(row.fights)
        noun = "fight" if count == 1 else "fights"
        parts.append(f"{labels.get(row.group, row.group)} in {count} {noun} ({_percent(row.party_win)} party wins)")
    return "; ".join(parts) + "."


def _win_bars(frame: pd.DataFrame, title: str, alt: str, labels: dict[str, str]) -> None:
    if frame.empty or "group" not in frame.columns:
        return
    shown = frame.dropna(subset=["party_win"]).copy()
    if shown.empty or shown["group"].nunique() < 2:
        return
    shown["group"] = shown["group"].map(lambda group: labels.get(group, group))
    figure = px.bar(shown, x="group", y="party_win", range_y=[0, 1], title=title, text="fights")
    figure.update_layout(yaxis_title="Party win rate", xaxis_title="", showlegend=False)
    _chart(figure, alt)


def _batch_caption(fights: pd.DataFrame, run_id: str | None) -> str:
    if run_id is None:
        runs = int(fights["run_id"].nunique()) if "run_id" in fights.columns else 1
        return f"Showing every saved run: {runs} runs, {len(fights)} fights."
    labels = []
    if "label" in fights.columns:
        labels = sorted({str(label) for label in fights["label"].dropna().unique()})
    policies = []
    if "policy" in fights.columns:
        policies = sorted({str(policy) for policy in fights["policy"].dropna().unique()})
    label = ", ".join(labels) if labels else "unlabeled"
    policy = ", ".join(policies) if policies else "unspecified policy"
    return f"Showing {run_id}: {len(fights)} fights, {label}, {policy}."


def analysis_page(catalog) -> None:
    st.header("Analysis")
    st.write(
        "This page starts on the latest Simulate batch. "
        "The charts are the factors that swing a fight: who went first, who scored the first knockout, "
        "how many turns each side took, and whether concentration held."
    )
    latest_only = st.toggle(
        "Last batch only",
        value=True,
        key="analysis_last_batch",
        help="Use only the latest Simulate batch. Turn this off to pool every saved run.",
    )
    session = st.session_state.get("sim_dir")
    if latest_only and session and not (Path(session) / "fights.csv").is_file():
        st.info(f"{Path(session).name} has not written a fight yet.")
        return
    fights, turns = load_runs(RUNS)
    run_id = None
    if latest_only:
        run_id = latest_run_id(RUNS, session if session and (Path(session) / "fights.csv").is_file() else None)
        if run_id is None or fights.empty:
            st.info("No fights saved yet. Run a heuristic batch on the Simulate page.")
            return
        fights, turns = filter_run(fights, turns, run_id)
        if fights.empty:
            st.info(f"{run_id} has not written a fight yet.")
            return
    elif fights.empty:
        st.info("No fights saved yet. Run a heuristic batch on the Simulate page.")
        return
    st.caption(_batch_caption(fights, run_id))
    factors = fight_factors(fights, turns)
    rate = win_rate(fights)
    pace = damage_per_round(fights, "damage_by_party")
    mean_rounds = pd.to_numeric(fights["rounds"], errors="coerce").mean() if "rounds" in fights.columns else None
    with st.container(horizontal=True):
        st.metric("Fights", len(fights), border=True)
        st.metric("Party win rate", _percent(rate), border=True)
        st.metric("Mean rounds", "—" if mean_rounds is None or pd.isna(mean_rounds) else f"{mean_rounds:.1f}", border=True)
        st.metric("Party damage per round", "—" if pace is None else f"{pace:.1f}", border=True)

    luck = luck_summary(turns, fights)
    with st.container(border=True):
        st.subheader("Luck")
        if not luck["d20_count"] and not luck["damage_count"]:
            st.caption("A fair d20 averages 10.5. A fair damage roll sits at the 50th percentile of its own dice.")
            st.write("This batch has no logged dice.")
        else:
            st.caption(
                "A fair d20 averages 10.5. A fair damage roll sits at the 50th percentile of its own dice. "
                "Modifiers are excluded. A save belongs to the creature who rolled it. "
                f"{luck['d20_count']} d20 faces and {luck['damage_count']} damage rolls."
            )
            with st.container(horizontal=True):
                st.metric(
                    "Mean d20",
                    "—" if luck["d20_mean"] is None else f"{luck['d20_mean']:.2f}",
                    border=True,
                )
                st.metric(
                    "Damage percentile",
                    "—" if luck["damage_percentile"] is None else f"{luck['damage_percentile']:.1f}",
                    border=True,
                )
            groups = luck["groups"]
            if not groups.empty:
                shown = groups.copy()
                shown["d20_mean"] = shown["d20_mean"].map(lambda value: "—" if pd.isna(value) else f"{float(value):.2f}")
                shown["damage_percentile"] = shown["damage_percentile"].map(
                    lambda value: "—" if pd.isna(value) else f"{float(value):.1f}"
                )
                shown = shown.rename(columns={
                    "side": "Side",
                    "winner": "Winner",
                    "d20_mean": "Mean d20",
                    "d20_count": "d20 faces",
                    "damage_percentile": "Damage percentile",
                    "damage_count": "Damage rolls",
                })
                st.dataframe(
                    shown,
                    hide_index=True,
                    alt="Mean d20 and damage percentile for each side and the winner of the fight",
                )

    first_labels = {"party": "Party went first", "enemy": "Enemy went first"}
    blood_labels = {"party": "Party scored the first knockout", "enemy": "Enemy scored the first knockout"}
    first = split_win_rate(fights, factors, "first_side")
    blood = split_win_rate(fights, factors, "first_blood_scorer")
    with st.container(border=True):
        st.subheader("Initiative")
        st.caption("The first turn in the log is the creature who won initiative.")
        st.write(_split_sentence(first, first_labels))
        _win_bars(first, "Party win rate by who went first", "Party win rate when the party or an enemy took the first turn", first_labels)
    with st.container(border=True):
        st.subheader("First blood")
        st.caption("The side that did not lose the first creature scored it, including damage recorded on the victim's own turn.")
        st.write(_split_sentence(blood, blood_labels))
        _win_bars(blood, "Party win rate by who scored the first knockout", "Party win rate by which side scored the first knockout", blood_labels)

    economy = action_economy_points(fights, factors)
    with st.container(border=True):
        st.subheader("Action economy")
        st.caption("Turns taken skip End and skipped turns. Attacks and spells are the hostile actions.")
        opening = headcount_sentence(fights)
        if opening:
            st.write(opening)
        if not economy.empty:
            figure = px.box(
                economy,
                x="measure",
                y="count",
                color="winner",
                points="outliers",
                title="Actions per fight",
            )
            figure.update_layout(xaxis_title="", yaxis_title="Count per fight")
            figure.update_xaxes(categoryorder="array", categoryarray=[
                "Party actions",
                "Enemy actions",
                "Party attacks and spells",
                "Enemy attacks and spells",
            ])
            _chart(figure, "Actions and attacks per fight for each side, split by who won")

    swing = swing_summary(factors)
    with st.container(border=True):
        st.subheader("One swing")
        st.caption("An area spell can remove several creatures, and their turns, before they act.")
        st.write(
            f"One action dropped two or more creatures in {swing['area_fights']} of {swing['fights']} fights. "
            f"The largest swing dropped {swing['largest']}."
        )

    conc = concentration_summary(fights, factors)
    with st.container(border=True):
        st.subheader("Concentration")
        st.caption("A failed save or a knockout drops the spell. Holding it keeps the area up for later turns.")
        if conc["fails"] or conc["holds"] or conc["drops"]:
            with st.container(horizontal=True):
                st.metric("Saves failed", conc["fails"], border=True)
                st.metric("Saves held", conc["holds"], border=True)
                st.metric("Lost on a knockout", conc["drops"], border=True)
            st.write(
                f"Broke concentration in {conc['broke_fights']} fights ({_percent(conc['broke_win_rate'])} party wins). "
                f"Held it without a break in {conc['held_fights']} ({_percent(conc['held_win_rate'])} party wins)."
            )
            groups = pd.DataFrame(
                [
                    {"group": "Broke", "fights": conc["broke_fights"], "party_win": conc["broke_win_rate"]},
                    {"group": "Held", "fights": conc["held_fights"], "party_win": conc["held_win_rate"]},
                ]
            )
            groups = groups[groups["fights"] > 0]
            _win_bars(groups, "Party win rate when concentration broke or held", "Party win rate for fights that broke concentration and fights that held it", {})
        else:
            st.write("No concentration save or lost concentration is in this batch.")

    with st.container(border=True):
        st.subheader("Pace")
        st.caption("Damage per round is how fast each side spends the other's hit points.")
        pace_rows = pace_points(fights)
        if not pace_rows.empty:
            figure = px.box(
                pace_rows,
                x="side",
                y="damage_per_round",
                points="outliers",
                title="Damage per round",
            )
            figure.update_layout(xaxis_title="", yaxis_title="Damage per round", showlegend=False)
            _chart(figure, "Damage per round dealt by the party and by the enemies")
        party_lost = dropped_fraction(fights, "party_size", "party_down")
        enemy_lost = dropped_fraction(fights, "enemy_count", "enemy_down")
        st.write(
            f"On average the party lost {_percent(party_lost)} of its members and the enemies lost {_percent(enemy_lost)}."
        )

    if "winner" in fights.columns and "rounds" in fights.columns:
        length = fights.loc[:, ["winner", "rounds"]].copy()
        length["rounds"] = pd.to_numeric(length["rounds"], errors="coerce")
        length = length.dropna(subset=["rounds"])
        if not length.empty:
            with st.container(border=True):
                st.subheader("Length")
                _chart(
                    px.box(length, x="winner", y="rounds", points="outliers", title="Rounds by winner"),
                    "How many rounds fights lasted, split by winner",
                )

    labels = fights["label"].nunique() if "label" in fights.columns else 0
    if labels > 1:
        with st.container(border=True):
            st.subheader("Layouts")
            st.caption("Challenge rating and headcount only change when more than one layout is included.")
            table = layout_summary(fights)
            shown = table.copy()
            if "party_win" in shown.columns:
                shown["party_win"] = shown["party_win"].map(_percent)
            shown = shown.rename(columns={
                "label": "Layout",
                "fights": "Fights",
                "party_win": "Party win rate",
                "mean_rounds": "Mean rounds",
                "party_size": "Party size",
                "enemy_count": "Enemies",
                "mean_enemy_cr": "Mean enemy CR",
            })
            st.dataframe(shown, hide_index=True, alt="Win rate, length, and size of each layout")
            if {"policy", "label", "party_win"}.issubset(fights.columns):
                grouped = fights.copy()
                grouped["party_win"] = pd.to_numeric(grouped["party_win"], errors="coerce")
                summary = grouped.groupby(["label", "policy"], as_index=False)["party_win"].mean()
                figure = px.bar(
                    summary,
                    x="label",
                    y="party_win",
                    color="policy",
                    barmode="group",
                    title="Party win rate by encounter and policy",
                    range_y=[0, 1],
                )
                _chart(figure, "Party win rate for each encounter label and enemy policy")

    with st.container(border=True):
        st.subheader("Outliers")
        st.caption("A fight is flagged when its round count or party damage falls outside 1.5 times the interquartile range.")
        outliers = outlier_fights(fights)
        if outliers.empty:
            st.write("No outlier fights under that rule.")
        else:
            st.dataframe(outliers, hide_index=True, alt="Fights whose length or party damage is an outlier")

    if not turns.empty and {"side", "action_kind"}.issubset(turns.columns):
        with st.container(border=True):
            st.subheader("How turns were spent")
            st.caption(
                "End turn only closes a turn. A skip is a creature at 0 hit points whose place in the order came up. "
                "The model is not asked on either one."
            )
            spent = living_actions(turns)
            if not spent.empty:
                kinds = spent.groupby(["side", "action_kind"]).size().reset_index(name="count")
                totals = kinds.groupby("side")["count"].transform("sum")
                kinds["share"] = kinds["count"] / totals
                figure = px.bar(
                    kinds,
                    x="action_kind",
                    y="share",
                    color="side",
                    barmode="group",
                    title="Share of turns",
                    range_y=[0, 1],
                )
                figure.update_layout(xaxis_title="", yaxis_title="Share of that side's turns")
                _chart(figure, "Share of each side's living actions, leaving out End and creatures at 0 hit points")
            if "source" in spent.columns:
                enemy_turns = spent[spent["side"] == "enemy"]
                if not enemy_turns.empty:
                    st.write("Who chose enemy actions:")
                    st.dataframe(
                        enemy_turns["source"].value_counts().rename_axis("source").reset_index(name="turns"),
                        hide_index=True,
                        alt="How many living enemy actions came from the model, the heuristic, or a fallback",
                    )

    described = describe_fights(fights)
    if not described.empty:
        with st.expander("Raw column summaries"):
            st.caption(
                "Each table is one winner. Rounds belong to the fight. "
                "Party sits above enemy for the same measure."
            )
            summary_format = {
                "count": st.column_config.NumberColumn(format="%d"),
                "mean": st.column_config.NumberColumn(format="%.2f"),
                "std": st.column_config.NumberColumn(format="%.2f"),
                "min": st.column_config.NumberColumn(format="%.2f"),
                "25%": st.column_config.NumberColumn(format="%.2f"),
                "50%": st.column_config.NumberColumn(format="%.2f"),
                "75%": st.column_config.NumberColumn(format="%.2f"),
                "max": st.column_config.NumberColumn(format="%.2f"),
            }
            summary_labels = {
                "all": "All fights",
                "party": "Won by the party",
                "enemy": "Won by the enemy",
                "draw": "Draw",
            }
            summary_alt = {
                "all": "Rounds, damage, and hit points remaining for the whole batch",
                "party": "Rounds, damage, and hit points remaining in fights the party won",
                "enemy": "Rounds, damage, and hit points remaining in fights the enemy won",
                "draw": "Rounds, damage, and hit points remaining in fights that drew",
            }
            for winner in described.index.get_level_values("Winner").unique():
                block = described.xs(winner, level="Winner").copy()
                block["count"] = block["count"].round().astype("Int64")
                st.markdown(f"**{summary_labels.get(winner, winner)}**")
                st.dataframe(
                    block,
                    hide_index=False,
                    column_config=summary_format,
                    alt=summary_alt.get(winner, "Rounds, damage, and hit points remaining"),
                )

    st.subheader("Written report")
    st.write(
        "The model writes a short reading of how the fights went. "
        "It is told to interpret the numbers above and not to recite them."
    )
    brief = report_brief(fights, turns=turns)
    if st.button("Write the report", width="stretch"):
        try:
            text = write_report(brief)
        except LLMError as exc:
            st.error(str(exc))
        else:
            st.session_state.report_text = text
            report_path = RUNS / "latest-report.md"
            report_path.write_text(text + "\n\n## Figures the model was given\n\n" + brief, encoding="utf-8")
            st.success(f"Saved {report_path}.")
    if st.session_state.get("report_text"):
        st.write(st.session_state.report_text)
    with st.expander("Numbers sent to the model"):
        st.code(brief)
