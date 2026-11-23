"""Turn order and resolution for one fight."""

from __future__ import annotations

import json
import random
import statistics

from src.actions import (
    EMANATION_INDEXES,
    GameAction,
    _afflict_pool,
    adjacent_foe,
    heal_expression,
    legal_actions,
    movement_budget,
    nick_weapon_name,
    same_action,
    spell_parts,
    targets_for,
    turn_speed,
    using_melee,
    weapon_reaches,
    _token_target,
)
from src.bestiary import AttackOption, ability_mod
from src.combatant import Combatant
from src.conditions import (
    BANISH_TYPES,
    OUTSIDERS,
    add_mark,
    affliction,
    attack_penalty,
    break_on_attack,
    can_see,
    charged_toward,
    clear_condition,
    creature_types,
    drop_marks,
    holds_concentration,
    hit_rider_marks,
    immune_to,
    is_elf,
    is_incapacitated,
    mark_by_id,
    marks_of,
    movement_cost,
    pending_order,
    reap,
    rider_skipped,
    save_mode,
    save_penalty,
    size_rank,
    spell_targets_extra,
    stand,
    stand_cost,
    sync,
)
from src.zones import each_zone as _each_zone
from src.dice import roll_dice, substitute_mod
from src.feats import melee_reach, pole_weapon, reroll_ones
from src.features import apply_feature, begin_turn_state, break_hide, push_from, spend_focus
from src.grid import area_squares, feet_between, footprint, in_bounds, space_gap, step_path
from src.zones import (
    already_saved,
    anchor_cost,
    cells_of,
    centered_block,
    compass,
    difficult_costs,
    drift_delta,
    effect_parts,
    follow_delta,
    footprint_free,
    is_zone,
    mark_saved,
    read_zone,
    travel_cost,
    trigger_cells,
    zone_spec,
)
from src.reactions import (
    apply_choice,
    asks_player,
    candidate_ids,
    heuristic_choice,
    offers_for,
    prompt_for,
)
from src.rules import (
    Effect,
    ability_name,
    apply_damage,
    apply_heal,
    armor_class,
    attack_mode,
    attack_reason_text,
    combine,
    complete_hit,
    d20,
    opposed_ac,
    resolve_attack,
    resolve_auto,
    resolve_save,
)


_CARRY = (
    "resume_squares",
    "move_actor",
    "move_budget",
    "move_dest",
    "move_cost",
    "move_origin",
    "clear_strike_id",
    "later_reactors",
    "resume_phase",
    "entered",
    "left",
    "opportunity",
    "opportunity_done",
)


def _path_squares(origin: tuple[int, int], dest: tuple[int, int]) -> list[tuple[int, int]]:
    return step_path(origin, dest)


def _spell_attacks(actor: Combatant, spell, slot: int, parts: list[tuple[str, str]]) -> tuple[int, list[tuple[str, str]]]:
    """How many attack rolls a spell makes, and the damage on each one."""
    repeats = 1
    beam_parts = list(parts)
    if spell.index == "eldritch-blast":
        level = int(actor.caster_level or actor.traits.get("level") or 1)
        if level >= 17:
            repeats = 4
        elif level >= 11:
            repeats = 3
        elif level >= 5:
            repeats = 2
    elif spell.index == "scorching-ray":
        repeats = 3 + max(0, int(slot) - 2)
    return repeats, beam_parts


class IllegalAction(ValueError):
    """The action is not legal for the current creature."""


class Encounter:
    def __init__(
        self,
        combatants: list[Combatant],
        *,
        width: int = 12,
        height: int = 12,
        seed: int = 1,
        round_cap: int = 15,
        label: str = "Encounter",
        enemy_policy: str = "llm",
    ):
        self.combatants = combatants
        self.width = width
        self.height = height
        self.seed = seed
        self.round_cap = round_cap
        self.label = label
        self.enemy_policy = enemy_policy
        self.rng = random.Random(seed)
        self.rng_seed = int(seed)
        self.order: list[str] = []
        self.round = 1
        self.turn_index = 0
        self.log: list[str] = []
        self.turns: list[dict] = []
        self.over = False
        self.winner: str | None = None
        self.damage_dealt = {"party": 0, "enemy": 0}
        self.blueprint: dict | None = None
        self.mean_cr = 0.0
        self.mean_attack_bonus = 0.0
        self.start_distance = 0.0
        self.started = False
        self._prepared: tuple[int, int] | None = None
        self.pending: dict | None = None
        self.reaction_policy = "ask"
        self.auto_side: str | None = None
        self.auto_policy: str | None = None

    def by_id(self, creature_id: str) -> Combatant:
        for creature in self.combatants:
            if creature.id == creature_id:
                return creature
        raise KeyError(creature_id)

    def current(self) -> Combatant:
        return self.by_id(self.order[self.turn_index])

    def to_setup(self) -> dict:
        return {
            "label": self.label,
            "width": self.width,
            "height": self.height,
            "seed": self.seed,
            "round_cap": self.round_cap,
            "enemy_policy": self.enemy_policy,
            "combatants": [creature.to_dict() for creature in self.combatants],
        }

    @classmethod
    def from_setup(cls, data: dict) -> Encounter:
        return cls(
            [Combatant.from_dict(item) for item in data["combatants"]],
            width=int(data.get("width", 12)),
            height=int(data.get("height", 12)),
            seed=int(data.get("seed", 1)),
            round_cap=int(data.get("round_cap", 15)),
            label=data.get("label", "Encounter"),
            enemy_policy=data.get("enemy_policy", "llm"),
        )

    def layout_setup(self) -> dict:
        """Opening positions, even if a fight has moved the tokens."""
        if self.blueprint:
            return json.loads(json.dumps(self.blueprint))
        return self.to_setup()

    def restore_opening_layout(self) -> None:
        """Put creatures back on their opening squares and allow the layout to be edited."""
        if self.blueprint:
            for saved in self.blueprint["combatants"]:
                creature = self.by_id(saved["id"])
                creature.position = (int(saved["position"][0]), int(saved["position"][1]))
            self.blueprint = None
        for creature in self.combatants:
            creature.restore_body()
        self.started = False
        self.over = False
        self.winner = None
        self.order = []
        self.log = []
        self.turns = []
        self.round = 1
        self.turn_index = 0
        self._prepared = None
        self.pending = None
        self.auto_side = None
        self.auto_policy = None
        self.damage_dealt = {"party": 0, "enemy": 0}

    def start(self) -> None:
        if self.blueprint:
            for saved in self.blueprint["combatants"]:
                creature = self.by_id(saved["id"])
                creature.position = (int(saved["position"][0]), int(saved["position"][1]))
        for creature in self.combatants:
            creature.restore_body()
            if creature.traits.get("starts_invisible"):
                add_mark(creature, creature, condition="invisible", until="fight")
        self.rng = random.Random(self.seed)
        self.rng_seed = int(self.seed)
        self.log = []
        self.turns = []
        self.over = False
        self.winner = None
        self._prepared = None
        self.pending = None
        self.auto_side = None
        self.auto_policy = None
        self.damage_dealt = {"party": 0, "enemy": 0}
        self.round = 1
        self.turn_index = 0
        self.started = True
        ranking = []
        totals = {}
        bonuses = {}
        for creature in self.combatants:
            roll = self.rng.randint(1, 20)
            bonus = ability_mod(creature.abilities.get("dexterity", 10)) + int(creature.traits.get("initiative_bonus") or 0)
            total = roll + bonus
            totals[creature.id] = total
            bonuses[creature.id] = bonus
        swaps = self._swap_alert(totals)
        ranking = [(totals[creature.id], bonuses[creature.id], creature.id) for creature in self.combatants]
        ranking.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        self.order = [item[2] for item in ranking]
        self.log.append(f"Dice seed {self.seed}.")
        self.log.append(
            "Initiative: " + ", ".join(f"{self.by_id(creature_id).name} {totals[creature_id]}" for creature_id in self.order)
        )
        self.log.extend(swaps)
        self._capture_start_stats()
        self.blueprint = self.to_setup()
        self._check_end()

    def _capture_start_stats(self) -> None:
        party = [creature for creature in self.combatants if creature.side == "party"]
        enemy = [creature for creature in self.combatants if creature.side == "enemy"]
        ratings = [creature.challenge_rating for creature in enemy if creature.challenge_rating is not None]
        bonuses = [
            attack.attack_bonus
            for creature in enemy
            for attack in creature.attacks
            if attack.attack_bonus is not None
        ]
        distances = []
        for foe in enemy:
            if party:
                distances.append(min(foe.gap_to(ally) * 5 for ally in party))
        self.mean_cr = statistics.fmean(ratings) if ratings else 0.0
        self.mean_attack_bonus = statistics.fmean(bonuses) if bonuses else 0.0
        self.start_distance = statistics.fmean(distances) if distances else 0.0

    def use_seed(self, seed: int) -> None:
        """Point later rolls at ``seed``. A fight that has started keeps its log."""
        seed = int(seed)
        self.seed = seed
        if not self.started or seed == self.rng_seed:
            return
        self.rng = random.Random(seed)
        self.rng_seed = seed
        self.log.append(f"Dice seed is now {seed}. Rolls from here use that seed.")

    def begin_turn(self) -> None:
        while not self.over:
            actor = self.current()
            marker = (self.round, self.turn_index)
            if actor.hp <= 0:
                self._record(actor, "skip", "skip", "engine", None, 0, 0, f"{actor.name} is at 0 hit points and skips.")
                self._advance()
                continue
            if self._prepared == marker:
                return
            self._prepared = marker
            actor.dodging = False
            actor.action_available = True
            actor.bonus_available = True
            begin_turn_state(self, actor)
            self.refresh_marks()
            self._arm_marks(actor)
            self._open_zones(actor)
            self._auras(actor)
            self._heroism_pulse(actor)
            self.refresh_positional()
            actor.movement_left = turn_speed(actor)
            self._recharge(actor)
            self._ongoing(actor)
            if self._check_end() or actor.hp <= 0:
                if actor.hp <= 0 and not self.over:
                    self._prepared = None
                    continue
                return
            return

    def legal_actions(self, actor: Combatant | None = None, *, automated: bool = False) -> list[GameAction]:
        return legal_actions(self, actor or self.current(), automated=automated)

    def apply(self, action: GameAction, source: str, reason: str = "") -> None:
        if self.over:
            raise IllegalAction("The fight is over.")
        if self.pending:
            raise IllegalAction("A reaction is waiting.")
        self.begin_turn()
        if self.over:
            return
        actor = self.current()
        legal = self.legal_actions(actor, automated=False)
        if not any(same_action(action, item) for item in legal):
            raise IllegalAction(action.label)
        if action.kind == "end":
            effect = combine([
                Effect(lines=[f"{actor.name} ends the turn."]),
                self._repeat_marks(actor),
                self._expire_marks(actor),
                self._emanation_end(actor),
                self._zone_end(actor),
                self._caster_pulse(actor),
            ])
        elif action.kind == "stand":
            effect = self._stand(actor)
        elif action.kind == "order":
            effect = self._order(actor, action)
        elif action.kind == "escape":
            effect = self._escape(actor, action)
        elif action.kind == "shake":
            effect = self._shake(actor, action)
        elif action.kind == "move":
            effect = self._relocate(actor, action)
        elif action.kind == "dash":
            gained = turn_speed(actor)
            if actor.traits.get("charger"):
                gained += 10
            actor.movement_left = movement_budget(actor) + gained
            if action.budget == "bonus":
                actor.bonus_available = False
            else:
                actor.action_available = False
                self._open_banked(actor)
            if action.attack_name:
                line = f"{actor.name} uses {action.attack_name}. {actor.movement_left} ft of movement left."
            else:
                line = f"{actor.name} dashes. {actor.movement_left} ft of movement left."
            effect = Effect(lines=[line])
        elif action.kind == "disengage":
            actor.traits["disengaged"] = True
            if action.budget == "bonus":
                actor.bonus_available = False
            else:
                actor.action_available = False
                self._open_banked(actor)
            if action.attack_name:
                line = f"{actor.name} uses {action.attack_name} and disengages."
            else:
                line = f"{actor.name} takes the Disengage action."
            effect = combine([
                Effect(lines=[line]),
                self._sentinel_disengage(actor),
            ])
        elif action.kind == "dodge":
            actor.dodging = True
            actor.action_available = False
            self._open_banked(actor)
            effect = Effect(lines=[f"{actor.name} dodges."])
        elif action.kind == "attack":
            effect = self._do_attack(actor, action)
            self._commit(actor, action)
        elif action.kind == "spell":
            effect = self._do_spell(actor, action)
            self._commit(actor, action)
        elif action.kind == "feature":
            effect = apply_feature(self, actor, action)
        else:
            raise IllegalAction(action.kind)
        if self.pending:
            self._hold(actor, action, source, reason, effect)
            return
        self.damage_dealt[actor.side] += effect.damage
        if reason:
            effect.lines.append(f"Reason: {reason}")
        detail = " ".join(effect.lines)
        for line in effect.lines:
            self.log.append(line)
        self._record(
            actor,
            action.kind,
            action.label,
            source,
            action.target_id,
            effect.damage,
            effect.healing,
            detail,
        )
        self.refresh_positional()
        for line in self._flush_saves():
            if line:
                self.log.append(line)
        self._reap()
        ended = self._check_end()
        if action.kind in {"end", "order"} and not ended:
            self._advance()

    def _do_attack(self, actor: Combatant, action: GameAction) -> Effect:
        if action.attack_name == "Multiattack":
            target = self.by_id(action.target_id or "")
            sequence = []
            skipped = []
            for name, count in actor.multiattack:
                option = actor.attack_named(name)
                if option is None or option.resolution != "attack":
                    skipped.append(Effect(lines=[f"{name} is not auto-resolved."]))
                    continue
                sequence.extend([name] * int(count))
            return combine([*skipped, self._run_strikes(actor, target, sequence)])
        option = actor.attack_named(action.attack_name or "")
        if option is None:
            raise IllegalAction(action.label)
        self._consume(actor, option)
        if option.disagrees:
            note = Effect(
                lines=[
                    f"{option.name}: the API marks a successful save as no damage, and the description says half. "
                    "This roll follows the structured field."
                ]
            )
        else:
            note = Effect()
        aimed = bool((option.rider or {}).get("aimed")) and action.aim is not None
        if option.area_type and (action.facing or aimed):
            effects = [note]
            targets = targets_for(self, actor, action)
            for index, target in enumerate(targets):
                effect = self._rolled_save(
                    actor,
                    target,
                    dc=option.dc_value or 10,
                    ability=option.dc_ability or "dex",
                    success=option.dc_success or "none",
                    damage=[(part["type"], part["expr"]) for part in option.damage],
                    rng=self.rng,
                    label=option.name,
                )
                effects.append(effect)
                if self.pending:
                    self.pending["save_rest"] = [item.id for item in targets[index + 1 :]]
                    self.pending["save_source"] = actor.id
                    break
                self._mark_prone_from_save(actor, target, option.rider or {}, effect)
            if len(effects) == 1:
                effects.append(Effect(lines=[f"{option.name} hits no one."]))
            return combine(effects)
        target = self.by_id(action.target_id or "")
        rider = option.rider or {}
        contest = bool(rider.get("grapple") or rider.get("shove"))
        actor.traits["swing_budget"] = str(action.budget or "")
        actor.traits["swing_tag"] = str(action.tag or "")
        if option.resolution == "invisible":
            note_line = add_mark(actor, actor, condition="invisible", until="start", spell=option.name)
            return combine([note, Effect(lines=[note_line or f"{actor.name} is invisible."])])
        if option.resolution == "save" and contest:
            if action.budget == "action" and not action.tag:
                actor.traits["on_attack_action"] = True
                actor.traits["monk_attacked"] = True
            effect = self._monk_contest(actor, target, option)
            if not self.pending:
                actor.traits.pop("on_attack_action", None)
            return combine([note, effect])
        if option.resolution == "save":
            rider = option.rider or {}
            condition = str(rider.get("save_condition") or "")
            if rider_skipped(target, rider):
                return combine([note, Effect(lines=[f"{option.name} does not affect {target.name}."])])
            if condition and immune_to(target, condition, actor):
                return combine([note, Effect(lines=[f"{target.name} is immune to {condition}."])])
            effect = self._rolled_save(
                actor,
                target,
                dc=option.dc_value or rider.get("save_dc") or 10,
                ability=option.dc_ability or rider.get("save_ability") or "dex",
                success=option.dc_success or "none",
                damage=[(part["type"], part["expr"]) for part in option.damage],
                rng=self.rng,
                label=option.name,
            )
            if not self.pending:
                self._save_rider(actor, target, rider, effect)
            return combine([note, effect])
        hew = action.tag == "hew"
        if hew:
            actor.traits["hew_strike"] = True
        if action.budget == "action" and not action.tag:
            actor.traits["on_attack_action"] = True
            if (option.rider or {}).get("unarmed") or (option.rider or {}).get("monk_weapon"):
                actor.traits["monk_attacked"] = True
        if action.tag == "dual":
            actor.traits["strip_mod"] = True
        try:
            if action.tag == "pole":
                return self._pole_strike(actor, target, option)
            return self._strike(actor, target, option)
        finally:
            if hew:
                actor.traits.pop("hew_strike", None)
            if not self.pending:
                actor.traits.pop("on_attack_action", None)
                actor.traits.pop("strip_mod", None)

    def _pole_strike(self, actor, target, option) -> Effect:
        butt = AttackOption(
            name="Pole strike",
            resolution="attack",
            attack_bonus=option.attack_bonus,
            reach_feet=option.reach_feet,
            damage=[{"type": "bludgeoning", "expr": "1d4"}],
            ability=option.ability or "strength",
        )
        return self._strike(actor, target, butt)

    def _strike(self, actor: Combatant, target: Combatant, option, *, reaction_attack: bool = False, sentinel_halt: bool = False) -> Effect:
        melee = using_melee(actor, target, option)
        bonus = int(option.attack_bonus or 0)
        if actor.traits.get("archery") and not melee and option.range_feet and not option.thrown:
            bonus += 2
        if self._is_nick_swing(actor, option):
            actor.traits["strip_mod"] = True
        effect = self._rolled_attack(
            actor,
            target,
            bonus=bonus,
            damage=[(part["type"], part["expr"]) for part in option.damage],
            melee=melee,
            label=option.name,
            option=option,
            spell=False,
            reaction_attack=reaction_attack,
            sentinel_halt=sentinel_halt,
        )
        if self.pending:
            return effect
        self._after_attack_extras(actor, target, option, effect, reaction_attack=reaction_attack)
        if self.pending:
            self._add_prefix(effect)
            return Effect()
        return effect

    def _mastery_follow(self, actor: Combatant, target: Combatant, option, effect: Effect) -> None:
        if not effect.hit or option is None or not option.mastery:
            return
        if option.mastery == "push":
            return
        if option.mastery == "topple":
            ability = option.ability or "strength"
            dc = 8 + int(actor.traits.get("proficiency") or 3) + ability_mod(actor.abilities.get(ability, 10))
            save = target.save_bonuses.get("strength", ability_mod(target.abilities.get("strength", 10)))
            roll, shown = d20(self.rng, 0)
            total = roll + save
            if total >= dc:
                effect.lines.append(f"{target.name} resists Topple ({shown[0]}+{save}={total} vs DC {dc}).")
            else:
                add_mark(target, actor, condition="prone", until="stand")
                effect.lines.append(f"{target.name} is knocked prone ({shown[0]}+{save}={total} vs DC {dc}).")
            return
        if option.mastery == "cleave" and not actor.traits.get("cleave_used"):
            actor.traits["cleave_used"] = True
            extra = None
            for foe in self.combatants:
                if foe.hp <= 0 or foe.id in {target.id, actor.id} or foe.side == actor.side:
                    continue
                if target.gap_to(foe) > 1:
                    continue
                if not weapon_reaches(actor, foe, option):
                    continue
                extra = foe
                break
            if extra is None:
                return
            effect.lines.append(f"Cleave reaches {extra.name}.")
            follow = self._strike(actor, extra, option)
            if self.pending:
                return
            effect.damage += follow.damage
            effect.lines.extend(follow.lines or [])

    def _do_spell(self, actor: Combatant, action: GameAction) -> Effect:
        spell = actor.spell_named(action.spell_index or "")
        if spell is None:
            raise IllegalAction(action.label)
        slot = int(action.slot_level or 0)
        targets = [
            target
            for target in targets_for(self, actor, action)
            if not (spell.kind != "heal" and target.id == actor.id and spell.area_type in {"cone", "line"})
        ]
        if spell.level > 0:
            actor.traits["leveled_cast"] = True
        window = {
            "actor_id": actor.id,
            "target_id": None,
            "spell_index": spell.index,
            "slot": slot,
            "label": action.label,
            "target_ids": [target.id for target in targets],
            "aim": action.aim,
            "phase": "counterspell",
            "queue": candidate_ids(self, None),
        }
        if self._walk(window, "counterspell"):
            return Effect()
        if window.get("countered"):
            return Effect()
        if spell.index == "spiritual-weapon":
            return self._place_spiritual_weapon(actor, spell, slot, action)
        if is_zone(spell.index):
            return self._place_zone(actor, spell, slot, action)
        if affliction(spell.index) and spell.kind == "afflict":
            return self._cast_affliction(actor, spell, slot, targets, action)
        effect = self._commit_spell(actor, spell, slot, targets, action.label)
        if not self.pending:
            effect.lines.extend(break_on_attack(actor))
        return effect

    def _spend_spell_slot(self, actor, spell, slot: int) -> str:
        """Spend a free casting first, then a slot. Returns the free-use name."""
        if spell is None or spell.level <= 0:
            return ""
        mapped = (actor.traits.get("free_spells") or {}).get(spell.index)
        if mapped and int(slot) == int(spell.level) and int(actor.uses_left.get(mapped, 0)) > 0:
            actor.uses_left[mapped] = int(actor.uses_left.get(mapped, 0)) - 1
            return str(mapped)
        actor.slots[slot] = actor.slots.get(slot, 0) - 1
        return ""

    def _begin_concentration(self, actor, spell, slot: int) -> None:
        """Keep an ongoing area, a placed zone, or Spiritual Weapon. Another spell ends both."""
        previous = actor.concentrating_on
        replacing = bool(spell.concentration) and bool(previous) and (
            previous != spell.index or is_zone(spell.index)
        )
        if replacing:
            self._release_caster(actor)
        keeps = holds_concentration(spell)
        if keeps:
            if previous in {"hex", "hunters-mark"} and previous != spell.index:
                actor.traits.pop("mark_id", None)
                actor.traits.pop("mark_die", None)
                actor.traits.pop("mark_type", None)
            if previous == "spiritual-weapon" and spell.index != "spiritual-weapon":
                actor.traits.pop("spiritual_weapon", None)
            if is_zone(previous) and previous != spell.index:
                actor.traits.pop("zone", None)
            actor.concentrating_on = spell.index
            actor.ongoing_slot = slot
            self._reap()
            return
        if spell.concentration:
            self._release_caster(actor)
            actor.concentrating_on = None
            actor.ongoing_slot = None
            actor.traits.pop("spiritual_weapon", None)
            actor.traits.pop("zone", None)
        self._reap()

    def _commit_spell(self, actor, spell, slot: int, targets: list[Combatant], label: str) -> Effect:
        free_name = self._spend_spell_slot(actor, spell, slot)
        self._begin_concentration(actor, spell, slot)
        if spell.index in EMANATION_INDEXES:
            self._mark_saved(actor, [target.id for target in targets])
        effect = self._apply_spell(actor, spell, slot, targets, label)
        if free_name:
            effect.lines.insert(0, f"{actor.name} casts {spell.name} without a spell slot.")
        return effect

    def _place_spiritual_weapon(self, actor, spell, slot: int, action: GameAction) -> Effect:
        if action.aim is None:
            return Effect(lines=[f"{actor.name} does not choose a square for Spiritual Weapon."])
        square = (int(action.aim[0]), int(action.aim[1]))
        free_name = self._spend_spell_slot(actor, spell, slot)
        self._begin_concentration(actor, spell, slot)
        actor.traits["spiritual_weapon"] = {"x": square[0], "y": square[1], "slot": int(slot)}
        lines = []
        if free_name:
            lines.append(f"{actor.name} casts {spell.name} without a spell slot.")
        lines.append(f"{actor.name} summons a Spiritual Weapon at {square}.")
        header = Effect(lines=lines)
        foe = adjacent_foe(self, actor, square)
        if foe is None:
            return header
        attack = self._rolled_attack(
            actor,
            foe,
            bonus=actor.spell_attack_bonus,
            damage=spell_parts(actor, spell, slot),
            melee=True,
            label=spell.name,
            option=None,
            spell=True,
        )
        if self.pending:
            return header
        return combine([header, attack])

    def move_spiritual_weapon(self, actor, action: GameAction) -> Effect:
        """Bonus action: move the token up to 20 feet and attack. No slot is spent."""
        actor.bonus_available = False
        placed = actor.traits.get("spiritual_weapon")
        if actor.concentrating_on != "spiritual-weapon" or not isinstance(placed, dict):
            return Effect(lines=[f"{actor.name} has no Spiritual Weapon."])
        origin = (int(placed.get("x", 0)), int(placed.get("y", 0)))
        square = tuple(action.aim) if action.aim is not None else origin
        square = (int(square[0]), int(square[1]))
        if space_gap(origin, 1, square, 1) > 4:
            return Effect(lines=[f"The Spiritual Weapon cannot reach {square}."])
        slot = int(placed.get("slot") or actor.ongoing_slot or 2)
        actor.traits["spiritual_weapon"] = {"x": square[0], "y": square[1], "slot": slot}
        if square == origin:
            line = f"{actor.name} strikes with the Spiritual Weapon at {square}."
        else:
            line = f"{actor.name} moves the Spiritual Weapon to {square}."
        header = Effect(lines=[line])
        spell = actor.spell_named("spiritual-weapon")
        foe = adjacent_foe(self, actor, square)
        if foe is None or spell is None:
            return header
        attack = self._rolled_attack(
            actor,
            foe,
            bonus=actor.spell_attack_bonus,
            damage=spell_parts(actor, spell, slot),
            melee=True,
            label="Spiritual Weapon",
            option=None,
            spell=True,
        )
        if self.pending:
            return header
        return combine([header, attack])

    def _release_caster(self, actor) -> None:
        """Drop a zone or token this caster was holding, and the conditions it set."""
        actor.traits.pop("zone", None)
        actor.traits.pop("spiritual_weapon", None)
        actor.traits.pop("outlined_ids", None)
        for creature in self.combatants:
            if creature.traits.get("outline_source") == actor.id:
                creature.conditions.pop("outlined", None)
                creature.traits.pop("outline_source", None)
            if creature.traits.get("restrained_by") == actor.id:
                creature.conditions.pop("restrained", None)
                creature.traits.pop("restrained_by", None)
            if creature.traits.get("frightened_by") == actor.id:
                creature.conditions.pop("frightened", None)
                creature.traits.pop("frightened_by", None)

    def refresh_marks(self) -> None:
        """A condition from a zone lasts only while that caster still holds the spell."""
        self._reap(moment="start")
        for creature in self.combatants:
            for key, condition in (
                ("outline_source", "outlined"),
                ("restrained_by", "restrained"),
                ("frightened_by", "frightened"),
            ):
                source = creature.traits.get(key)
                if not source:
                    continue
                try:
                    owner = self.by_id(source)
                except KeyError:
                    owner = None
                if owner is None or owner.hp <= 0 or not owner.concentrating_on or read_zone(owner) is None:
                    if key == "frightened_by" and owner is not None and owner.concentrating_on == "conjure-fey":
                        continue
                    creature.conditions.pop(condition, None)
                    creature.traits.pop(key, None)

    def _place_zone(self, actor, spell, slot: int, action: GameAction) -> Effect:
        spec = zone_spec(spell.index)
        if spec is None or action.aim is None:
            return Effect(lines=[f"{actor.name} does not choose a square for {spell.name}."])
        square = (int(action.aim[0]), int(action.aim[1]))
        form = action.attack_name or spec.shape
        step_x, step_y = (1, 0)
        if form == "line":
            step_x, step_y = compass(tuple(actor.position), square)
        free_name = self._spend_spell_slot(actor, spell, slot)
        placed = {
            "index": spec.index,
            "x": square[0],
            "y": square[1],
            "slot": int(slot),
            "form": form,
            "dx": step_x,
            "dy": step_y,
            "saved": {"turn": [], "ids": []},
            "hold": "",
            "rounds": int(spec.linger or 0),
        }
        if spec.linger:
            lingers = [item for item in (actor.traits.get("lingers") or []) if isinstance(item, dict)]
            lingers.append(placed)
            actor.traits["lingers"] = lingers
        else:
            self._begin_concentration(actor, spell, slot)
            actor.traits["zone"] = placed
        lines = []
        if free_name:
            lines.append(f"{actor.name} casts {spell.name} without a spell slot.")
        lines.append(f"{actor.name} casts {spell.name} at {square}.")
        effects = [Effect(lines=lines)]
        if spec.attack:
            foe = _token_target(self, actor, square, spec)
            if foe is not None:
                effects.append(self._pay_zone(actor, self._zone_attack(actor, foe, spec, placed)))
            return combine(effects)
        if spec.allows("cast"):
            for creature in list(self.combatants):
                if creature.hp <= 0 or (spec.skip_self and creature.id == actor.id):
                    continue
                if not self._zone_covers(actor, spec, placed, creature, "cast"):
                    continue
                effects.append(self._pay_zone(actor, self._zone_trigger(actor, creature, spec, placed, "cast")))
                if self.pending:
                    break
        return combine(effects)

    def _zone_parts(self, caster, spec, slot: int, phase: str) -> list[tuple[str, str]]:
        spell = caster.spell_named(spec.index)
        level = spell.level if spell is not None else 0
        parts = []
        for damage_type, expression in effect_parts(spec, slot, level, phase):
            if "MOD" in expression.upper():
                expression = substitute_mod(expression, caster.spellcasting_mod)
            parts.append((damage_type, expression))
        return parts

    def _zone_covers(self, caster, spec, placed, creature, phase: str) -> bool:
        anchor = (int(creature.position[0]), int(creature.position[1]))
        if spec.bolt and phase in {"cast", "push"} and spec.shape == "cloud":
            return bool(creature.squares() & trigger_cells(self, caster, spec, placed, later=False))
        if spec.index == "conjure-elemental" and phase == "enter":
            return bool(creature.squares() & cells_of(spec, placed))
        if spec.shape == "token" and spec.near and phase in {"enter", "end", "start", "push"}:
            gap = space_gap((int(placed["x"]), int(placed["y"])), spec.span, anchor, creature.span)
            return gap * 5 <= spec.near
        later = phase in {"enter", "end"} and bool(spec.side)
        return bool(creature.squares() & trigger_cells(self, caster, spec, placed, later=later))

    def _zone_label(self, caster, spec) -> str:
        spell = caster.spell_named(spec.index)
        return spell.name if spell is not None else spec.index

    def _failed(self, effect: Effect) -> bool:
        text = " ".join(effect.lines or [])
        return "Failed save" in text

    def _keep_zone(self, caster, placed) -> None:
        if caster.traits.get("zone") is placed:
            caster.traits["zone"] = placed

    def _impose(self, caster, target, condition: str, spec=None) -> str:
        if immune_to(target, condition, caster):
            return f"{target.name} is immune to {condition}."
        until = "concentration"
        extra = {}
        if spec is not None and getattr(spec, "until", ""):
            until = spec.until
        if spec is not None and spec.lock_action:
            extra["lock_action"] = True
        if condition == "outlined":
            target.traits["outline_source"] = caster.id
            ids = [str(item) for item in caster.traits.get("outlined_ids") or []]
            if target.id not in ids:
                ids.append(target.id)
            caster.traits["outlined_ids"] = ids
            clear_condition(target, "invisible")
            line = add_mark(target, caster, condition="outlined", until="concentration", tied="concentration", spell=getattr(spec, "index", "") or caster.concentrating_on or "")
            return line or f"{target.name} is outlined."
        if condition == "restrained":
            target.traits["restrained_by"] = caster.id
            line = add_mark(target, caster, condition="restrained", until=until, tied="concentration" if until == "concentration" else "", spell=getattr(spec, "index", "") or "", **extra)
            return line or f"{target.name} is restrained."
        if condition == "frightened":
            target.traits["frightened_by"] = caster.id
            line = add_mark(target, caster, condition="frightened", until="start", spell=getattr(spec, "index", "") or "")
            return line or f"{target.name} is frightened until the start of {caster.name}'s next turn."
        if condition == "prone":
            return add_mark(target, caster, condition="prone", until="stand")
        if condition == "poisoned":
            return add_mark(target, caster, condition="poisoned", until=until or "end", spell=getattr(spec, "index", "") or "", **extra)
        if condition == "blinded":
            rounds = 10 if until == "rounds" else 0
            return add_mark(target, caster, condition="blinded", until=until or "rounds", rounds=rounds, repeat=until == "rounds", save="constitution", dc=int(caster.spell_save_dc or 0), spell=getattr(spec, "index", "") or "")
        return add_mark(target, caster, condition=condition, until=until, spell=getattr(spec, "index", "") or "", **extra)

    def _clear_imposed(self, target, condition: str) -> None:
        drop_marks(target, lambda mark: mark.get("condition") == condition)
        target.conditions.pop(condition, None)
        if condition == "restrained":
            target.traits.pop("restrained_by", None)
        elif condition == "outlined":
            target.traits.pop("outline_source", None)
        elif condition == "frightened":
            target.traits.pop("frightened_by", None)

    def _remember_zone(self, caster, placed) -> None:
        mark_saved(placed, self, [])
        self._keep_zone(caster, placed)

    def _zone_save(self, caster, target, spec, slot: int, phase: str, label: str) -> Effect:
        return resolve_save(
            caster,
            target,
            dc=int(caster.spell_save_dc or 0),
            ability=spec.save or "dexterity",
            success=spec.success or "none",
            damage=self._zone_parts(caster, spec, slot, phase),
            rng=self.rng,
            label=label,
            advantage=bool(spec.condition == "poisoned" and target.traits.get("poison_ward")),
        )

    def _zone_auto(self, caster, target, spec, slot: int, phase: str, label: str) -> Effect:
        parts = self._zone_parts(caster, spec, slot, phase)
        if not parts:
            return Effect()
        return resolve_auto(caster, target, damage=parts, rng=self.rng, label=label)

    def _zone_attack(self, caster, target, spec, placed) -> Effect:
        slot = int(placed.get("slot") or caster.ongoing_slot or 0)
        label = self._zone_label(caster, spec)
        effect = self._rolled_attack(
            caster,
            target,
            bonus=caster.spell_attack_bonus,
            damage=self._zone_parts(caster, spec, slot, "main"),
            melee=True,
            label=label,
            option=None,
            spell=True,
        )
        if self.pending or not effect.hit or not spec.condition:
            return effect
        note = self._impose(caster, target, spec.condition, spec)
        if note:
            effect.lines.append(note)
        return effect

    def _zone_trigger(self, caster, target, spec, placed, phase: str) -> Effect:
        if spec.index == "conjure-elemental":
            return self._elemental_trigger(caster, target, spec, placed, phase)
        slot = int(placed.get("slot") or caster.ongoing_slot or 0)
        label = self._zone_label(caster, spec)
        if phase == "cast" and spec.appear:
            parts_phase = "appear"
        elif phase != "cast" and spec.later:
            parts_phase = "later"
        else:
            parts_phase = "main"
        automatic = phase != "cast" and spec.side and not spec.later_save
        if spec.save and not automatic:
            effect = self._zone_save(caster, target, spec, slot, parts_phase, label)
            if self._failed(effect) and spec.condition:
                note = self._impose(caster, target, spec.condition, spec)
                if note:
                    effect.lines.append(note)
                if spec.drop_conc and target.concentrating_on:
                    lost = target.concentrating_on
                    target.concentrating_on = None
                    target.ongoing_slot = None
                    target.traits.pop("zone", None)
                    effect.lines.append(f"{target.name} loses concentration on {lost}.")
        else:
            effect = self._zone_auto(caster, target, spec, slot, parts_phase, label)
        if not self.pending:
            mark_saved(placed, self, [target.id])
            self._keep_zone(caster, placed)
        return effect

    def _elemental_busy(self, caster, placed, target) -> bool:
        held = str(placed.get("hold") or "")
        if not held or held == target.id:
            return False
        try:
            other = self.by_id(held)
        except KeyError:
            placed["hold"] = ""
            return False
        return other.hp > 0 and other.traits.get("restrained_by") == caster.id

    def _elemental_trigger(self, caster, target, spec, placed, phase: str) -> Effect:
        slot = int(placed.get("slot") or caster.ongoing_slot or 0)
        label = self._zone_label(caster, spec)
        held = str(placed.get("hold") or "")
        if phase == "start" and held == target.id and target.traits.get("restrained_by") == caster.id:
            effect = self._zone_save(caster, target, spec, slot, "later", label)
            if not self._failed(effect):
                self._clear_imposed(target, "restrained")
                placed["hold"] = ""
                effect.lines.append(f"{target.name} is no longer restrained.")
            mark_saved(placed, self, [target.id])
            self._keep_zone(caster, placed)
            return effect
        if self._elemental_busy(caster, placed, target):
            return Effect()
        effect = self._zone_save(caster, target, spec, slot, "main", label)
        if self._failed(effect):
            note = self._impose(caster, target, "restrained")
            placed["hold"] = target.id
            if note:
                effect.lines.append(note)
        mark_saved(placed, self, [target.id])
        self._keep_zone(caster, placed)
        return effect

    def _pay_zone(self, caster, effect: Effect) -> Effect:
        if not self.pending:
            self.damage_dealt[caster.side] += int(effect.damage)
        return Effect(lines=list(effect.lines or []), healing=int(effect.healing))

    def _zones_along(self, mover, origin, squares) -> Effect:
        effects = []
        previous = (int(origin[0]), int(origin[1]))
        for raw in squares:
            square = (int(raw[0]), int(raw[1]))
            effects.append(self._zone_step(mover, previous, square))
            if mover.hp <= 0:
                mover.position = square
                mover.movement_left = 0
                break
            previous = square
        return combine(effects) if effects else Effect()

    def _zone_step(self, mover, previous, square) -> Effect:
        self.refresh_marks()
        effects = []
        previous = (int(previous[0]), int(previous[1]))
        square = (int(square[0]), int(square[1]))
        for caster, spec, placed in _each_zone(self):
            if spec.skip_self and mover.id == caster.id:
                continue
            if spec.allows("squares") and bool(footprint(square, mover.span) & cells_of(spec, placed)):
                label = self._zone_label(caster, spec)
                slot = int(placed.get("slot") or 0)
                effects.append(self._pay_zone(caster, self._zone_auto(caster, mover, spec, slot, "main", label)))
                if mover.hp <= 0:
                    return combine(effects)
            if spec.allows("enter") and not already_saved(placed, self, mover.id):
                now = self._covers_anchor(caster, spec, placed, square, mover.span, "enter")
                then = self._covers_anchor(caster, spec, placed, previous, mover.span, "enter")
                if now and not then:
                    effects.append(self._pay_zone(caster, self._zone_trigger(caster, mover, spec, placed, "enter")))
                    if mover.hp <= 0:
                        return combine(effects)
            if spec.hold == "inside" and mover.traits.get("restrained_by") == caster.id:
                if not bool(footprint(square, mover.span) & cells_of(spec, placed)):
                    self._clear_imposed(mover, "restrained")
        return combine(effects) if effects else Effect()

    def _covers_anchor(self, caster, spec, placed, anchor, span: int, phase: str) -> bool:
        if spec.index == "conjure-elemental" and phase == "enter":
            return bool(footprint(anchor, span) & cells_of(spec, placed))
        if spec.shape == "token" and spec.near:
            gap = space_gap((int(placed["x"]), int(placed["y"])), spec.span, anchor, span)
            return gap * 5 <= spec.near
        later = phase in {"enter", "end"} and bool(spec.side)
        return bool(footprint(anchor, span) & trigger_cells(self, caster, spec, placed, later=later))

    def _open_zones(self, actor) -> None:
        effect = combine([self._drift(actor), self._zone_start(actor)])
        if not effect.lines:
            return
        for line in effect.lines:
            self.log.append(line)
        self._record(actor, "ongoing", "zone", "engine", None, 0, 0, " ".join(effect.lines))

    def _drift(self, actor) -> Effect:
        found = read_zone(actor)
        if found is None:
            return Effect()
        spec, placed = found
        if spec.move != "drift":
            return Effect()
        origin = (int(placed.get("x", 0)), int(placed.get("y", 0)))
        step_x, step_y = drift_delta(tuple(actor.position), origin)
        nxt = (
            min(max(0, origin[0] + step_x), self.width - 1),
            min(max(0, origin[1] + step_y), self.height - 1),
        )
        if nxt == origin:
            return Effect()
        old = cells_of(spec, placed)
        placed["x"], placed["y"] = nxt
        actor.traits["zone"] = placed
        new = cells_of(spec, placed)
        effects = [Effect(lines=[f"{self._zone_label(actor, spec)} drifts to {nxt}."])]
        for creature in list(self.combatants):
            if creature.hp <= 0 or (spec.skip_self and creature.id == actor.id):
                continue
            body = creature.squares()
            if (body & new) and not (body & old) and not already_saved(placed, self, creature.id):
                effects.append(self._pay_zone(actor, self._zone_trigger(actor, creature, spec, placed, "push")))
        return combine(effects)

    def _zone_start(self, actor) -> Effect:
        effects = []
        for caster, spec, placed in _each_zone(self):
            if spec.skip_self and actor.id == caster.id:
                continue
            if already_saved(placed, self, actor.id):
                continue
            if spec.index == "conjure-elemental":
                anchor = (int(actor.position[0]), int(actor.position[1]))
                gap = space_gap((int(placed["x"]), int(placed["y"])), spec.span, anchor, actor.span)
                held = str(placed.get("hold") or "") == actor.id
                if held or gap * 5 <= spec.near:
                    effects.append(self._pay_zone(caster, self._elemental_trigger(caster, actor, spec, placed, "start")))
                continue
            if not spec.allows("start"):
                continue
            if not self._zone_covers(caster, spec, placed, actor, "start"):
                continue
            effects.append(self._pay_zone(caster, self._zone_trigger(caster, actor, spec, placed, "start")))
        return combine(effects) if effects else Effect()

    def _zone_end(self, actor) -> Effect:
        effects = []
        for caster, spec, placed in _each_zone(self):
            if not spec.allows("end") or already_saved(placed, self, actor.id):
                continue
            if spec.skip_self and actor.id == caster.id:
                continue
            if not self._zone_covers(caster, spec, placed, actor, "end"):
                continue
            effects.append(self._pay_zone(caster, self._zone_trigger(caster, actor, spec, placed, "end")))
        return combine(effects) if effects else Effect()

    def _follow_pack(self, actor, origin, dest) -> Effect:
        found = read_zone(actor)
        if found is None:
            return Effect()
        spec, placed = found
        if spec.move != "follow":
            return Effect()
        delta = follow_delta(origin, dest, spec.move_feet)
        if delta is None:
            return Effect()
        old = (int(placed.get("x", 0)), int(placed.get("y", 0)))
        nxt = (old[0] + delta[0], old[1] + delta[1])
        if not footprint_free(self, nxt, spec.span):
            return Effect()
        placed["x"], placed["y"] = nxt
        actor.traits["zone"] = placed
        effects = [Effect(lines=[f"The pack moves to {nxt}."])]
        for creature in list(self.combatants):
            if creature.hp <= 0 or already_saved(placed, self, creature.id):
                continue
            now = space_gap(nxt, spec.span, tuple(creature.position), creature.span) * 5 <= spec.near
            before = space_gap(old, spec.span, tuple(creature.position), creature.span) * 5 <= spec.near
            if now and not before:
                effects.append(self._pay_zone(actor, self._zone_trigger(actor, creature, spec, placed, "push")))
        return combine(effects)

    def move_zone(self, actor, action: GameAction) -> Effect:
        found = read_zone(actor)
        if found is None:
            return Effect(lines=[f"{actor.name} has nothing left to move."])
        spec, placed = found
        if spec.move == "bolt":
            actor.action_available = False
            return self._repeat_bolt(actor, spec, placed, action)
        if spec.move == "action":
            actor.action_available = False
        elif spec.move == "bonus":
            actor.bonus_available = False
        else:
            return Effect(lines=[f"{actor.name} cannot move {self._zone_label(actor, spec)}."])
        if action.aim is None:
            return Effect(lines=[f"{actor.name} does not choose a square."])
        origin = (int(placed.get("x", 0)), int(placed.get("y", 0)))
        square = (int(action.aim[0]), int(action.aim[1]))
        if space_gap(origin, 1, square, 1) * 5 > spec.move_feet:
            return Effect(lines=[f"{self._zone_label(actor, spec)} cannot reach {square}."])
        if spec.index == "flaming-sphere":
            return self._roll_sphere(actor, spec, placed, origin, square)
        if spec.shape == "token" and not footprint_free(self, square, spec.span) and square != origin:
            return Effect(lines=[f"{self._zone_label(actor, spec)} needs an empty square."])
        old_cells = cells_of(spec, placed)
        placed["x"], placed["y"] = square
        actor.traits["zone"] = placed
        name = self._zone_label(actor, spec)
        effects = [Effect(lines=[f"{actor.name} moves {name} to {square}."])]
        if spec.attack:
            foe = _token_target(self, actor, square, spec)
            if foe is not None:
                effects.append(self._pay_zone(actor, self._zone_attack(actor, foe, spec, placed)))
            return combine(effects)
        new_cells = cells_of(spec, placed)
        if spec.allows("push"):
            for creature in list(self.combatants):
                if creature.hp <= 0 or already_saved(placed, self, creature.id):
                    continue
                body = creature.squares()
                if (body & new_cells) and not (body & old_cells):
                    effects.append(self._pay_zone(actor, self._zone_trigger(actor, creature, spec, placed, "push")))
        return combine(effects)

    def _roll_sphere(self, actor, spec, placed, origin, dest) -> Effect:
        stopped = origin
        struck = None
        for step in step_path(origin, dest):
            if not in_bounds(step, self.width, self.height):
                break
            stopped = step
            for creature in self.combatants:
                if creature.hp > 0 and step in creature.squares():
                    struck = creature
                    break
            if struck is not None:
                break
        placed["x"], placed["y"] = stopped
        actor.traits["zone"] = placed
        effects = [Effect(lines=[f"{actor.name} rolls the Flaming Sphere to {stopped}."])]
        if struck is not None and not already_saved(placed, self, struck.id):
            effects.append(self._pay_zone(actor, self._zone_trigger(actor, struck, spec, placed, "push")))
        return combine(effects)

    def _repeat_bolt(self, actor, spec, placed, action: GameAction) -> Effect:
        if action.aim is None:
            return Effect(lines=[f"{actor.name} does not choose a point under the cloud."])
        cloud = (int(placed.get("x", 0)), int(placed.get("y", 0)))
        square = (int(action.aim[0]), int(action.aim[1]))
        if space_gap(cloud, 1, square, 1) * 5 > spec.size_feet:
            return Effect(lines=[f"That point is outside the storm cloud."])
        bolt = {
            "index": spec.index,
            "x": square[0],
            "y": square[1],
            "form": "bolt",
            "dx": 0,
            "dy": 0,
            "slot": placed.get("slot"),
            "saved": placed.get("saved"),
        }
        effects = [Effect(lines=[f"{actor.name} calls lightning down on {square}."])]
        for creature in list(self.combatants):
            if creature.hp <= 0 or already_saved(placed, self, creature.id):
                continue
            if not creature.in_cells(trigger_cells(self, actor, spec, bolt, later=False)):
                continue
            effects.append(self._pay_zone(actor, self._zone_trigger(actor, creature, spec, placed, "cast")))
        return combine(effects)

    def break_free(self, actor, action: GameAction) -> Effect:
        actor.action_available = False
        source = actor.traits.get("restrained_by")
        if not source:
            return Effect(lines=[f"{actor.name} is not restrained."])
        try:
            caster = self.by_id(source)
        except KeyError:
            return Effect(lines=[f"{actor.name} is not restrained."])
        found = read_zone(caster)
        if found is None or not found[0].escape:
            return Effect(lines=[f"{actor.name} cannot break free that way."])
        spec, placed = found
        bonus = int(actor.save_bonuses.get("strength", ability_mod(actor.abilities.get("strength", 10))))
        roll, _shown = d20(self.rng, 0)
        total = roll + bonus
        dc = int(caster.spell_save_dc or 0)
        if total >= dc:
            self._clear_imposed(actor, "restrained")
            if str(placed.get("hold") or "") == actor.id:
                placed["hold"] = ""
                self._keep_zone(caster, placed)
            if actor.movement_left == 0:
                actor.movement_left = turn_speed(actor)
            return Effect(lines=[f"{actor.name} breaks free (Roll  {roll} + {bonus} = {total} vs DC {dc})."])
        return Effect(lines=[f"{actor.name} fails to break free (Roll  {roll} + {bonus} = {total} vs DC {dc})."])

    def _apply_spell(self, actor, spell, slot: int, targets: list[Combatant], label: str) -> Effect:
        effects = []
        parts = spell_parts(actor, spell, slot)
        for index, target in enumerate(targets):
            if spell.index == "magic-missile":
                effects.append(self._magic_missile(actor, target, parts, label))
                if self.pending:
                    self._remember_targets(actor, spell, slot, label, targets[index + 1:])
                    return combine(effects)
                continue
            if spell.kind == "heal":
                expression = heal_expression(actor, spell, slot)
                if expression is None:
                    effects.append(Effect(lines=[f"{spell.name} has no healing at slot {slot}."]))
                    continue
                rolled = roll_dice(expression, self.rng)
                if actor.traits.get("healer"):
                    rolled = reroll_ones(rolled, self.rng)
                amount = rolled.total
                effect = apply_heal(target, amount)
                effect.lines.insert(0, f"{actor.name} casts {label} on {target.name} ({expression} -> {rolled.total}).")
                effect.lines.extend(self._clear_listed(target, spell))
                effects.append(effect)
            elif spell.kind == "attack":
                repeats, beam_parts = _spell_attacks(actor, spell, slot, parts)
                for beam in range(repeats):
                    effects.append(
                        self._rolled_attack(
                            actor,
                            target,
                            bonus=actor.spell_attack_bonus,
                            damage=beam_parts,
                            melee=spell.attack_type == "melee",
                            label=label,
                            option=None,
                            spell=True,
                        )
                    )
                    if self.pending:
                        self.pending["beams_left"] = repeats - beam - 1
                        self.pending["beam_parts"] = beam_parts
                        self.pending["beam_label"] = label
                        self.pending["beam_melee"] = spell.attack_type == "melee"
                        self.pending["beam_actor"] = actor.id
                        self.pending["beam_target"] = target.id
                        self._remember_targets(actor, spell, slot, label, targets[index + 1:])
                        return combine(effects)
                if effects and not self.pending:
                    self._spell_rider(actor, spell, target, effects[-1])
            elif spell.kind == "auto":
                effects.append(resolve_auto(actor, target, damage=parts, rng=self.rng, label=label))
                self._spell_rider(actor, spell, target, effects[-1])
            elif spell.kind == "save":
                success = spell.dc_success or "none"
                dc = actor.spell_save_dc
                found = affliction(spell.index) or {}
                advantage = found.get("condition") == "poisoned" and bool(target.traits.get("poison_ward"))
                effects.append(
                    self._rolled_save(
                        actor,
                        target,
                        dc=dc,
                        ability=spell.dc_ability or "dex",
                        success=success,
                        damage=parts,
                        rng=self.rng,
                        label=label,
                        advantage=advantage,
                    )
                )
                if self.pending:
                    self._remember_targets(actor, spell, slot, label, targets[index + 1 :])
                    return combine(effects)
                if not self.pending:
                    self._spell_rider(actor, spell, target, effects[-1])
            else:
                effects.append(Effect(lines=[f"{spell.name} cannot be resolved."]))
        if not effects:
            effects.append(Effect(lines=[f"{actor.name} casts {label}, but it affects no one."]))
        return combine(effects)

    def answer_reaction(self, choice: str | None) -> None:
        window = self.pending
        if not window:
            raise IllegalAction("No reaction is waiting.")
        legal = {offer["id"] for offer in window.get("offers") or []}
        if choice is not None and choice not in legal:
            raise IllegalAction(str(choice))
        self.pending = None
        phase = window["phase"]
        if phase == "rider":
            self._answer_rider(window, choice)
            return
        reactor = self.by_id(window["reactor_id"])
        if apply_choice(self, reactor, choice, window, phase, ""):
            window["queue"] = []
        if window.get("reask"):
            window.pop("reask", None)
            self.pending = window
            return
        if self._walk(window, phase):
            return
        self._finish_window(window)

    def _walk(self, window: dict, phase: str) -> bool:
        """Ask or auto-resolve the queue. True means a player still has to answer."""
        if window.get("follow_attack") or window.get("follow_spell"):
            pending = list(window.get("queue") or [])
            if pending:
                window["later_reactors"] = list(window.get("later_reactors") or []) + pending
            window["queue"] = []
            return False
        queue = list(window.get("queue") or [])
        window["phase"] = phase
        while queue:
            reactor_id = queue.pop(0)
            window["queue"] = queue
            reactor = self.by_id(reactor_id)
            offers = offers_for(self, reactor, window, phase)
            if not offers:
                continue
            if asks_player(self, reactor):
                window["reactor_id"] = reactor_id
                window["offers"] = offers
                window["prompt"] = prompt_for(self, window, phase, reactor)
                self.pending = window
                return True
            window["offers"] = offers
            choice, reason = heuristic_choice(self, reactor, window, phase, offers)
            if apply_choice(self, reactor, choice, window, phase, reason):
                window["queue"] = []
                return False
            if window.get("follow_attack") or window.get("follow_spell"):
                window["later_reactors"] = list(window.get("queue") or [])
                window["queue"] = []
                return False
            queue = list(window.get("queue") or [])
        window["queue"] = []
        return False

    def _rolled_attack(
        self, actor, target, *, bonus, damage, melee, label, option, spell: bool,
        reaction_attack: bool = False, sentinel_halt: bool = False,
    ) -> Effect:
        protected = self._mark_protection(target)
        self._pack_flag(actor, target)
        try:
            mode = attack_mode(actor, target, melee=melee, spell=spell)
            reason = attack_reason_text(actor, target, melee=melee, spell=spell)
            roll, shown = d20(self.rng, mode)
        finally:
            if protected:
                target.traits.pop("protection_now", None)
            actor.traits.pop("pack_now", None)
        total = roll + bonus
        penalty = attack_penalty(actor)
        crit_min = int(getattr(actor, "crit_min", 20) or 20)
        critical = roll >= crit_min and roll != 1
        ac = opposed_ac(target, melee=melee)
        hit = roll != 1 and (critical or total - penalty >= ac)
        window = {
            "actor_id": actor.id,
            "target_id": target.id,
            "label": label,
            "option_name": option.name if option is not None else None,
            "roll": roll,
            "shown": shown,
            "bonus": bonus,
            "total": total,
            "critical": critical,
            "crit_min": crit_min,
            "melee": melee,
            "spell_attack": spell,
            "penalty": penalty,
            "damage": damage,
            "mode": mode,
            "attack_reason": reason,
            "ac": ac,
            "phase": "hit",
            "reaction_attack": reaction_attack,
            "sentinel_halt": sentinel_halt,
            "on_attack_action": bool(actor.traits.get("on_attack_action")),
            "queue": [],
        }
        if not hit and self._attacker_luck(actor, mode):
            window["phase"] = "luck"
            window["queue"] = [actor.id]
            if self._walk(window, "luck"):
                return Effect()
        effect = self._finish_rolled(window)
        if not self.pending:
            self._maybe_halt(window, effect)
            ended = break_on_attack(actor)
            if ended and effect.lines is not None:
                effect.lines.extend(ended)
            if effect.hit:
                actor.traits["last_hit"] = target.id
        return effect

    def _swap_alert(self, totals: dict) -> list[str]:
        """Swap each Alert creature with the higher ally, lowest total first."""
        lines = []
        owners = [
            creature
            for creature in self.combatants
            if creature.hp > 0 and creature.traits.get("alert") and self._can_act(creature)
        ]
        owners.sort(key=lambda creature: (totals[creature.id], creature.id))
        for owner in owners:
            best = None
            for ally in self.combatants:
                if ally.id == owner.id or ally.side != owner.side or ally.hp <= 0 or not self._can_act(ally):
                    continue
                if totals[ally.id] <= totals[owner.id]:
                    continue
                if best is None or (totals[ally.id], ally.id) > (totals[best.id], best.id):
                    best = ally
            if best is None:
                continue
            totals[owner.id], totals[best.id] = totals[best.id], totals[owner.id]
            lines.append(f"{owner.name} swaps initiative with {best.name}.")
        return lines

    def _can_act(self, creature) -> bool:
        return not is_incapacitated(creature)

    def _can_react(self, creature) -> bool:
        return creature.hp > 0 and creature.reaction_available and self._can_act(creature)

    def _mark_protection(self, target) -> bool:
        protector_id = target.traits.get("protected_by")
        if not protector_id:
            return False
        try:
            protector = self.by_id(protector_id)
        except KeyError:
            return False
        if protector.hp <= 0 or protector.gap_to(target) > 1:
            return False
        target.traits["protection_now"] = True
        return True

    def _attacker_luck(self, actor, mode) -> bool:
        return int(actor.uses_left.get("Luck", 0)) > 0 and int(mode) <= 0

    def _finish_rolled(self, window: dict) -> Effect:
        target = self.by_id(window["target_id"])
        melee = bool(window.get("melee"))
        ac = opposed_ac(target, melee=melee, ac_bonus=int(window.get("ac_bonus") or 0))
        window["ac"] = ac
        total = int(window["total"]) - int(window.get("penalty") or 0)
        critical = bool(window.get("critical"))
        hit = int(window["roll"]) != 1 and (critical or total >= ac)
        window["phase"] = "hit"
        window["queue"] = candidate_ids(self, target) if hit else []
        if hit and self._walk(window, "hit"):
            return Effect()
        return self._deliver_attack(window)

    def _save_pause_needed(self) -> bool:
        for creature in self.combatants:
            if creature.hp <= 0:
                continue
            if creature.traits.get("shield_master"):
                return True
            if int(creature.uses_left.get("Guarded Mind", 0)) > 0:
                return True
        return False

    def _rolled_save(
        self,
        source,
        target,
        *,
        dc,
        ability,
        success,
        damage,
        rng,
        label,
        echo: bool = False,
        advantage: bool = False,
    ) -> Effect:
        if not self._save_pause_needed():
            return resolve_save(
                source,
                target,
                dc=dc,
                ability=ability,
                success=success,
                damage=damage,
                rng=rng,
                label=label,
                echo=echo,
                advantage=advantage,
            )
        full = ability_name(ability)
        mode = save_mode(target, full)
        auto_failed = mode is None
        if auto_failed:
            roll, shown, mode = 0, (0,), 0
        else:
            if advantage:
                mode = min(1, mode + 1)
            roll, shown = d20(rng, mode)
        bonus = target.save_bonuses.get(full, ability_mod(target.abilities.get(full, 10)))
        penalty = 0 if auto_failed else save_penalty(target)
        total = 0 if auto_failed else roll + bonus - penalty
        window = {
            "actor_id": source.id,
            "target_id": target.id,
            "label": label,
            "roll": roll,
            "shown": shown,
            "bonus": bonus,
            "total": total,
            "dc": dc,
            "ability": full,
            "success": success,
            "damage": list(damage),
            "echo": echo,
            "mode": mode,
            "saved": (not auto_failed) and total >= dc,
            "phase": "save",
            "queue": [target.id],
        }
        if self._walk(window, "save"):
            return Effect()
        return self._finish_save(window)

    def _finish_save(self, window: dict) -> Effect:
        return resolve_save(
            self.by_id(window["actor_id"]),
            self.by_id(window["target_id"]),
            dc=int(window["dc"]),
            ability=str(window.get("ability") or "dexterity"),
            success=str(window.get("success") or "none"),
            damage=list(window.get("damage") or []),
            rng=self.rng,
            label=str(window.get("label") or ""),
            echo=bool(window.get("echo")),
            fixed_roll=(int(window["roll"]), tuple(window.get("shown") or (window["roll"],))),
            forced_success=bool(window.get("forced_success")),
            negate_damage=bool(window.get("negate_damage")),
        )

    def _clear_strike(self, window: dict) -> None:
        creature_id = window.get("clear_strike_id")
        if not creature_id:
            return
        if self.pending:
            if self.pending.get("clear_strike_id") is None:
                self.pending["clear_strike_id"] = creature_id
            return
        try:
            self.by_id(creature_id).traits.pop("reaction_strike", None)
        except KeyError:
            return

    def _carry(self, source: dict, dest: dict) -> None:
        if dest is source:
            return
        for key in _CARRY:
            if source.get(key) is not None and key not in dest:
                dest[key] = source[key]

    def _copy_resume(self, old: dict) -> None:
        if self.pending is not None:
            self._carry(old, self.pending)

    def _path_reactions(self) -> bool:
        return any(self._could_react_to_move(creature) for creature in self.combatants)

    def _could_react_to_move(self, creature) -> bool:
        if creature.hp <= 0:
            return False
        if creature.traits.get("polearm_master") or creature.traits.get("war_caster"):
            return True
        if not creature.reaction_available:
            return False
        return self._weapon_reach(creature) > 0

    def _weapon_reach(self, creature) -> int:
        reaches = [
            int(option.reach_feet)
            for option in creature.attacks
            if option.resolution == "attack" and option.reach_feet
        ]
        return max(reaches) if reaches else 0

    def _left_reach(self, old: int, new: int, reach: int) -> bool:
        return reach > 0 and old > 0 and old * 5 <= reach and new * 5 > reach

    def _anchor_gap(self, creature, square, span: int) -> int:
        return space_gap(tuple(creature.position), creature.span, (int(square[0]), int(square[1])), span)

    def _pole_covers(self, creature, gap: int) -> bool:
        if gap <= 0:
            return False
        for option in creature.attacks:
            if pole_weapon(option) and gap * 5 <= int(option.reach_feet or 0):
                return True
        return False

    def _melee_reaches(self, actor, target) -> bool:
        gap = actor.gap_to(target)
        if gap <= 0:
            return False
        for option in actor.attacks:
            if option.resolution == "attack" and option.reach_feet and gap * 5 <= int(option.reach_feet):
                return True
        return False

    def _relocate(self, actor, action) -> Effect:
        if action.destination is None:
            raise IllegalAction(action.label)
        dest = (int(action.destination[0]), int(action.destination[1]))
        origin = (int(actor.position[0]), int(actor.position[1]))
        costs = difficult_costs(self)
        if "prone" in actor.conditions:
            cost = movement_cost(self, actor, origin, dest)
        else:
            cost = travel_cost(origin, dest, actor.span, costs) if costs else feet_between(origin, dest)
        budget = movement_budget(actor)
        if cost <= 0 or cost > budget:
            raise IllegalAction(action.label)
        path = _path_squares(origin, dest)
        if not self._path_reactions():
            stepped = self._zones_along(actor, origin, path)
            if actor.hp <= 0:
                here = (int(actor.position[0]), int(actor.position[1]))
                spent = travel_cost(origin, here, actor.span, costs) if costs else feet_between(origin, here)
                moved = Effect(lines=[f"{actor.name} moves to {here} ({spent} ft). 0 ft left."])
                return combine([moved, stepped])
            actor.position = dest
            actor.movement_left = budget - cost
            actor.traits["moved_feet"] = int(actor.traits.get("moved_feet") or 0) + cost
            self._spend_shift_feet(actor, cost)
            self._note_steps(actor, path)
            break_hide(self, actor)
            moved = Effect(lines=[f"{actor.name} moves to {dest} ({cost} ft). {actor.movement_left} ft left."])
            return combine([
                moved,
                stepped,
                self._sweep_emanation(actor, origin, path),
                self._follow_pack(actor, origin, dest),
            ])
        return self._walk_path(actor, origin, path, budget, cost, dest)

    def _step_feet(self, actor, square) -> int:
        costs = difficult_costs(self)
        step_feet = anchor_cost(square, actor.span, costs) * 5 if costs else 5
        if "prone" in actor.conditions:
            step_feet += 5
        return step_feet

    def _spend_shift_feet(self, actor, feet: int) -> bool:
        """Spend Tactical Shift feet. A step that costs more than the remainder still clears it."""
        left = int(actor.traits.get("shift_feet") or 0)
        if left <= 0 or feet <= 0:
            return False
        if feet > left:
            actor.traits.pop("shift_feet", None)
            return False
        remain = left - feet
        if remain > 0:
            actor.traits["shift_feet"] = remain
        else:
            actor.traits.pop("shift_feet", None)
        return True

    def _shift_pays(self, actor, square) -> bool:
        return self._spend_shift_feet(actor, self._step_feet(actor, square))

    def _leave_events(self, actor, previous, square):
        """Weapon attacks and War Caster, while the mover is still on ``previous``."""
        opportunity = []
        left = []
        span = actor.span
        disengaged = bool(actor.traits.get("disengaged"))
        for creature in self.combatants:
            if creature.id == actor.id or creature.side == actor.side or not self._can_react(creature):
                continue
            if any(mark.get("no_opportunity") for mark in marks_of(creature)):
                continue
            old = self._anchor_gap(creature, previous, span)
            new = self._anchor_gap(creature, square, span)
            sentinel = bool(creature.traits.get("sentinel"))
            if disengaged and not sentinel:
                continue
            if self._left_reach(old, new, self._weapon_reach(creature)):
                opportunity.append(creature.id)
            if creature.traits.get("war_caster") and self._left_reach(old, new, melee_reach(creature)):
                left.append(creature.id)
        seen = set(opportunity) | set(left)
        queue = [creature_id for creature_id in self.order if creature_id in seen]
        return opportunity, left, queue

    def _enter_events(self, actor, previous, square):
        entered = []
        span = actor.span
        for creature in self.combatants:
            if creature.id == actor.id or creature.side == actor.side or not self._can_react(creature):
                continue
            if not creature.traits.get("polearm_master"):
                continue
            old = self._anchor_gap(creature, previous, span)
            new = self._anchor_gap(creature, square, span)
            if self._pole_covers(creature, new) and not self._pole_covers(creature, old):
                entered.append(creature.id)
        queue = [creature_id for creature_id in self.order if creature_id in entered]
        return entered, queue

    def _move_window(
        self,
        actor,
        origin,
        budget,
        cost,
        dest,
        *,
        queue,
        entered,
        left,
        opportunity,
        resume_squares,
        opportunity_done,
    ) -> dict:
        return {
            "actor_id": actor.id,
            "target_id": actor.id,
            "label": "Move",
            "phase": "move",
            "queue": queue,
            "entered": list(entered),
            "left": list(left),
            "opportunity": list(opportunity),
            "resume_squares": list(resume_squares),
            "move_actor": actor.id,
            "move_budget": budget,
            "move_dest": dest,
            "move_cost": cost,
            "move_origin": origin,
            "resume_phase": "move",
            "opportunity_done": opportunity_done,
        }

    def _finish_stopped(self, actor, origin) -> Effect:
        here = (int(actor.position[0]), int(actor.position[1]))
        origin = (int(origin[0]), int(origin[1]))
        if here == origin:
            return Effect()
        costs = difficult_costs(self)
        spent = travel_cost(origin, here, actor.span, costs) if costs else feet_between(origin, here)
        stopped = Effect(lines=[f"{actor.name} moves to {here} ({spent} ft). 0 ft left."])
        return combine([stopped, self._sweep_emanation(actor, origin, _path_squares(origin, here))])

    def _walk_path(self, actor, origin, squares, budget, cost, dest, *, skip_opportunity: bool = False) -> Effect:
        effects = []
        previous = (int(actor.position[0]), int(actor.position[1]))
        skip = skip_opportunity
        for index, raw in enumerate(list(squares)):
            square = (int(raw[0]), int(raw[1]))
            # The swing resolves on the square being left, before this step.
            # Tactical Shift pays for this step and does not provoke, Sentinel included.
            if not skip and not self._shift_pays(actor, square):
                opportunity, left, queue = self._leave_events(actor, previous, square)
                if queue:
                    window = self._move_window(
                        actor,
                        origin,
                        budget,
                        cost,
                        dest,
                        queue=queue,
                        entered=[],
                        left=left,
                        opportunity=opportunity,
                        resume_squares=list(squares[index:]),
                        opportunity_done=True,
                    )
                    effects.append(self._play_reactions(window))
                    if self.pending:
                        return combine(effects)
                    if actor.hp <= 0 or actor.movement_left == 0:
                        effects.append(self._finish_stopped(actor, origin))
                        return combine(effects)
            skip = False
            actor.position = square
            self._note_steps(actor, [square])
            costs = difficult_costs(self)
            step_feet = anchor_cost(square, actor.span, costs) * 5 if costs else 5
            if "prone" in actor.conditions:
                step_feet += 5
            actor.traits["moved_feet"] = int(actor.traits.get("moved_feet") or 0) + step_feet
            break_hide(self, actor)
            effects.append(self._zones_along(actor, previous, [square]))
            if actor.hp <= 0 or actor.movement_left == 0:
                effects.append(self._finish_stopped(actor, origin))
                return combine(effects)
            entered, queue = self._enter_events(actor, previous, square)
            previous = square
            if not queue:
                continue
            window = self._move_window(
                actor,
                origin,
                budget,
                cost,
                dest,
                queue=queue,
                entered=entered,
                left=[],
                opportunity=[],
                resume_squares=list(squares[index + 1 :]),
                opportunity_done=False,
            )
            effects.append(self._play_reactions(window))
            if self.pending:
                return combine(effects)
            if actor.hp <= 0 or actor.movement_left == 0:
                effects.append(self._finish_stopped(actor, origin))
                return combine(effects)
        actor.movement_left = budget - cost
        effects.append(Effect(lines=[f"{actor.name} moves to {dest} ({cost} ft). {actor.movement_left} ft left."]))
        effects.append(self._sweep_emanation(actor, origin, _path_squares(origin, tuple(dest))))
        effects.append(self._follow_pack(actor, origin, tuple(dest)))
        return combine(effects)

    def _resume_move(self, window: dict) -> Effect:
        try:
            actor = self.by_id(window.get("move_actor"))
        except KeyError:
            return Effect()
        origin = tuple(window.get("move_origin") or actor.position)
        if actor.hp <= 0 or actor.movement_left == 0:
            return self._finish_stopped(actor, origin)
        squares = list(window.get("resume_squares") or [])
        budget = int(window.get("move_budget") or 0)
        cost = int(window.get("move_cost") or 0)
        dest = tuple(window.get("move_dest") or actor.position)
        if not squares:
            actor.movement_left = budget - cost
            finished = Effect(lines=[f"{actor.name} moves to {dest} ({cost} ft). {actor.movement_left} ft left."])
            return combine([finished, self._sweep_emanation(actor, origin, _path_squares(origin, dest))])
        return self._walk_path(
            actor,
            origin,
            squares,
            budget,
            cost,
            dest,
            skip_opportunity=bool(window.get("opportunity_done")),
        )

    def _play_reactions(self, window: dict) -> Effect:
        if window.get("follow_attack") or window.get("follow_spell"):
            pending = list(window.get("queue") or [])
            if pending:
                window["later_reactors"] = list(window.get("later_reactors") or []) + pending
            window["queue"] = []
        elif self._walk(window, window.get("phase") or "sentinel"):
            return Effect()
        effect = self._take_follow(window)
        if self.pending:
            self._copy_resume(window)
            return effect
        rest = list(window.pop("later_reactors", []) or [])
        if rest:
            window["queue"] = rest
            window.pop("follow_attack", None)
            window.pop("follow_spell", None)
            return combine([effect, self._play_reactions(window)])
        return effect

    def _continue_later(self, window: dict) -> Effect:
        rest = list(window.get("later_reactors") or [])
        if not rest:
            return Effect()
        phase = str(window.get("resume_phase") or "sentinel")
        window["later_reactors"] = []
        nxt = {
            "actor_id": window.get("move_actor") or window.get("actor_id"),
            "target_id": window.get("target_id"),
            "label": window.get("label") or "Reaction",
            "phase": phase,
            "queue": rest,
            "entered": list(window.get("entered") or []),
            "left": list(window.get("left") or []),
            "opportunity": list(window.get("opportunity") or []),
            "opportunity_done": bool(window.get("opportunity_done")),
            "resume_squares": window.get("resume_squares"),
            "move_actor": window.get("move_actor"),
            "move_budget": window.get("move_budget"),
            "move_dest": window.get("move_dest"),
            "move_cost": window.get("move_cost"),
            "move_origin": window.get("move_origin"),
            "resume_phase": phase,
        }
        return self._play_reactions(nxt)

    def _sentinel_reactors(self, quarry) -> list[str]:
        reactors = []
        for creature_id in self.order:
            reactor = self.by_id(creature_id)
            if reactor.id == quarry.id or reactor.side == quarry.side or not reactor.traits.get("sentinel"):
                continue
            if not self._can_react(reactor) or not (0 < reactor.gap_to(quarry) <= 1):
                continue
            if not self._melee_reaches(reactor, quarry):
                continue
            reactors.append(reactor.id)
        return reactors

    def _sentinel_disengage(self, actor) -> Effect:
        reactors = self._sentinel_reactors(actor)
        if not reactors:
            return Effect()
        return self._play_reactions({
            "actor_id": actor.id,
            "target_id": actor.id,
            "label": "Disengage",
            "phase": "sentinel",
            "queue": reactors,
            "resume_phase": "sentinel",
        })

    def _take_follow(self, window: dict) -> Effect:
        info = window.pop("follow_attack", None)
        spell = window.pop("follow_spell", None)
        if info:
            effect = self._reaction_attack(window, info)
        elif spell:
            effect = self._reaction_spell(window, spell)
        else:
            effect = Effect()
        if self.pending is not None and self.pending is not window:
            self._copy_resume(window)
        return effect

    def _reaction_attack(self, parent: dict, info: dict) -> Effect:
        reactor = self.by_id(info["reactor_id"])
        target = self.by_id(info["target_id"])
        option = reactor.attack_named(info.get("weapon") or "")
        if option is None:
            return Effect()
        reactor.traits["reaction_strike"] = True
        try:
            return self._strike(
                reactor,
                target,
                option,
                reaction_attack=True,
                sentinel_halt=bool(info.get("halt")),
            )
        finally:
            if self.pending:
                if self.pending.get("clear_strike_id") is None:
                    self.pending["clear_strike_id"] = reactor.id
                self._carry(parent, self.pending)
            else:
                reactor.traits.pop("reaction_strike", None)

    def _reaction_spell(self, parent: dict, info: dict) -> Effect:
        reactor = self.by_id(info["reactor_id"])
        target = self.by_id(info["target_id"])
        spell = reactor.spell_named(info.get("spell_index"))
        if spell is None:
            return Effect()
        slot = int(info.get("slot") or 0)
        if slot > 0 and self.current().id == reactor.id:
            reactor.traits["leveled_cast"] = True
        reactor.traits["reaction_strike"] = True
        window = {
            "actor_id": reactor.id,
            "target_id": target.id,
            "spell_index": spell.index,
            "slot": slot,
            "label": spell.name,
            "target_ids": [target.id],
            "phase": "counterspell",
            "queue": candidate_ids(self, None),
            "clear_strike_id": reactor.id,
        }
        self._carry(parent, window)
        try:
            if self._walk(window, "counterspell"):
                return Effect()
            if window.get("countered"):
                return Effect()
            return self._commit_spell(reactor, spell, slot, [target], spell.name)
        finally:
            if self.pending:
                if self.pending.get("clear_strike_id") is None:
                    self.pending["clear_strike_id"] = reactor.id
                self._carry(window, self.pending)
            else:
                reactor.traits.pop("reaction_strike", None)

    def _maybe_halt(self, window: dict, effect: Effect) -> None:
        if not window.get("sentinel_halt") or not effect.hit or window.get("halted"):
            return
        target = self.by_id(window["target_id"])
        target.movement_left = 0
        window["halted"] = True
        effect.lines.append(f"{target.name} stops. Speed is 0 for the rest of the turn.")

    def _after_attack_extras(self, actor, target, option, effect, *, reaction_attack: bool = False) -> None:
        self._run_extras(
            actor,
            target,
            option,
            effect,
            reaction_attack=reaction_attack,
            on_attack=bool(actor.traits.get("on_attack_action")),
        )

    def _finish_attack_extras(self, window: dict, effect: Effect) -> None:
        actor = self.by_id(window["actor_id"])
        target = self.by_id(window["target_id"])
        option = actor.attack_named(window.get("option_name")) if window.get("option_name") else None
        if option is None and window.get("option_name") == "Pole strike":
            option = AttackOption(
                name="Pole strike",
                resolution="attack",
                damage=[{"type": "bludgeoning", "expr": "1d4"}],
                reach_feet=5,
            )
        self._run_extras(
            actor,
            target,
            option,
            effect,
            reaction_attack=bool(window.get("reaction_attack")),
            on_attack=bool(window.get("on_attack_action")),
        )

    def _run_extras(self, actor, target, option, effect, *, reaction_attack: bool, on_attack: bool) -> None:
        if not effect.hit or option is None:
            return
        if effect.lines is None:
            effect.lines = []
        self._hit_riders(actor, target, option, effect)
        if self.pending:
            return
        steps = [] if reaction_attack else self._rider_steps(actor, target, option, on_attack=on_attack)
        if steps and asks_player(self, actor):
            self._open_rider(actor, target, option, steps, on_attack=on_attack, reaction_attack=reaction_attack)
            return
        for step in steps:
            choice = self._heuristic_rider(actor, target, option, step)
            self._apply_rider_step(actor, target, option, step, choice, effect)
        self._automatic_extras(actor, target, option, effect, reaction_attack=reaction_attack, on_attack=on_attack)

    def _automatic_extras(self, actor, target, option, effect, *, reaction_attack: bool, on_attack: bool) -> None:
        if not effect.hit or option is None:
            return
        if effect.lines is None:
            effect.lines = []
        rider = option.rider or {}
        feet = int(rider.get("push_feet") or 0)
        if feet > 0:
            self._push_target(actor, target, max(1, feet // 5), effect, f"{option.name} pushes {target.name}")
        gap = actor.gap_to(target)
        if (
            on_attack
            and not reaction_attack
            and actor.traits.get("shield_master")
            and actor.traits.get("has_shield")
            and not actor.traits.get("shield_bash_used")
            and option.reach_feet
            and gap == 1
        ):
            self._shield_bash(actor, target, effect)
        self._mastery_follow(actor, target, option, effect)
        if self.pending or reaction_attack:
            return
        extra = self._offer_sentinel_hit(actor, target)
        effect.damage += extra.damage
        effect.lines.extend(extra.lines or [])

    def _push_target(self, actor, target, squares: int, effect: Effect, lead: str) -> None:
        origin = tuple(target.position)
        dest, visited = push_from(self, actor, target, int(squares))
        effect.lines.append(f"{lead} to {dest}.")
        self._fold_emanation(effect, target, origin, visited)

    def _push_dest(self, actor, target, squares: int):
        origin = tuple(target.position)
        dest, _visited = push_from(self, actor, target, int(squares))
        target.position = origin
        return dest

    def _crusher_squares(self, target) -> list[tuple[int, int]]:
        origin = tuple(target.position)
        occupied: set[tuple[int, int]] = set()
        for creature in self.combatants:
            if creature.hp > 0 and creature.id != target.id:
                occupied.update(creature.squares())
        squares = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nxt = (origin[0] + dx, origin[1] + dy)
                body = footprint(nxt, target.span)
                if any(not in_bounds(cell, self.width, self.height) or cell in occupied for cell in body):
                    continue
                squares.append(nxt)
        return squares

    def _auto_succeeds(self, target, ability: str, dc: int) -> bool:
        full = ability_name(ability)
        if save_mode(target, full) is None:
            return False
        bonus = target.save_bonuses.get(full, ability_mod(target.abilities.get(full, 10)))
        return 1 + int(bonus) >= int(dc)

    def _best_save(self, target, names) -> str:
        best_name = ability_name(str(names[0] if names else "strength"))
        best_bonus = None
        for name in names or ["strength"]:
            full = ability_name(str(name))
            bonus = target.save_bonuses.get(full, ability_mod(target.abilities.get(full, 10)))
            if best_bonus is None or bonus > best_bonus:
                best_bonus = bonus
                best_name = full
        return best_name

    def _nick_after_swing(self, actor, option) -> bool:
        if actor.traits.get("nick_used") or self._is_nick_swing(actor, option):
            return False
        previous = actor.traits.get("light_used")
        if option is not None and getattr(option, "light", False) and not previous:
            actor.traits["light_used"] = option.name
            try:
                return nick_weapon_name(actor) is not None
            finally:
                actor.traits.pop("light_used", None)
        return nick_weapon_name(actor) is not None

    def _later_melee(self, actor, option) -> bool:
        budget = str(actor.traits.get("swing_budget") or "")
        tag = str(actor.traits.get("swing_tag") or "")
        made = int(actor.attacks_made or 0)
        per = max(1, int(actor.attacks_per_action or 1))
        more = 0
        if budget == "action" and not self._is_nick_swing(actor, option):
            if made + 1 < per:
                more += 1
            if self._nick_after_swing(actor, option):
                more += 1
        flurry = int(actor.traits.get("flurry_left") or 0)
        if tag == "flurry":
            flurry = max(0, flurry - 1)
        more += flurry
        monk = actor.traits.get("monk") or {}
        if tag != "flurry" and actor.bonus_available and isinstance(monk, dict):
            focus = str(monk.get("focus") or "")
            left = int(actor.uses_left.get(focus, 0)) if focus else 0
            if monk.get("flurry") and left > 0:
                more += 2
            elif actor.traits.get("monk_attacked") and monk.get("bonus_unarmed"):
                more += 1
        return more > 0

    def _later_melee_reach(self, actor, option) -> int | None:
        if not self._later_melee(actor, option):
            return None
        reaches = []
        for item in actor.attacks:
            rider = item.rider or {}
            if not item.reach_feet:
                continue
            if item.resolution == "attack" or rider.get("grapple") or rider.get("shove"):
                reaches.append(int(item.reach_feet))
        return min(reaches) if reaches else 5

    def _accept_push(self, actor, target, option, squares: int) -> bool:
        origin = tuple(target.position)
        dest = self._push_dest(actor, target, squares)
        if dest == origin:
            return False
        reach = self._later_melee_reach(actor, option)
        if reach is None:
            return True
        gap = space_gap(tuple(actor.position), actor.span, dest, target.span)
        return gap * 5 <= reach

    def _crusher_choice(self, actor, target, option):
        reach = self._later_melee_reach(actor, option)
        current = actor.gap_to(target)
        best = None
        best_gap = current
        for square in self._crusher_squares(target):
            gap = space_gap(tuple(actor.position), actor.span, square, target.span)
            if gap <= current:
                continue
            if reach is not None and gap * 5 > reach:
                continue
            if best is None or gap > best_gap:
                best = square
                best_gap = gap
        return best

    def _rider_steps(self, actor, target, option, *, on_attack: bool) -> list[dict]:
        steps = []
        rider = option.rider or {}
        monk = actor.traits.get("monk") if isinstance(actor.traits.get("monk"), dict) else {}
        gap = actor.gap_to(target)
        unarmed = bool(rider.get("unarmed"))
        monk_weapon = bool(rider.get("monk_weapon") or unarmed)
        focus = str(monk.get("focus") or "")
        focus_left = int(actor.uses_left.get(focus, 0)) if focus else 0
        tag = str(actor.traits.get("swing_tag") or "")
        if monk.get("stunning") and monk_weapon and not actor.traits.get("stunning_used") and focus_left > 0:
            steps.append({"kind": "stun", "decline": "Don't use Stunning Strike"})
        techniques = [name for name in ("addle", "push", "topple") if name in set(monk.get("open_hand") or [])]
        if tag == "flurry" and techniques:
            steps.append({"kind": "open", "decline": "No technique", "techniques": techniques})
        if option.mastery == "push" and size_rank(getattr(target, "size", None)) <= size_rank("large"):
            steps.append({"kind": "mastery", "decline": "Don't push", "squares": 2})
        if (
            on_attack
            and unarmed
            and actor.traits.get("tavern_brawler")
            and not actor.traits.get("tavern_push_used")
            and gap == 1
        ):
            steps.append({"kind": "tavern", "decline": "Don't push", "squares": 1})
        kind = str(option.damage[0].get("type") or "") if option.damage else ""
        if (
            actor.traits.get("crusher")
            and not actor.traits.get("crusher_used")
            and kind == "bludgeoning"
            and gap == 1
            and size_rank(getattr(target, "size", None)) <= size_rank(getattr(actor, "size", None)) + 1
            and self._crusher_squares(target)
        ):
            steps.append({"kind": "crusher", "decline": "Don't push"})
        return steps

    def _rider_offers(self, actor, target, step: dict) -> list[dict]:
        kind = step.get("kind")
        if kind == "stun":
            return [{"id": "stun", "label": "Stunning Strike"}]
        if kind == "open":
            labels = {"addle": "Addle", "push": "Push", "topple": "Topple"}
            return [{"id": name, "label": labels[name]} for name in step.get("techniques") or [] if name in labels]
        if kind in {"mastery", "tavern"}:
            feet = int(step.get("squares") or 1) * 5
            return [{"id": "yes", "label": f"Push {feet} feet"}]
        if kind == "crusher":
            return [
                {"id": f"{square[0]},{square[1]}", "label": f"Move to ({square[0]}, {square[1]})"}
                for square in self._crusher_squares(target)
            ]
        return []

    def _rider_prompt(self, actor, target, step: dict) -> str:
        kind = step.get("kind")
        if kind == "stun":
            return f"{actor.name} can spend 1 Focus to use Stunning Strike on {target.name}."
        if kind == "open":
            return f"{actor.name} can use an Open Hand technique on {target.name}."
        if kind == "tavern":
            return f"{actor.name} can push {target.name} 5 feet."
        if kind == "mastery":
            return f"{actor.name} can push {target.name} 10 feet."
        return f"{actor.name} can move {target.name} 5 feet."

    def _open_rider(self, actor, target, option, steps, *, on_attack: bool, reaction_attack: bool) -> None:
        step = steps[0]
        self.pending = {
            "phase": "rider",
            "actor_id": actor.id,
            "target_id": target.id,
            "reactor_id": actor.id,
            "label": option.name,
            "option_name": option.name,
            "offers": self._rider_offers(actor, target, step),
            "prompt": self._rider_prompt(actor, target, step),
            "decline": step.get("decline") or "Don't",
            "spends_reaction": False,
            "rider_steps": steps,
            "rider_index": 0,
            "on_attack_action": bool(on_attack),
            "reaction_attack": bool(reaction_attack),
            "rider_lines": [],
            "rider_damage": 0,
        }

    def _answer_rider(self, window: dict, choice: str | None) -> None:
        actor = self.by_id(window["actor_id"])
        target = self.by_id(window["target_id"])
        option = actor.attack_named(window.get("option_name") or "") if window.get("option_name") else None
        steps = list(window.get("rider_steps") or [])
        index = int(window.get("rider_index") or 0)
        effect = Effect(lines=list(window.get("rider_lines") or []), damage=int(window.get("rider_damage") or 0))
        if option is not None and index < len(steps):
            self._apply_rider_step(actor, target, option, steps[index], choice, effect)
        index += 1
        if index < len(steps):
            step = steps[index]
            window["rider_index"] = index
            window["rider_lines"] = list(effect.lines or [])
            window["rider_damage"] = int(effect.damage)
            window["offers"] = self._rider_offers(actor, target, step)
            window["prompt"] = self._rider_prompt(actor, target, step)
            window["decline"] = step.get("decline") or "Don't"
            window["phase"] = "rider"
            window["spends_reaction"] = False
            self.pending = window
            return
        if option is not None:
            self._automatic_extras(
                actor,
                target,
                option,
                effect,
                reaction_attack=bool(window.get("reaction_attack")),
                on_attack=bool(window.get("on_attack_action")),
            )
        if self.pending and self.pending is not window:
            self._keep_publish(window)
            pending = self.pending
            pending["prefix_lines"] = list(window.get("prefix_lines") or []) + list(effect.lines or [])
            pending["prefix_damage"] = int(window.get("prefix_damage") or 0) + int(effect.damage)
            pending["prefix_healing"] = int(window.get("prefix_healing") or 0) + int(pending.get("prefix_healing") or 0)
            return
        window["rider_lines"] = list(effect.lines or [])
        window["rider_damage"] = int(effect.damage)
        self._finish_rider(window)

    def _finish_rider(self, window: dict) -> None:
        effect = Effect(lines=list(window.get("rider_lines") or []), damage=int(window.get("rider_damage") or 0))
        self._publish_window(window, effect)
        if self.over:
            return
        self.refresh_positional()
        for line in self._flush_saves():
            if line:
                self.log.append(line)
        self._reap()

    def _heuristic_rider(self, actor, target, option, step: dict):
        kind = step.get("kind")
        if kind == "stun":
            dc = int((actor.traits.get("monk") or {}).get("save_dc") or 10)
            if self._auto_succeeds(target, "constitution", dc):
                return None
            return "stun"
        if kind == "open":
            techniques = list(step.get("techniques") or [])
            if "prone" not in target.conditions and "topple" in techniques:
                return "topple"
            if "push" in techniques:
                return "push"
            return techniques[0] if techniques else None
        if kind in {"mastery", "tavern"}:
            if self._accept_push(actor, target, option, int(step.get("squares") or 1)):
                return "yes"
            return None
        if kind == "crusher":
            square = self._crusher_choice(actor, target, option)
            if square is None:
                return None
            return f"{square[0]},{square[1]}"
        return None

    def _apply_rider_step(self, actor, target, option, step: dict, choice: str | None, effect: Effect) -> None:
        kind = step.get("kind")
        if not choice:
            if kind == "stun":
                effect.lines.append(f"{actor.name} does not use Stunning Strike.")
            elif kind == "open":
                effect.lines.append(f"{actor.name} uses no technique.")
            else:
                effect.lines.append(f"{actor.name} does not push.")
            return
        if kind == "stun":
            self._stunning_strike(actor, target, effect)
            return
        if kind == "open":
            self._open_hand(actor, target, choice, effect)
            return
        if kind in {"mastery", "tavern"} and choice == "yes":
            if kind == "tavern":
                actor.traits["tavern_push_used"] = True
                lead = f"Tavern Brawler pushes {target.name}"
            else:
                lead = f"Push moves {target.name}"
            self._push_target(actor, target, int(step.get("squares") or 1), effect, lead)
            return
        if kind == "crusher" and "," in str(choice):
            raw_x, raw_y = str(choice).split(",", 1)
            square = (int(raw_x), int(raw_y))
            origin = tuple(target.position)
            target.position = square
            actor.traits["crusher_used"] = True
            effect.lines.append(f"Crusher moves {target.name} to {square}.")
            self._fold_emanation(effect, target, origin, [square])

    def _stunning_strike(self, actor, target, effect: Effect) -> None:
        if not spend_focus(actor):
            effect.lines.append(f"{actor.name} has no Focus left.")
            return
        actor.traits["stunning_used"] = True
        dc = int((actor.traits.get("monk") or {}).get("save_dc") or 10)
        save = resolve_save(
            actor,
            target,
            dc=dc,
            ability="constitution",
            success="none",
            damage=[],
            rng=self.rng,
            label="Stunning Strike",
        )
        effect.lines.append(f"{actor.name} spends 1 Focus on Stunning Strike.")
        effect.lines.extend(save.lines or [])
        if self._failed(save):
            note = add_mark(target, actor, condition="stunned", until="start", spell="Stunning Strike")
            if note:
                effect.lines.append(note)
            return
        add_mark(target, actor, until="start", half_speed=True, spell="Stunning Strike")
        add_mark(target, actor, until="start", next_attack=True, spell="Stunning Strike")
        effect.lines.append(f"{target.name}'s speed is halved until {actor.name}'s next turn.")
        effect.lines.append(f"The next attack against {target.name} has advantage.")

    def _open_hand(self, actor, target, choice: str, effect: Effect) -> None:
        dc = int((actor.traits.get("monk") or {}).get("save_dc") or 10)
        if choice == "addle":
            add_mark(target, actor, until="own", no_opportunity=True, spell="Addle")
            effect.lines.append(f"{target.name} cannot take opportunity attacks until their next turn.")
            return
        if choice == "push":
            save = resolve_save(
                actor, target, dc=dc, ability="strength", success="none", damage=[], rng=self.rng, label="Open Hand Push",
            )
            effect.lines.extend(save.lines or [])
            if self._failed(save):
                self._push_target(actor, target, 3, effect, f"Open Hand pushes {target.name}")
            return
        if choice == "topple":
            save = resolve_save(
                actor, target, dc=dc, ability="dexterity", success="none", damage=[], rng=self.rng, label="Open Hand Topple",
            )
            effect.lines.extend(save.lines or [])
            if self._failed(save):
                add_mark(target, actor, condition="prone", until="stand")
                effect.lines.append(f"{target.name} is knocked prone.")

    def _monk_contest(self, actor, target, option) -> Effect:
        rider = option.rider or {}
        ability = self._best_save(target, list(rider.get("save_choice") or [option.dc_ability or "strength"]))
        effect = self._rolled_save(
            actor,
            target,
            dc=option.dc_value or 10,
            ability=ability,
            success="none",
            damage=[],
            rng=self.rng,
            label=option.name,
        )
        if self.pending or not self._failed(effect):
            return effect
        if rider.get("grapple"):
            note = add_mark(
                target,
                actor,
                condition="grappled",
                until="fight",
                grapple=True,
                escape_dc=int(rider.get("escape_dc") or option.dc_value or 10),
                reach=int(option.reach_feet or 5),
            )
            if note:
                effect.lines.append(note)
        elif rider.get("shove") == "push":
            self._push_target(actor, target, int(rider.get("squares") or 1), effect, f"Shove moves {target.name}")
        elif rider.get("shove") == "prone":
            add_mark(target, actor, condition="prone", until="stand")
            effect.lines.append(f"{target.name} is knocked prone.")
        return effect

    def _redirect_targets(self, reactor, window: dict) -> list:
        monk = reactor.traits.get("monk") if isinstance(reactor.traits.get("monk"), dict) else {}
        info = monk.get("redirect") or {}
        focus = str(monk.get("focus") or "")
        if not info or not focus or int(reactor.uses_left.get(focus, 0)) <= 0:
            return []
        melee = bool(window.get("melee"))
        limit = int(info.get("melee") if melee else info.get("ranged") or 0)
        attacker_id = window.get("actor_id")
        dc = int(monk.get("save_dc") or 10)
        found = []
        for creature in self.combatants:
            if creature.hp <= 0 or creature.id == reactor.id:
                continue
            if not can_see(reactor, creature):
                continue
            gap = reactor.gap_to(creature)
            if gap <= 0 or gap * 5 > limit:
                continue
            if self._auto_succeeds(creature, "dexterity", dc):
                continue
            found.append(creature)
        found.sort(key=lambda creature: (creature.id != attacker_id, creature.id))
        return found

    def deflect_hit(self, reactor, window: dict) -> bool:
        info = ((reactor.traits.get("monk") or {}).get("deflect") or {})
        expr = str(info.get("expr") or "1d10")
        rolled = roll_dice(expr, self.rng)
        amount = int(rolled.total)
        left = amount
        parts = window.get("rolled_parts") or []
        for index, (kind, value) in enumerate(list(parts)):
            cut = min(int(value), left)
            parts[index] = (kind, int(value) - cut)
            left -= cut
            if left <= 0:
                break
        reactor.reaction_available = False
        notes = window.get("rider_notes")
        if not isinstance(notes, list):
            notes = []
            window["rider_notes"] = notes
        notes.append(f"Deflect Attack reduces the damage by {amount} ({rolled.expression}).")
        remaining = sum(int(value) for _, value in parts)
        targets = self._redirect_targets(reactor, window) if remaining <= 0 else []
        if not targets:
            return False
        if asks_player(self, reactor):
            window["offers"] = [
                {"id": f"redirect:{creature.id}", "label": f"Redirect to {creature.name}"}
                for creature in targets
            ]
            window["prompt"] = f"{reactor.name} reduced the damage to 0 and can redirect it."
            window["decline"] = "Don't redirect"
            window["spends_reaction"] = False
            window["reask"] = True
            return False
        self.redirect_deflect(reactor, targets[0], window)
        return False

    def redirect_deflect(self, reactor, target, window: dict) -> None:
        if not spend_focus(reactor):
            return
        monk = reactor.traits.get("monk") if isinstance(reactor.traits.get("monk"), dict) else {}
        info = monk.get("redirect") or {}
        expr = str(info.get("expr") or "1d8")
        rolled = roll_dice(expr, self.rng)
        faces = " + ".join(str(face) for face in rolled.rolls) if rolled.rolls else "0"
        dc = int(monk.get("save_dc") or 10)
        mode = save_mode(target, "dexterity")
        notes = window.get("rider_notes")
        if not isinstance(notes, list):
            notes = []
            window["rider_notes"] = notes
        kind = "bludgeoning"
        parts = window.get("rolled_parts") or []
        if parts:
            kind = str(parts[0][0] or kind)
        if mode is None:
            saved = False
            bonus = 0
            shown = (0,)
            total = 0
        else:
            bonus = target.save_bonuses.get("dexterity", ability_mod(target.abilities.get("dexterity", 10)))
            roll, shown = d20(self.rng, mode)
            total = roll + bonus
            saved = total >= dc
        if saved:
            notes.append(f"{target.name} resists the redirect ({shown[0]}+{bonus}={total} vs DC {dc}).")
        else:
            dealt = apply_damage(target, [(kind, rolled.total)], self.rng, source=reactor)
            window["extra_damage"] = int(window.get("extra_damage") or 0) + int(dealt.damage)
            notes.append(f"Deflect Attack redirects {rolled.expression} ({faces}) to {target.name}.")
            notes.extend(dealt.lines or [])

    def _shield_bash(self, actor, target, effect: Effect) -> None:
        actor.traits["shield_bash_used"] = True
        dc = 8 + int(actor.traits.get("proficiency") or 2) + ability_mod(actor.abilities.get("strength", 10))
        bonus = target.save_bonuses.get("strength", ability_mod(target.abilities.get("strength", 10)))
        roll, shown = d20(self.rng, 0)
        total = roll + bonus
        if total >= dc:
            effect.lines.append(f"{target.name} resists Shield Bash ({shown[0]}+{bonus}={total} vs DC {dc}).")
            return
        add_mark(target, actor, condition="prone", until="stand")
        effect.lines.append(f"{target.name} is knocked prone by Shield Bash ({shown[0]}+{bonus}={total} vs DC {dc}).")

    def _offer_sentinel_hit(self, actor, target) -> Effect:
        reactors = []
        for creature_id in self.order:
            reactor = self.by_id(creature_id)
            if reactor.id in {actor.id, target.id} or reactor.side == actor.side or not reactor.traits.get("sentinel"):
                continue
            if not self._can_react(reactor) or not (0 < reactor.gap_to(actor) <= 1):
                continue
            if not self._melee_reaches(reactor, actor):
                continue
            reactors.append(reactor.id)
        if not reactors:
            return Effect()
        return self._play_reactions({
            "actor_id": actor.id,
            "target_id": actor.id,
            "label": "Sentinel",
            "phase": "sentinel",
            "queue": reactors,
            "resume_phase": "sentinel",
        })

    def _deliver_attack(self, window: dict) -> Effect:
        actor = self.by_id(window["actor_id"])
        target = self.by_id(window["target_id"])
        option = actor.attack_named(window["option_name"]) if window.get("option_name") else None

        def before_damage(parts, riders, dice_bits, rider_notes, math_line, title, critical):
            window["rolled_parts"] = parts
            window["riders"] = riders
            window["dice_bits"] = dice_bits
            window["rider_notes"] = rider_notes
            window["math_line"] = math_line
            window["title"] = title
            window["critical"] = critical
            window["queue"] = candidate_ids(self, target)
            if self._walk(window, "damage"):
                return "pause"
            return None

        effect = resolve_attack(
            actor,
            target,
            bonus=int(window["bonus"]),
            damage=list(window["damage"]),
            rng=self.rng,
            melee=bool(window["melee"]),
            label=window["label"],
            option=option,
            allies=self.combatants,
            spell=bool(window.get("spell_attack")),
            fixed_roll=(int(window["roll"]), tuple(window["shown"])),
            penalty=int(window.get("penalty") or 0),
            ac_bonus=int(window.get("ac_bonus") or 0),
            before_damage=before_damage,
            mode_override=window.get("mode_override") if window.get("mode_override") is not None else window.get("mode"),
            reason="" if window.get("mode_override") is not None else str(window.get("attack_reason") or ""),
        )
        extra = int(window.get("extra_damage") or 0)
        if extra and not self.pending:
            effect.damage += extra
        return effect

    def _complete_saved_hit(self, window: dict) -> Effect:
        actor = self.by_id(window["actor_id"])
        target = self.by_id(window["target_id"])
        option = actor.attack_named(window["option_name"]) if window.get("option_name") else None
        parts = list(window.get("rolled_parts") or []) + list(window.get("riders") or [])
        effect = complete_hit(
            actor,
            target,
            parts=parts,
            dice_bits=list(window.get("dice_bits") or []),
            rider_notes=list(window.get("rider_notes") or []),
            math_line=str(window.get("math_line") or ""),
            title=str(window.get("title") or "Hit"),
            critical=bool(window.get("critical")),
            melee=bool(window.get("melee")),
            option=option,
            rng=self.rng,
            label=window["label"],
            spell=bool(window.get("spell_attack")),
        )
        extra = int(window.get("extra_damage") or 0)
        if extra:
            effect.damage += extra
        return effect

    def _magic_missile(self, actor, target, parts, label: str) -> Effect:
        if target.traits.get("shield_ac"):
            return Effect(lines=[f"{target.name} is shielded. Magic Missile deals no damage."])
        window = {
            "actor_id": actor.id,
            "target_id": target.id,
            "label": label,
            "spell_index": "magic-missile",
            "missile_parts": parts,
            "phase": "missile",
            "queue": candidate_ids(self, target),
        }
        if self._walk(window, "missile"):
            return Effect()
        if target.traits.get("shield_ac"):
            return Effect(lines=[f"Magic Missile deals no damage to {target.name}."])
        return resolve_auto(actor, target, damage=parts, rng=self.rng, label=label)

    def _finish_window(self, window: dict) -> None:
        phase = window.get("phase")
        if phase == "counterspell":
            effect = Effect()
            if not window.get("countered"):
                actor = self.by_id(window["actor_id"])
                spell = actor.spell_named(window.get("spell_index"))
                slot = int(window.get("slot") or 0)
                if spell is not None and spell.index == "spiritual-weapon":
                    aim = window.get("aim")
                    placed = GameAction(
                        id=-1,
                        kind="spell",
                        label=str(window.get("label") or spell.name),
                        spell_index=spell.index,
                        aim=tuple(aim) if aim else None,
                        slot_level=slot,
                        budget="bonus",
                    )
                    effect = self._place_spiritual_weapon(actor, spell, slot, placed)
                else:
                    targets = [self.by_id(creature_id) for creature_id in window.get("target_ids") or []]
                    effect = self._commit_spell(actor, spell, slot, targets, window["label"])
        elif phase == "luck":
            effect = self._finish_rolled(window)
            if not self.pending and effect.hit:
                self._finish_attack_extras(window, effect)
                self._maybe_halt(window, effect)
        elif phase == "save":
            effect = self._finish_save(window)
            source = self.by_id(window["actor_id"])
            target = self.by_id(window["target_id"])
            option = source.attack_named(str(window.get("label") or ""))
            self._mark_prone_from_save(source, target, option.rider if option is not None else {}, effect)
        elif phase in {"move", "sentinel"}:
            effect = self._take_follow(window)
            if not self.pending:
                rest = list(window.pop("later_reactors", []) or [])
                if rest:
                    window["queue"] = rest
                    window["phase"] = phase
                    effect = combine([effect, self._play_reactions(window)])
            if not self.pending and phase == "move":
                effect = combine([effect, self._resume_move(window)])
        elif phase == "hit":
            effect = self._deliver_attack(window)
            if not self.pending:
                self._finish_attack_extras(window, effect)
                self._maybe_halt(window, effect)
        elif phase == "damage":
            effect = self._complete_saved_hit(window)
            if not self.pending:
                self._finish_attack_extras(window, effect)
                self._maybe_halt(window, effect)
        elif phase == "missile":
            actor = self.by_id(window["actor_id"])
            target = self.by_id(window["target_id"])
            if target.traits.get("shield_ac"):
                effect = Effect(lines=[f"Magic Missile deals no damage to {target.name}."])
            else:
                effect = resolve_auto(actor, target, damage=window.get("missile_parts") or [], rng=self.rng, label=window["label"])
        else:
            effect = Effect()
        if self.pending:
            self._keep_publish(window)
            self._add_prefix(effect)
            self._clear_strike(window)
            return
        if window.get("later_reactors") and phase not in {"move", "sentinel"}:
            effect = combine([effect, self._continue_later(window)])
            if self.pending:
                self._keep_publish(window)
                self._add_prefix(effect)
                self._clear_strike(window)
                return
        if window.get("resume_squares") is not None and phase not in {"move", "sentinel"}:
            effect = combine([effect, self._resume_move(window)])
            if self.pending:
                self._keep_publish(window)
                self._add_prefix(effect)
                self._clear_strike(window)
                return
        effect = self._followups(window, effect)
        if self.pending:
            self._keep_publish(window)
            self._add_prefix(effect)
            self._clear_strike(window)
            return
        self._clear_strike(window)
        self._publish_window(window, effect)

    def _mastery_after(self, window: dict, effect: Effect) -> None:
        if not effect.hit or not window.get("option_name"):
            return
        actor = self.by_id(window["actor_id"])
        target = self.by_id(window["target_id"])
        option = actor.attack_named(window["option_name"])
        self._mastery_follow(actor, target, option, effect)

    def _followups(self, window: dict, effect: Effect) -> Effect:
        if window.get("multi_rest"):
            names = list(window.pop("multi_rest"))
            actor = self.by_id(window["multi_actor"])
            target = self.by_id(window["multi_target"])
            more = self._run_strikes(actor, target, names)
            effect = combine([effect, more])
            if self.pending:
                self._keep_emanation(window)
                return effect
        if window.get("beams_left"):
            left = int(window.pop("beams_left"))
            actor = self.by_id(window["beam_actor"])
            target = self.by_id(window["beam_target"])
            parts = window.get("beam_parts") or []
            beams = []
            for beam in range(left):
                beams.append(
                    self._rolled_attack(
                        actor,
                        target,
                        bonus=actor.spell_attack_bonus,
                        damage=parts,
                        melee=bool(window.get("beam_melee")),
                        label=window.get("beam_label") or window["label"],
                        option=None,
                        spell=True,
                    )
                )
                if self.pending:
                    self.pending["beams_left"] = left - beam - 1
                    self.pending["beam_parts"] = parts
                    self.pending["beam_label"] = window.get("beam_label") or window["label"]
                    self.pending["beam_melee"] = window.get("beam_melee")
                    self.pending["beam_actor"] = actor.id
                    self.pending["beam_target"] = target.id
                    for key in ("rest_targets", "rest_actor", "rest_spell", "rest_slot", "rest_label"):
                        if key in window:
                            self.pending[key] = window[key]
                    break
            effect = combine([effect, *beams])
            if self.pending:
                self._keep_emanation(window)
                return effect
        if window.get("rest_targets"):
            actor = self.by_id(window["rest_actor"])
            spell = actor.spell_named(window.get("rest_spell"))
            targets = [self.by_id(creature_id) for creature_id in window.pop("rest_targets")]
            more = self._apply_spell(actor, spell, int(window.get("rest_slot") or 0), targets, window.get("rest_label") or window["label"])
            effect = combine([effect, more])
            if self.pending:
                self._keep_emanation(window)
                return effect
        if window.get("save_rest"):
            source = self.by_id(window["save_source"])
            rest = list(window.pop("save_rest"))
            more = []
            resumed = source.attack_named(str(window.get("label") or ""))
            rider = resumed.rider if resumed is not None else {}
            for index, creature_id in enumerate(rest):
                target = self.by_id(creature_id)
                effect = self._rolled_save(
                    source,
                    target,
                    dc=int(window["dc"]),
                    ability=str(window.get("ability") or "dexterity"),
                    success=str(window.get("success") or "none"),
                    damage=list(window.get("damage") or []),
                    rng=self.rng,
                    label=str(window.get("label") or ""),
                    echo=bool(window.get("echo")),
                )
                more.append(effect)
                if not self.pending:
                    self._mark_prone_from_save(source, target, rider, effect)
                if self.pending:
                    self.pending["save_rest"] = rest[index + 1 :]
                    self.pending["save_source"] = source.id
                    for key in ("dc", "ability", "success", "damage", "echo", "label"):
                        if key in window and key not in self.pending:
                            self.pending[key] = window[key]
                    break
            effect = combine([effect, *more])
        if self.pending:
            self._keep_emanation(window)
            return effect
        if window.get("emanation_rest"):
            rest = list(window.pop("emanation_rest"))
            effect = combine([effect, self._resume_emanation_rest(rest)])
            if self.pending:
                self._keep_emanation(window)
        return effect

    def _run_strikes(self, actor, target, names: list[str]) -> Effect:
        effects = []
        for index, name in enumerate(names):
            option = actor.attack_named(name)
            if option is None or target.hp <= 0 or not weapon_reaches(actor, target, option):
                if option is not None:
                    effects.append(Effect(lines=[f"{name} cannot reach {target.name}."]))
                continue
            effects.append(self._strike(actor, target, option))
            if self.pending:
                self.pending["multi_rest"] = list(names[index + 1:])
                self.pending["multi_actor"] = actor.id
                self.pending["multi_target"] = target.id
                break
        return combine(effects)

    def _remember_targets(self, actor, spell, slot: int, label: str, targets: list) -> None:
        if not self.pending:
            return
        self.pending["rest_targets"] = [target.id for target in targets]
        self.pending["rest_actor"] = actor.id
        self.pending["rest_spell"] = spell.index
        self.pending["rest_slot"] = slot
        self.pending["rest_label"] = label

    def _hold(self, actor, action, source: str, reason: str, effect: Effect) -> None:
        self.pending["publish"] = {
            "actor_id": actor.id,
            "kind": action.kind,
            "label": action.label,
            "source": source,
            "target_id": action.target_id or "",
            "reason": reason,
        }
        self._add_prefix(effect)

    def _add_prefix(self, effect: Effect) -> None:
        if not self.pending or effect is None:
            return
        self.pending["prefix_lines"] = list(effect.lines or []) + list(self.pending.get("prefix_lines") or [])
        self.pending["prefix_damage"] = int(effect.damage) + int(self.pending.get("prefix_damage") or 0)
        self.pending["prefix_healing"] = int(effect.healing) + int(self.pending.get("prefix_healing") or 0)

    def _keep_publish(self, old: dict) -> None:
        pending = self.pending
        if not pending or pending is old:
            return
        if old.get("publish") and "publish" not in pending:
            pending["publish"] = old["publish"]
        pending["prefix_lines"] = list(old.get("prefix_lines") or []) + list(pending.get("prefix_lines") or [])
        pending["prefix_damage"] = int(old.get("prefix_damage") or 0) + int(pending.get("prefix_damage") or 0)
        pending["prefix_healing"] = int(old.get("prefix_healing") or 0) + int(pending.get("prefix_healing") or 0)
        for key in _CARRY:
            if old.get(key) is not None and key not in pending:
                pending[key] = old[key]

    def _publish_window(self, window: dict, effect: Effect) -> None:
        info = window.get("publish")
        if not info:
            return
        lines = list(window.get("prefix_lines") or [])
        lines.extend(effect.lines or [])
        damage = int(window.get("prefix_damage") or 0) + int(effect.damage)
        healing = int(window.get("prefix_healing") or 0) + int(effect.healing)
        if info.get("reason"):
            lines.append(f"Reason: {info['reason']}")
        actor = self.by_id(info["actor_id"])
        for line in lines:
            self.log.append(line)
        actor.traits.pop("on_attack_action", None)
        actor.traits.pop("strip_mod", None)
        self.damage_dealt[actor.side] += damage
        self._record(
            actor,
            info["kind"],
            info["label"],
            info["source"],
            info.get("target_id") or "",
            damage,
            healing,
            " ".join(lines),
        )
        self._check_end()
        if info.get("kind") == "end" and not self.over:
            self._advance()

    def _turn_mark(self) -> list[int]:
        # Traits are JSON-serialized, so the key is a list rather than a tuple.
        return [int(self.round), int(self.turn_index)]

    def _emanation_spell(self, caster: Combatant):
        if caster.hp <= 0 or caster.concentrating_on not in EMANATION_INDEXES:
            return None
        spell = caster.spell_named(caster.concentrating_on)
        if spell is None or not spell.ongoing or not spell.area_type:
            return None
        return spell

    def _emanation_cells(self, spell, anchor) -> set[tuple[int, int]]:
        return area_squares((int(anchor[0]), int(anchor[1])), spell.area_type, int(spell.area_size or 0), None)

    def _saved_ids(self, caster: Combatant) -> list[str]:
        saved = caster.traits.get("emanation_saved")
        if not isinstance(saved, dict) or list(saved.get("turn") or []) != self._turn_mark():
            return []
        return [str(item) for item in saved.get("ids") or []]

    def _already_saved(self, caster: Combatant, creature_id: str) -> bool:
        return str(creature_id) in self._saved_ids(caster)

    def _mark_saved(self, caster: Combatant, creature_ids) -> None:
        ids = self._saved_ids(caster)
        for creature_id in creature_ids:
            text = str(creature_id)
            if text not in ids:
                ids.append(text)
        caster.traits["emanation_saved"] = {"turn": self._turn_mark(), "ids": ids}

    def _note_entered(self, entered: dict, caster_id: str, creature_id: str) -> None:
        bucket = entered.setdefault(caster_id, [])
        if creature_id not in bucket:
            bucket.append(creature_id)

    def _collect_entered(self, mover: Combatant, previous, square, entered: dict) -> None:
        previous = (int(previous[0]), int(previous[1]))
        square = (int(square[0]), int(square[1]))
        for caster in self.combatants:
            spell = self._emanation_spell(caster)
            if spell is None:
                continue
            if mover.id == caster.id:
                old_cells = self._emanation_cells(spell, previous)
                new_cells = self._emanation_cells(spell, square)
                for creature in self.combatants:
                    if creature.id == caster.id or creature.hp <= 0 or creature.side == caster.side:
                        continue
                    body = creature.squares()
                    if (body & new_cells) and not (body & old_cells):
                        self._note_entered(entered, caster.id, creature.id)
                continue
            if mover.hp <= 0 or mover.side == caster.side:
                continue
            cells = self._emanation_cells(spell, tuple(caster.position))
            old_body = footprint(previous, mover.span)
            new_body = footprint(square, mover.span)
            if (new_body & cells) and not (old_body & cells):
                self._note_entered(entered, caster.id, mover.id)

    def _attribute_emanation(self, caster: Combatant, effect: Effect) -> Effect:
        if self.pending:
            return effect
        self.damage_dealt[caster.side] += int(effect.damage)
        return Effect(lines=list(effect.lines or []), healing=int(effect.healing))

    def _detach_move(self) -> None:
        # The walk is already finished. A paused save must not resume it.
        if not self.pending:
            return
        for key in (
            "resume_squares",
            "move_actor",
            "move_budget",
            "move_dest",
            "move_cost",
            "move_origin",
            "resume_phase",
            "opportunity_done",
        ):
            self.pending[key] = None

    def _keep_emanation(self, window: dict) -> None:
        pending = self.pending
        if not pending or pending is window:
            return
        if window.get("emanation_rest") and "emanation_rest" not in pending:
            pending["emanation_rest"] = window["emanation_rest"]

    def _run_emanation_jobs(self, jobs) -> Effect:
        effects = []
        for index, (caster, creature_ids) in enumerate(jobs):
            spell = self._emanation_spell(caster)
            if spell is None:
                continue
            fresh = []
            for creature_id in creature_ids:
                if self._already_saved(caster, creature_id):
                    continue
                try:
                    creature = self.by_id(creature_id)
                except KeyError:
                    continue
                if creature.hp <= 0 or creature.side == caster.side:
                    continue
                fresh.append(creature)
            if not fresh:
                continue
            self._mark_saved(caster, [creature.id for creature in fresh])
            slot = int(caster.ongoing_slot or spell.level or 0)
            applied = self._apply_spell(caster, spell, slot, fresh, f"{spell.name} (emanation)")
            effects.append(self._attribute_emanation(caster, applied))
            if self.pending:
                rest = [
                    {"caster_id": later.id, "ids": list(later_ids)}
                    for later, later_ids in jobs[index + 1 :]
                ]
                if rest:
                    self.pending["emanation_rest"] = rest
                self._detach_move()
                return combine(effects)
        return combine(effects) if effects else Effect()

    def _resume_emanation_rest(self, rest) -> Effect:
        jobs = []
        for job in rest:
            try:
                caster = self.by_id(job.get("caster_id"))
            except KeyError:
                continue
            jobs.append((caster, list(job.get("ids") or [])))
        return self._run_emanation_jobs(jobs)

    def _sweep_emanation(self, mover: Combatant, origin, squares) -> Effect:
        entered: dict[str, list[str]] = {}
        previous = (int(origin[0]), int(origin[1]))
        for square in squares:
            square = (int(square[0]), int(square[1]))
            self._collect_entered(mover, previous, square, entered)
            previous = square
        jobs = [(caster, entered[caster.id]) for caster in self.combatants if caster.id in entered]
        return self._run_emanation_jobs(jobs)

    def _fold_emanation(self, effect: Effect, mover: Combatant, origin, squares) -> None:
        sweep = self._sweep_emanation(mover, origin, squares)
        effect.damage += sweep.damage
        effect.healing += sweep.healing
        effect.lines.extend(sweep.lines or [])

    def _emanation_end(self, actor: Combatant) -> Effect:
        jobs = []
        for caster in self.combatants:
            spell = self._emanation_spell(caster)
            if spell is None or actor.id == caster.id or actor.hp <= 0 or actor.side == caster.side:
                continue
            if not actor.in_cells(self._emanation_cells(spell, tuple(caster.position))):
                continue
            if self._already_saved(caster, actor.id):
                continue
            jobs.append((caster, [actor.id]))
        return self._run_emanation_jobs(jobs)

    def _ongoing(self, actor: Combatant) -> None:
        if not actor.concentrating_on:
            return
        spell = actor.spell_named(actor.concentrating_on)
        if spell is None or not spell.ongoing or spell.index in EMANATION_INDEXES:
            return
        slot = int(actor.ongoing_slot or spell.level or 0)
        probe = GameAction(
            id=-1,
            kind="spell",
            label=spell.name,
            spell_index=spell.index,
            aim=tuple(actor.position),
            slot_level=slot,
        )
        targets = [target for target in targets_for(self, actor, probe) if target.id != actor.id and target.hp > 0]
        effect = self._apply_spell(actor, spell, slot, targets, f"{spell.name} (ongoing)")
        self.damage_dealt[actor.side] += effect.damage
        for line in effect.lines:
            self.log.append(line)
        self._record(actor, "ongoing", spell.name, "engine", None, effect.damage, effect.healing, " ".join(effect.lines))

    def _recharge(self, actor: Combatant) -> None:
        still = []
        for name in list(actor.recharging):
            option = actor.attack_named(name)
            minimum = option.recharge_min if option and option.recharge_min is not None else 6
            roll = self.rng.randint(1, 6)
            if roll >= minimum:
                self.log.append(f"{actor.name} recharges {name} (d6 = {roll}).")
            else:
                still.append(name)
                self.log.append(f"{name} is still recharging (d6 = {roll}, needs {minimum}+).")
        actor.recharging = still

    def _spend_budget(self, actor: Combatant, budget: str) -> None:
        if budget == "bonus":
            actor.bonus_available = False
        else:
            actor.action_available = False

    def _is_swing(self, actor: Combatant, action: GameAction) -> bool:
        if action.kind != "attack" or action.budget != "action":
            return False
        if action.attack_name == "Multiattack" or action.facing:
            return False
        option = actor.attack_named(action.attack_name or "")
        if option is None or option.area_type:
            return False
        rider = option.rider or {}
        if rider.get("grapple") or rider.get("shove"):
            return True
        return option.resolution == "attack"

    def _is_nick_swing(self, actor: Combatant, option) -> bool:
        if option is None:
            return False
        per = max(1, int(actor.attacks_per_action or 1))
        return actor.attacks_made >= per and option.name == nick_weapon_name(actor)

    def _open_banked(self, actor: Combatant) -> None:
        if actor.action_available or not actor.traits.get("banked_action"):
            return
        actor.traits.pop("banked_action", None)
        actor.action_available = True
        actor.attacks_made = 0
        actor.traits.pop("light_used", None)
        actor.traits.pop("nick_used", None)
        actor.traits["surge_blocks_magic"] = True

    def _commit(self, actor: Combatant, action: GameAction) -> None:
        if action.kind == "feature":
            return
        if action.budget == "bonus":
            if action.tag == "flurry":
                left = int(actor.traits.get("flurry_left") or 0) - 1
                if left > 0:
                    actor.traits["flurry_left"] = left
                else:
                    actor.traits.pop("flurry_left", None)
                return
            actor.bonus_available = False
            if action.tag == "hew":
                actor.traits.pop("hew_ready", None)
            if action.tag == "pole":
                actor.traits.pop("pole_ready", None)
            if action.tag == "dual":
                actor.traits["nick_used"] = True
            return
        if action.budget != "action":
            return
        if action.kind == "attack" and self._is_swing(actor, action):
            option = actor.attack_named(action.attack_name or "")
            if option is not None and actor.traits.get("polearm_master") and pole_weapon(option):
                actor.traits["pole_ready"] = option.name
            if self._is_nick_swing(actor, option):
                actor.traits["nick_used"] = True
                actor.action_available = False
                self._open_banked(actor)
                return
            actor.attacks_made += 1
            if option is not None and option.light:
                actor.traits["light_used"] = option.name
            per = max(1, int(actor.attacks_per_action or 1))
            if actor.attacks_made >= per and not nick_weapon_name(actor):
                actor.action_available = False
            self._open_banked(actor)
            return
        actor.action_available = False
        self._open_banked(actor)

    def _consume(self, actor: Combatant, option) -> None:
        if option.recharge_min is not None and option.name not in actor.recharging:
            actor.recharging.append(option.name)
        if option.uses is not None:
            actor.uses_left[option.name] = actor.uses_left.get(option.name, option.uses) - 1

    def _record(self, actor, kind, label, source, target_id, damage, healing, detail) -> None:
        self.turns.append(
            {
                "round": self.round,
                "actor_id": actor.id,
                "actor_name": actor.name,
                "side": actor.side,
                "action_kind": kind,
                "action_label": label,
                "source": source,
                "target_id": target_id or "",
                "damage": damage,
                "healing": healing,
                "detail": detail,
            }
        )

    def _note_steps(self, actor, squares) -> None:
        path = [tuple(step) for step in (actor.traits.get("move_path") or [])]
        for square in squares:
            path.append((int(square[0]), int(square[1])))
        actor.traits["move_path"] = path

    def _pack_flag(self, actor, target) -> None:
        actor.traits.pop("pack_now", None)
        if not actor.traits.get("pack_tactics"):
            return
        for ally in self.combatants:
            if ally.id == actor.id or ally.side != actor.side or ally.hp <= 0 or is_incapacitated(ally):
                continue
            if ally.gap_to(target) <= 1:
                actor.traits["pack_now"] = True
                return

    def _tick_lingers(self) -> None:
        for creature in self.combatants:
            if "lingers" not in creature.traits:
                continue
            kept = []
            for placed in creature.traits.get("lingers") or []:
                if not isinstance(placed, dict):
                    continue
                placed["rounds"] = int(placed.get("rounds") or 0) - 1
                if placed["rounds"] > 0:
                    kept.append(placed)
            creature.traits["lingers"] = kept

    def _reap(self, moment: str = "") -> list[str]:
        homes = {
            creature.id: creature.traits.get("banish_home")
            for creature in self.combatants
            if creature.traits.get("banished")
        }
        lines = [line for line in reap(self, moment=moment) if line]
        for creature in self.combatants:
            if creature.traits.get("banished") or creature.traits.get("removed") or creature.traits.get("banish_away"):
                continue
            home = homes.get(creature.id)
            if not home:
                continue
            self._restore_home(creature, home)
            creature.traits.pop("banish_home", None)
        for line in lines:
            self.log.append(line)
        return lines

    def _restore_home(self, creature, home) -> None:
        home = (int(home[0]), int(home[1]))
        occupied: set[tuple[int, int]] = set()
        for other in self.combatants:
            if other.id == creature.id or other.hp <= 0 or other.traits.get("banished") or other.traits.get("removed"):
                continue
            occupied.update(other.squares())
        if not (footprint(home, creature.span) & occupied) and all(in_bounds(cell, self.width, self.height) for cell in footprint(home, creature.span)):
            creature.position = home
            return
        for radius in range(1, max(self.width, self.height)):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if max(abs(dx), abs(dy)) != radius:
                        continue
                    square = (home[0] + dx, home[1] + dy)
                    body = footprint(square, creature.span)
                    if any(not in_bounds(cell, self.width, self.height) or cell in occupied for cell in body):
                        continue
                    creature.position = square
                    return

    def _mark_save(self, source, target, ability: str, dc: int, label: str, *, advantage: bool = False) -> tuple[bool, list[str]]:
        mode = save_mode(target, ability)
        if mode is None:
            return True, [f"Failed save  {target.name} automatically fails the {ability.title()} save."]
        if advantage:
            mode = min(1, mode + 1)
        roll, shown = d20(self.rng, mode)
        full = ability.lower()
        if full in {"str", "dex", "con", "int", "wis", "cha"}:
            full = {"str": "strength", "dex": "dexterity", "con": "constitution", "int": "intelligence", "wis": "wisdom", "cha": "charisma"}[full]
        bonus = target.save_bonuses.get(full, ability_mod(target.abilities.get(full, 10)))
        penalty = save_penalty(target)
        total = roll + bonus - penalty
        failed = total < dc
        lead = "Failed save" if failed else "Saved"
        if penalty:
            math = f"{roll} + {bonus} - {penalty} = {total}"
        elif len(shown) > 1:
            math = f"{' and '.join(str(item) for item in shown)}, keeps {roll}, +{bonus} = {total}"
        else:
            math = f"{roll} + {bonus} = {total}"
        return failed, [f"{lead}  {target.name} vs DC {dc} {full.title()} ({label}). {math}."]

    def _linked(self, target, spell: str, source: str) -> list[str]:
        return drop_marks(
            target,
            lambda mark, spell=spell, source=source: mark.get("spell") == spell and mark.get("source") == source,
        )

    def _stamp(self, actor, target, spec: dict, condition: str, **extra) -> str:
        until = str(extra.pop("until", None) or spec.get("until") or "fight")
        if spec.get("concentration") and until == "fight":
            until = "concentration"
        tied = "concentration" if until == "concentration" or (spec.get("concentration") and until != "rounds") else ""
        if until in {"end", "start", "source-next-end", "stand"}:
            tied = ""
        return add_mark(
            target,
            actor,
            condition=condition,
            until=until,
            rounds=int(spec.get("rounds") or 0) if until == "rounds" else int(extra.pop("rounds", 0) or 0),
            save=str(extra.pop("save", None) or spec.get("save") or ""),
            dc=int(extra.pop("dc", None) or actor.spell_save_dc or 0),
            repeat=bool(extra.pop("repeat", spec.get("repeat"))),
            spell=str(extra.pop("spell", None) or ""),
            tied=tied,
            **extra,
        )

    def _flush_saves(self) -> list[str]:
        lines = []
        pending = []
        for creature in self.combatants:
            for mark in marks_of(creature):
                if mark.get("pending_save"):
                    pending.append((creature.id, mark.get("id")))
        for creature_id, mark_id in pending:
            creature = self.by_id(creature_id)
            mark = mark_by_id(creature, mark_id)
            if mark is None:
                continue
            mark["pending_save"] = False
            advantage = bool(mark.get("save_advantage")) or (
                mark.get("condition") == "poisoned" and creature.traits.get("poison_ward")
            )
            mark["save_advantage"] = False
            failed, rolled = self._mark_save(
                self._mark_source(mark, creature),
                creature,
                str(mark.get("save") or "wisdom"),
                int(mark.get("dc") or 0),
                str(mark.get("spell") or "the spell"),
                advantage=advantage,
            )
            lines.extend(rolled)
            if not failed:
                lines.extend(self._linked(creature, str(mark.get("spell") or ""), str(mark.get("source") or "")))
        return lines

    def _mark_source(self, mark, fallback):
        try:
            return self.by_id(str(mark.get("source") or ""))
        except KeyError:
            return fallback

    def _repeat_marks(self, actor) -> Effect:
        lines = []
        rows = list(marks_of(actor))
        for mark in rows:
            current = mark_by_id(actor, mark.get("id"))
            if current is None:
                continue
            if current.get("sleep_stage") == 1 or (current.get("second") and not current.get("repeat")):
                lines.extend(self._second_save(actor, current))
                continue
            if not current.get("repeat") or not current.get("save"):
                continue
            failed, rolled = self._mark_save(
                self._mark_source(current, actor),
                actor,
                str(current.get("save") or "wisdom"),
                int(current.get("dc") or 0),
                str(current.get("spell") or "the spell"),
                advantage=bool(current.get("condition") == "poisoned" and actor.traits.get("poison_ward")),
            )
            lines.extend(rolled)
            if current.get("weird") and failed:
                pulse = resolve_auto(self._mark_source(current, actor), actor, damage=[("psychic", "5d10")], rng=self.rng, label="Weird")
                lines.extend(pulse.lines or [])
            if current.get("track") == "thirds" or current.get("spell") == "flesh-to-stone":
                if failed:
                    current["failures"] = int(current.get("failures") or 0) + 1
                    if current["failures"] >= 3:
                        lines.extend(self._linked(actor, str(current.get("spell") or ""), str(current.get("source") or "")))
                        lines.append(self._stamp(self._mark_source(current, actor), actor, {"concentration": True, "until": "concentration"}, "petrified", spell=str(current.get("spell") or "")))
                else:
                    current["successes"] = int(current.get("successes") or 0) + 1
                    if current["successes"] >= 3:
                        lines.extend(self._linked(actor, str(current.get("spell") or ""), str(current.get("source") or "")))
                continue
            if not failed:
                lines.extend(self._linked(actor, str(current.get("spell") or ""), str(current.get("source") or "")))
        return Effect(lines=[line for line in lines if line])

    def _second_save(self, actor, mark) -> list[str]:
        failed, lines = self._mark_save(
            self._mark_source(mark, actor),
            actor,
            str(mark.get("save") or "wisdom"),
            int(mark.get("dc") or 0),
            str(mark.get("spell") or "Sleep"),
        )
        spell = str(mark.get("spell") or "")
        source = str(mark.get("source") or "")
        if failed and mark.get("sleep_stage") == 1:
            lines.extend(self._linked(actor, spell, source))
            owner = self._mark_source(mark, actor)
            lines.append(add_mark(
                actor, owner, condition="unconscious", until="concentration", tied="concentration",
                spell=spell, ends_on_damage=True, shake=True, save=str(mark.get("save") or ""), dc=int(mark.get("dc") or 0),
            ))
            return lines
        if failed and mark.get("second"):
            lines.extend(self._linked(actor, spell, source))
            lines.append(add_mark(actor, self._mark_source(mark, actor), condition=str(mark.get("second")), until="fight", spell=spell))
            return lines
        lines.extend(self._linked(actor, spell, source))
        return lines

    def _expire_marks(self, actor) -> Effect:
        lines = []
        for creature in self.combatants:
            kept = []
            changed = False
            for mark in marks_of(creature):
                drop = False
                if creature.id == actor.id and mark.get("until") == "end":
                    drop = True
                if mark.get("source") == actor.id and mark.get("until") == "source-next-end":
                    if mark.get("seen"):
                        drop = True
                    else:
                        mark["seen"] = True
                if creature.id == actor.id and mark.get("until") == "rounds" and int(mark.get("rounds") or 0) > 0:
                    mark["rounds"] = int(mark["rounds"]) - 1
                    if mark["rounds"] <= 0:
                        drop = True
                if mark.get("banish") and mark.get("source") == actor.id and int(mark.get("rounds") or 0) > 0:
                    mark["rounds"] = int(mark["rounds"]) - 1
                    if mark["rounds"] <= 0:
                        kinds = creature_types(creature)
                        if kinds & OUTSIDERS:
                            creature.traits["removed"] = True
                            creature.traits["banish_away"] = True
                            creature.hp = 0
                        drop = True
                if drop:
                    changed = True
                    if mark.get("condition"):
                        lines.append(f"{creature.name} is no longer {mark.get('condition')}.")
                    continue
                kept.append(mark)
            if changed:
                from src.conditions import _store
                _store(creature, kept)
                sync(creature)
        return Effect(lines=lines)

    def _arm_marks(self, actor) -> None:
        for mark in marks_of(actor):
            if mark.get("repeat_order"):
                mark["order_used"] = False
        for mark in list(marks_of(actor)):
            if mark.get("spell") == "ensnaring-strike" and "restrained" in actor.conditions:
                owner = self._mark_source(mark, actor)
                effect = resolve_auto(owner, actor, damage=[("piercing", "1d6")], rng=self.rng, label="Ensnaring Strike")
                for line in effect.lines or []:
                    self.log.append(line)
                self._record(actor, "ongoing", "Ensnaring Strike", "engine", actor.id, effect.damage, 0, " ".join(effect.lines or []))

    def _auras(self, actor) -> None:
        for other in self.combatants:
            if other.hp <= 0 or other.id == actor.id:
                continue
            for aura in other.traits.get("auras") or []:
                if actor.gap_to(other) * 5 > int(aura.get("feet") or 0):
                    continue
                condition = str(aura.get("condition") or "")
                if condition and immune_to(actor, condition, other):
                    self.log.append(f"{actor.name} is immune to {condition}.")
                    continue
                failed, lines = self._mark_save(other, actor, str(aura.get("save") or "constitution"), int(aura.get("dc") or 10), "aura")
                for line in lines:
                    self.log.append(line)
                if failed and condition:
                    note = add_mark(actor, other, condition=condition, until=str(aura.get("until") or "end"), spell="aura")
                    if note:
                        self.log.append(note)

    def _heroism_pulse(self, actor) -> None:
        for mark in marks_of(actor):
            if not mark.get("heroism"):
                continue
            amount = int(mark.get("temp") or 0)
            if amount > actor.temp_hp:
                actor.temp_hp = amount
                self.log.append(f"{actor.name} gains {amount} temporary hit points.")

    def refresh_positional(self) -> None:
        for creature in self.combatants:
            creature.traits.pop("inside_silence", None)
            creature.traits.pop("inside_blind", None)
            creature.traits.pop("inside_deaf", None)
            creature.traits.pop("thunder_immune", None)
        for _caster, spec, placed in _each_zone(self):
            cells = cells_of(spec, placed)
            for creature in self.combatants:
                if creature.hp <= 0:
                    continue
                body = creature.squares()
                if not body:
                    continue
                inside = body <= cells
                overlap = bool(body & cells)
                if spec.silence and inside:
                    creature.traits["inside_silence"] = True
                    creature.traits["thunder_immune"] = True
                    creature.traits["inside_deaf"] = True
                elif spec.deaf_inside and inside:
                    creature.traits["inside_deaf"] = True
                if spec.blind_inside and overlap:
                    creature.traits["inside_blind"] = True

    def _caster_pulse(self, actor) -> Effect:
        found = read_zone(actor)
        if found is None:
            return Effect()
        spec, placed = found
        if not spec.allows("caster-end"):
            return Effect()
        effects = []
        for creature in list(self.combatants):
            if creature.hp <= 0 or already_saved(placed, self, creature.id):
                continue
            if not self._zone_covers(actor, spec, placed, creature, "end"):
                continue
            effects.append(self._pay_zone(actor, self._zone_trigger(actor, creature, spec, placed, "end")))
        return combine(effects) if effects else Effect()

    def _stand(self, actor) -> Effect:
        cost = stand_cost(actor)
        if actor.movement_left is None:
            actor.movement_left = turn_speed(actor)
        if cost <= 0 or cost > int(actor.movement_left or 0):
            raise IllegalAction("Stand")
        actor.movement_left -= cost
        lines = stand(actor)
        return Effect(lines=[f"{actor.name} stands ({cost} ft). {actor.movement_left} ft left.", *lines])

    def _order(self, actor, action) -> Effect:
        mark = pending_order(actor)
        name = str(action.attack_name or (mark or {}).get("order") or "")
        if mark is not None:
            if mark.get("repeat_order"):
                mark["order_used"] = True
            else:
                drop_marks(actor, lambda item, mark_id=mark.get("id"): item.get("id") == mark_id)
        lines = [f"{actor.name} follows the command {name}."]
        if name == "grovel":
            lines.append(add_mark(actor, actor, condition="prone", until="stand"))
        elif name == "drop":
            lines.append(f"{actor.name} drops what it is holding.")
        elif name == "halt":
            lines.append(f"{actor.name} does not move.")
        if name == "away":
            actor.movement_left = int(actor.movement_left or 0) + turn_speed(actor)
        if action.destination is not None:
            moved = self._relocate(actor, action)
            lines.extend(moved.lines or [])
        actor.action_available = False
        actor.bonus_available = False
        return Effect(lines=[line for line in lines if line])

    def _escape(self, actor, action) -> Effect:
        actor.action_available = False
        mark = None
        for item in marks_of(actor):
            if item.get("grapple") or item.get("spell") == "ensnaring-strike":
                mark = item
                break
        if mark is None:
            return Effect(lines=[f"{actor.name} has nothing to escape."])
        ability = "strength" if action.attack_name == "athletics" else "dexterity"
        bonus = ability_mod(actor.abilities.get(ability, 10))
        mode = -1 if "poisoned" in actor.conditions else 0
        roll, _shown = d20(self.rng, mode)
        total = roll + bonus
        if mark.get("grapple"):
            dc = int(mark.get("escape_dc") or 10)
        else:
            try:
                dc = int(self.by_id(str(mark.get("source") or "")).spell_save_dc or 10)
            except KeyError:
                dc = 10
        if total >= dc:
            if mark.get("grapple"):
                lines = drop_marks(actor, lambda item: item.get("grapple") or item.get("tied") == "grapple")
            else:
                lines = self._linked(actor, "ensnaring-strike", str(mark.get("source") or ""))
            return Effect(lines=[f"{actor.name} escapes ({roll}+{bonus}={total} vs DC {dc}).", *lines])
        return Effect(lines=[f"{actor.name} fails to escape ({roll}+{bonus}={total} vs DC {dc})."])

    def _shake(self, actor, action) -> Effect:
        actor.action_available = False
        try:
            target = self.by_id(action.target_id or "")
        except KeyError:
            return Effect(lines=[f"{actor.name} has nobody to shake."])
        lines = drop_marks(target, lambda mark: mark.get("shake"))
        if not lines:
            lines = [f"{target.name} is not in a stupor."]
        return Effect(lines=[f"{actor.name} shakes {target.name}.", *lines])

    def _mark_prone_from_save(self, source, target, rider: dict, effect: Effect) -> None:
        """A failed area save knocks a creature prone when the stat block names a size."""
        if not rider or rider.get("charge_feet") or not self._failed(effect):
            return
        prone_size = rider.get("prone_size")
        if not prone_size or size_rank(getattr(target, "size", None)) > size_rank(str(prone_size)):
            return
        note = add_mark(target, source, condition="prone", until="stand")
        if note:
            if effect.lines is None:
                effect.lines = []
            effect.lines.append(note)

    def _save_rider(self, actor, target, rider: dict, effect: Effect) -> None:
        if not rider or rider_skipped(target, rider):
            return
        if effect.lines is None:
            effect.lines = []
        condition = str(rider.get("save_condition") or "")
        first = str(rider.get("first") or "")
        if not self._failed(effect):
            return
        self._mark_prone_from_save(actor, target, rider, effect)
        if condition:
            until = "start" if rider.get("save_until") == "start" else "end"
            note = add_mark(target, actor, condition=condition, until=until, spell=str(rider.get("name") or "rider"))
            if note:
                effect.lines.append(note)
            return
        if first:
            note = add_mark(
                target,
                actor,
                condition=first,
                until="end",
                save=str(rider.get("save_ability") or ""),
                dc=int(rider.get("save_dc") or 0),
                second=str(rider.get("second") or ""),
                spell="breath",
            )
            if note:
                effect.lines.append(note)

    def _hit_riders(self, actor, target, option, effect: Effect) -> None:
        rider = getattr(option, "rider", None) or {}
        if not rider:
            return
        if effect.lines is None:
            effect.lines = []
        if rider_skipped(target, rider):
            effect.lines.append(f"{option.name} does not affect {target.name} that way.")
            return
        toward = False
        if rider.get("charge_feet"):
            toward = charged_toward(actor, target, int(rider["charge_feet"]))
        effect.lines.extend(hit_rider_marks(actor, target, rider, toward=toward))
        condition = str(rider.get("save_condition") or "")
        if not condition and not rider.get("first"):
            return
        if condition and immune_to(target, condition, actor):
            effect.lines.append(f"{target.name} is immune to {condition}.")
            return
        if rider.get("first") and immune_to(target, str(rider.get("first")), actor):
            effect.lines.append(f"{target.name} is immune to {rider.get('first')}.")
            return
        save = self._rolled_save(
            actor,
            target,
            dc=int(rider.get("save_dc") or 10),
            ability=str(rider.get("save_ability") or "constitution"),
            success="none",
            damage=[],
            rng=self.rng,
            label=option.name,
        )
        effect.lines.extend(save.lines or [])
        if not self.pending:
            self._save_rider(actor, target, rider, effect)

    def _spell_rider(self, actor, spell, target, effect: Effect) -> None:
        found = affliction(spell.index)
        if not found or found.get("mode") != "rider":
            return
        if effect.lines is None:
            effect.lines = []
        if found.get("on") == "hit" and not effect.hit:
            return
        if found.get("on") == "fail" and not self._failed(effect):
            return
        if found.get("special") == "prismatic":
            return
        if found.get("clear_invisible"):
            effect.lines.extend(clear_condition(target, "invisible"))
            until = str(found.get("until") or "source-next-end")
            note = add_mark(target, actor, condition="", until=until, no_invisible=True, spell=spell.index, tied="concentration" if until == "concentration" else "")
            if until == "concentration":
                note = add_mark(target, actor, condition="outlined", until="concentration", tied="concentration", spell=spell.index)
            if note:
                effect.lines.append(note)
            return
        condition = str(found.get("condition") or "")
        if not condition:
            return
        if immune_to(target, condition, actor):
            effect.lines.append(f"{target.name} is immune to {condition}.")
            return
        until = str(found.get("until") or "fight")
        if found.get("concentration") and until == "fight":
            until = "concentration"
        note = add_mark(
            target,
            actor,
            condition=condition,
            until=until,
            rounds=int(found.get("rounds") or 0) if until == "rounds" else 0,
            repeat=bool(found.get("repeat")),
            save=str(found.get("save") or ""),
            dc=int(actor.spell_save_dc or 0),
            spell=spell.index,
            tied="concentration" if until == "concentration" else "",
            weird=bool(found.get("weird")),
        )
        if note:
            effect.lines.append(note)

    def _clear_listed(self, target, spell) -> list[str]:
        found = affliction(spell.index) or {}
        lines = []
        for condition in found.get("clears") or []:
            lines.extend(clear_condition(target, condition))
        return lines

    def _cast_affliction(self, actor, spell, slot: int, targets, action) -> Effect:
        spec = affliction(spell.index) or {}
        break_on_attack(actor)
        free_name = self._spend_spell_slot(actor, spell, slot)
        self._begin_concentration(actor, spell, slot)
        chosen = self._afflict_targets(actor, spell, spec, action, targets, slot)
        effects = []
        if free_name:
            effects.append(Effect(lines=[f"{actor.name} casts {spell.name} without a spell slot."]))
        if spec.get("special") == "mass-heal":
            effects.append(self._mass_heal(actor, spell))
            return combine(effects)
        if not chosen:
            effects.append(Effect(lines=[f"{actor.name} casts {spell.name}, but it affects no one."]))
            return combine(effects)
        for target in chosen:
            effects.append(self._afflict_one(actor, spell, spec, action, target, slot))
            if self.pending:
                break
        return combine(effects)

    def _afflict_targets(self, actor, spell, spec: dict, action, targets, slot: int):
        pool = [creature for creature in _afflict_pool(self, actor, spell, spec) if creature.hp > 0]
        if spec.get("self_cast"):
            return [actor]
        allowed = {creature.id for creature in pool}
        if spec.get("each"):
            return pool
        if spec.get("area") == "cube" and action.aim is not None:
            cells = centered_block(tuple(action.aim), int(spec.get("area_feet") or 0))
            return [creature for creature in pool if creature.in_cells(cells)]
        if spec.get("area") in {"cone", "line", "sphere"} and targets:
            return [creature for creature in targets if creature.id in allowed and creature.hp > 0]
        chosen = []
        primary = None
        if action.target_id:
            try:
                primary = self.by_id(action.target_id)
            except KeyError:
                primary = None
        if primary is not None and primary.id in allowed:
            chosen.append(primary)
        elif targets:
            chosen.extend(creature for creature in targets if creature.id in allowed)
        count = spell_targets_extra(slot, spell.level) if spec.get("extra") else max(1, len(chosen))
        if not spec.get("extra"):
            return chosen[:1]
        rest = sorted((creature for creature in pool if creature.id not in {item.id for item in chosen}), key=lambda creature: (actor.gap_to(creature), creature.id))
        for creature in rest:
            if len(chosen) >= count:
                break
            chosen.append(creature)
        return chosen[:count]

    def _afflict_one(self, actor, spell, spec: dict, action, target, slot: int) -> Effect:
        special = spec.get("special")
        if special == "sleep" and (is_elf(target) or immune_to(target, "exhaustion")):
            return Effect(lines=[f"{target.name} is unaffected by {spell.name}."])
        if special == "flesh" and "construct" in creature_types(target):
            return Effect(lines=[f"{target.name} is unaffected by {spell.name}."])
        if special:
            return self._afflict_special(actor, spell, spec, action, target, slot)
        condition = str(spec.get("condition") or "")
        choices = set(spec.get("choices") or [])
        if action.attack_name and action.attack_name in choices:
            condition = action.attack_name
        if not condition:
            return Effect()
        if immune_to(target, condition, actor):
            return Effect(lines=[f"{target.name} is immune to {condition}."])
        lines = []
        if spec.get("save"):
            advantage = bool(spec.get("fight_advantage")) or (condition == "poisoned" and target.traits.get("poison_ward"))
            failed, rolled = self._mark_save(actor, target, str(spec.get("save")), int(actor.spell_save_dc or 0), spell.name, advantage=advantage)
            lines.extend(rolled)
            if not failed:
                return Effect(lines=lines)
        note = self._stamp(actor, target, spec, condition, spell=spell.index, ends_on_attack=bool(spec.get("ends_on_attack")), charm_damage=bool(spec.get("charm_damage")))
        if note:
            lines.append(note)
        return Effect(lines=lines)

    def _afflict_special(self, actor, spell, spec: dict, action, target, slot: int) -> Effect:
        name = spec.get("special")
        dc = int(actor.spell_save_dc or 0)
        if name == "see":
            note = add_mark(target, actor, condition="", until="fight", see=True, spell=spell.index)
            sync(target)
            return Effect(lines=[note or f"{target.name} can see invisible creatures."])
        if name == "heroism":
            temp = max(0, int(actor.spellcasting_mod or 0))
            note = self._stamp(actor, target, spec, "", spell=spell.index, heroism=True, temp=temp)
            return Effect(lines=[note or f"{target.name} is brave."])
        if name == "protection":
            note = self._stamp(actor, target, spec, "", spell=spell.index, ward=True)
            sync(target)
            return Effect(lines=[note or f"{target.name} is warded."])
        if name == "freedom":
            note = add_mark(target, actor, condition="", until="fight", freedom=True, spell=spell.index)
            lines = [note or f"{target.name} moves freely."]
            for condition in ("grappled", "paralyzed", "restrained"):
                lines.extend(clear_condition(target, condition))
            sync(target)
            return Effect(lines=lines)
        if name == "poison-ward":
            lines = list(clear_condition(target, "poisoned"))
            note = add_mark(target, actor, condition="", until="fight", poison_ward=True, spell=spell.index)
            sync(target)
            lines.append(note or f"{target.name} is warded against poison.")
            return Effect(lines=lines)
        if name == "lesser":
            choice = action.attack_name or ""
            return Effect(lines=clear_condition(target, choice) or [f"{target.name} is not {choice}."])
        if name == "greater":
            choice = action.attack_name or ""
            if choice == "exhaustion":
                level = int(target.conditions.get("exhaustion") or 0)
                clear_condition(target, "exhaustion")
                if level > 1:
                    add_mark(target, actor, condition="exhaustion", level=level - 1, until="fight")
                return Effect(lines=[f"{target.name} loses one exhaustion level."])
            return Effect(lines=clear_condition(target, choice) or [f"{target.name} is not {choice}."])
        if name == "power-heal":
            healed = target.max_hp - target.hp
            effect = apply_heal(target, healed) if healed else Effect()
            for condition in ("charmed", "frightened", "paralyzed", "poisoned", "stunned"):
                effect.lines.extend(clear_condition(target, condition))
            if "prone" in target.conditions and target.reaction_available:
                effect.lines.extend(stand(target))
                target.reaction_available = False
            return effect
        if name == "mass-heal":
            return self._mass_heal(actor, spell)
        if name == "smite":
            if actor.traits.get("last_hit") != target.id:
                return Effect(lines=[f"{spell.name} needs a hit this turn."])
            effect = resolve_auto(actor, target, damage=spell_parts(actor, spell, slot), rng=self.rng, label=spell.name)
            effect.lines.extend(clear_condition(target, "invisible"))
            effect.lines.append(self._stamp(actor, target, spec, "outlined", spell=spell.index))
            return effect
        if name == "ensnare":
            if actor.traits.get("last_hit") != target.id:
                return Effect(lines=[f"{spell.name} needs a hit this turn."])
            advantage = size_rank(target.size) >= size_rank("large")
            failed, lines = self._mark_save(actor, target, "strength", dc, spell.name, advantage=advantage)
            if not failed:
                lines.append("The vines shrivel away.")
                return Effect(lines=lines)
            lines.append(self._stamp(actor, target, spec, "restrained", spell=spell.index))
            return Effect(lines=lines)
        if name == "prismatic":
            return combine(self._prismatic_rays(actor, target, 0))
        if name in {"laughter", "sleep", "hypnotic", "banish", "power-stun", "flesh", "dance", "eyebite", "command", "divine", "fear", "calm"}:
            return self._named_special(actor, spell, spec, action, target, name, dc)
        return Effect(lines=[f"{spell.name} cannot be resolved."])

    def _named_special(self, actor, spell, spec, action, target, name: str, dc: int) -> Effect:
        if name == "power-stun":
            if target.hp <= 150:
                if immune_to(target, "stunned", actor):
                    return Effect(lines=[f"{target.name} is immune to stunned."])
                note = add_mark(target, actor, condition="stunned", until="fight", repeat=True, save="constitution", dc=dc, spell=spell.index)
                return Effect(lines=[note])
            note = add_mark(target, actor, condition="", until="start", speed_zero=True, spell=spell.index)
            sync(target)
            return Effect(lines=[note or f"{target.name} cannot move until the start of {actor.name}'s next turn."])
        if name == "divine":
            return self._divine_word(actor, spell, target, dc)
        if name == "command":
            word = action.attack_name or "halt"
            failed, lines = self._mark_save(actor, target, "wisdom", dc, spell.name)
            if not failed:
                return Effect(lines=lines)
            add_mark(target, actor, condition="", order=word, spell=spell.index, until="end")
            lines.append(f"{target.name} must {word} on its next turn.")
            return Effect(lines=lines)
        if name == "mass-heal":
            return self._mass_heal(actor, spell)
        advantage = name in {"calm"} and False
        ability = str(spec.get("save") or "wisdom")
        if name == "banish":
            ability = "charisma"
        failed, lines = self._mark_save(actor, target, ability, dc, spell.name, advantage=advantage)
        if not failed and name != "dance":
            if name == "flesh":
                add_mark(target, actor, condition="", until="start", speed_zero=True, spell=spell.index)
                sync(target)
                lines.append(f"{target.name} cannot move until the start of {actor.name}'s next turn.")
            return Effect(lines=lines)
        if name == "laughter":
            lines.append(self._stamp(actor, target, spec, "prone", spell=spell.index, no_stand=True))
            lines.append(add_mark(target, actor, condition="incapacitated", until="concentration", tied="concentration", spell=spell.index, damage_save=True, repeat=True, save="wisdom", dc=dc))
        elif name == "sleep":
            lines.append(add_mark(target, actor, condition="incapacitated", until="end", tied="concentration", spell=spell.index, sleep_stage=1, save="wisdom", dc=dc, ends_on_damage=True, shake=True))
        elif name == "hypnotic":
            lines.append(self._stamp(actor, target, spec, "charmed", spell=spell.index, ends_on_damage=True, shake=True))
            lines.append(add_mark(target, actor, condition="incapacitated", until="concentration", tied="concentration", spell=spell.index, speed_zero=True, ends_on_damage=True, shake=True))
        elif name == "fear":
            lines.append(self._stamp(actor, target, spec, "frightened", spell=spell.index))
            add_mark(target, actor, condition="", order="away", repeat_order=True, until="concentration", tied="concentration", spell=spell.index)
            lines.append(f"{target.name} must dash away from {actor.name}.")
        elif name == "banish":
            target.traits["banished"] = True
            target.traits["banish_home"] = [int(target.position[0]), int(target.position[1])]
            lines.append(add_mark(target, actor, condition="incapacitated", until="concentration", tied="concentration", spell=spell.index, banish=True, rounds=10))
            lines.append(f"{target.name} vanishes.")
        elif name == "flesh":
            lines.append(add_mark(target, actor, condition="restrained", until="concentration", tied="concentration", spell=spell.index, repeat=True, save="constitution", dc=dc, track="thirds"))
        elif name == "dance":
            if failed:
                lines.append(add_mark(target, actor, condition="charmed", until="concentration", tied="concentration", spell=spell.index, dance=True))
            else:
                lines.append(add_mark(target, actor, condition="", until="end", dance=True, spell=spell.index))
                sync(target)
                lines.append(f"{target.name} dances in place until the end of its next turn.")
        elif name == "eyebite":
            choice = action.attack_name or "asleep"
            if choice == "asleep":
                lines.append(add_mark(target, actor, condition="unconscious", until="concentration", tied="concentration", spell=spell.index, ends_on_damage=True, shake=True))
            elif choice == "panicked":
                lines.append(self._stamp(actor, target, spec, "frightened", spell=spell.index))
                add_mark(target, actor, condition="", order="away", repeat_order=True, until="concentration", tied="concentration", spell=spell.index)
            else:
                lines.append(self._stamp(actor, target, spec, "poisoned", spell=spell.index))
        elif name == "calm":
            lines.append(add_mark(target, actor, condition="", until="concentration", tied="concentration", spell=spell.index, calm=True))
            sync(target)
            lines.append(f"{target.name} is calmed.")
        return Effect(lines=[line for line in lines if line])

    def _divine_word(self, actor, spell, target, dc: int) -> Effect:
        failed, lines = self._mark_save(actor, target, "charisma", dc, spell.name)
        if not failed:
            return Effect(lines=lines)
        before = target.hp
        kinds = creature_types(target)
        if kinds & BANISH_TYPES:
            target.traits["removed"] = True
            target.hp = 0
            lines.append(f"{target.name} is banished from this plane.")
        if before <= 20:
            target.hp = 0
            lines.append(f"{target.name} dies.")
        elif before <= 30:
            for condition in ("blinded", "deafened", "stunned"):
                lines.append(add_mark(target, actor, condition=condition, until="fight", spell=spell.index))
        elif 41 <= before <= 50:
            lines.append(add_mark(target, actor, condition="deafened", until="rounds", rounds=10, spell=spell.index))
        sync(target)
        return Effect(lines=[line for line in lines if line])

    def _mass_heal(self, actor, spell) -> Effect:
        remaining = 700
        lines = []
        healed = 0
        allies = sorted(
            (creature for creature in self.combatants if creature.side == actor.side and creature.hp > 0 and creature.hp < creature.max_hp),
            key=lambda creature: creature.id,
        )
        for ally in allies:
            if remaining <= 0:
                break
            amount = min(remaining, ally.max_hp - ally.hp)
            effect = apply_heal(ally, amount)
            remaining -= amount
            healed += effect.healing
            lines.extend(effect.lines or [])
            for condition in ("blinded", "deafened", "poisoned"):
                lines.extend(clear_condition(ally, condition))
        lines.insert(0, f"{actor.name} casts {spell.name} and restores {healed} hit points.")
        return Effect(lines=lines, healing=healed)

    def _prismatic_rays(self, actor, target, depth: int) -> list[Effect]:
        roll = self.rng.randint(1, 8)
        if roll == 8 and depth < 4:
            return self._prismatic_rays(actor, target, depth + 1) + self._prismatic_rays(actor, target, depth + 1)
        return [self._prismatic_ray(actor, target, roll)]

    def _prismatic_ray(self, actor, target, roll: int) -> Effect:
        if roll == 2:
            return self._rolled_save(actor, target, dc=int(actor.spell_save_dc or 0), ability="dexterity", success="half", damage=[("acid", "12d6")], rng=self.rng, label="Orange ray")
        if roll == 4:
            return self._rolled_save(actor, target, dc=int(actor.spell_save_dc or 0), ability="dexterity", success="half", damage=[("poison", "12d6")], rng=self.rng, label="Green ray")
        if roll == 6:
            effect = self._rolled_save(actor, target, dc=int(actor.spell_save_dc or 0), ability="dexterity", success="none", damage=[], rng=self.rng, label="Indigo ray")
            if self._failed(effect):
                note = add_mark(target, actor, condition="restrained", until="fight", repeat=True, save="constitution", dc=int(actor.spell_save_dc or 0), track="thirds", spell="prismatic-spray")
                if note:
                    effect.lines.append(note)
            return effect
        return Effect(lines=[f"Ray {roll} does not affect {target.name}."])

    def _advance(self) -> None:
        if self.over:
            return
        self._prepared = None
        self.turn_index += 1
        if self.turn_index >= len(self.order):
            self.turn_index = 0
            self.round += 1
            self._tick_lingers()
            if self.round > self.round_cap:
                self.over = True
                self.winner = self.winner or "draw"
                self.log.append(f"The round cap of {self.round_cap} is reached.")

    def _check_end(self) -> bool:
        if self.winner:
            return True
        party = any(creature.hp > 0 and creature.side == "party" for creature in self.combatants)
        enemy = any(creature.hp > 0 and creature.side == "enemy" for creature in self.combatants)
        if party and enemy:
            return False
        self.over = True
        if party:
            self.winner = "party"
        elif enemy:
            self.winner = "enemy"
        else:
            self.winner = "draw"
        self.log.append(f"Fight over. Winner: {self.winner}.")
        return True

    def summary(self, fight_id: str) -> dict:
        party = [creature for creature in self.combatants if creature.side == "party"]
        enemy = [creature for creature in self.combatants if creature.side == "enemy"]
        rounds = self.round_cap if self.round > self.round_cap else self.round
        party_win = None
        if self.winner == "party":
            party_win = 1
        elif self.winner == "enemy":
            party_win = 0
        return {
            "fight_id": fight_id,
            "label": self.label,
            "policy": self.enemy_policy,
            "seed": self.seed,
            "winner": self.winner or "unfinished",
            "rounds": rounds,
            "party_size": len(party),
            "enemy_count": len(enemy),
            "party_hp_remaining": sum(creature.hp for creature in party),
            "enemy_hp_remaining": sum(creature.hp for creature in enemy),
            "party_down": sum(1 for creature in party if creature.hp <= 0),
            "enemy_down": sum(1 for creature in enemy if creature.hp <= 0),
            "damage_by_party": self.damage_dealt["party"],
            "damage_by_enemy": self.damage_dealt["enemy"],
            "llm_turns": sum(1 for turn in self.turns if turn["source"] == "llm"),
            "fallback_turns": sum(1 for turn in self.turns if turn["source"] == "fallback"),
            "mean_enemy_cr": round(self.mean_cr, 4),
            "mean_attack_bonus": round(self.mean_attack_bonus, 4),
            "start_distance": round(self.start_distance, 4),
            "party_win": party_win,
        }
