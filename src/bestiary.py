"""Turn SRD API records into combat data the engine can resolve."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, fields

from src.conditions import affliction, parse_rider, parse_special
from src.zones import NOT_AURA, zone_spec

ABILITIES = (
    "strength",
    "dexterity",
    "constitution",
    "intelligence",
    "wisdom",
    "charisma",
)
SHORT_ABILITY = {
    "strength": "str",
    "dexterity": "dex",
    "constitution": "con",
    "intelligence": "int",
    "wisdom": "wis",
    "charisma": "cha",
}
ABILITY_FROM_SHORT = {short: full for full, short in SHORT_ABILITY.items()}
KNOWN_AREAS = {"sphere", "cube", "cone", "line", "cylinder"}

EXCLUDED_CAST = "casting time is not an action or a bonus action"
EXCLUDED_FIELDS = "no structured damage, healing, attack, or save"
EXCLUDED_BOTH = "structured healing and damage together"
EXCLUDED_AREA = "area type is not supported"
EXCLUDED_RIDER = "the damage waits for a later hit"


def ability_mod(score: int) -> int:
    return (score - 10) // 2


@dataclass
class AttackOption:
    name: str
    resolution: str
    attack_bonus: int | None = None
    reach_feet: int | None = None
    range_feet: int | None = None
    damage: list[dict] = field(default_factory=list)
    dc_ability: str | None = None
    dc_value: int | None = None
    dc_success: str | None = None
    recharge_min: int | None = None
    uses: int | None = None
    area_type: str | None = None
    area_size: int | None = None
    disagrees: bool = False
    activation: str = "action"
    mastery: str | None = None
    light: bool = False
    finesse: bool = False
    heavy: bool = False
    two_handed: bool = False
    versatile: bool = False
    thrown: bool = False
    crossbow: bool = False
    ability: str = ""
    rider: dict = field(default_factory=dict)
    needs_sight: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> AttackOption:
        names = {item.name for item in fields(cls)}
        return cls(**{key: data[key] for key in names if key in data})


@dataclass
class SpellRecord:
    index: str
    name: str
    level: int
    kind: str
    casting_time: str
    duration: str
    concentration: bool
    range_mode: str
    range_feet: int
    area_type: str | None
    area_size: int | None
    attack_type: str | None
    dc_ability: str | None
    dc_success: str | None
    damage: list[dict]
    healing: dict[str, str]
    ongoing: bool
    exclude_reason: str | None
    classes: list[str]
    activation: str = "action"
    has_components: bool = True
    verbal: bool = False
    needs_sight: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> SpellRecord:
        names = {item.name for item in fields(cls)}
        return cls(**{key: data[key] for key in names if key in data})


@dataclass
class ReactionRecord:
    """One reaction the fight can offer, or a sentence it only lists.

    effect is ac, subtract, spell, or unresolved. A spell record points at
    shield or counterspell. Anything else stays unresolved and is not offered.
    """

    name: str
    effect: str
    triggers: list[str] = field(default_factory=list)
    bonus: int = 0
    die: str = ""
    range_feet: int = 0
    melee_only: bool = False
    needs_weapon: bool = False
    pool: str = ""
    uses: int | None = None
    spell_index: str = ""
    text: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ReactionRecord:
        names = {item.name for item in fields(cls)}
        return cls(**{key: data[key] for key in names if key in data})


@dataclass
class MonsterRecord:
    index: str
    name: str
    ac: int
    hp: int
    speed: int
    abilities: dict[str, int]
    save_bonuses: dict[str, int]
    challenge_rating: float
    attacks: list[AttackOption]
    multiattack: list[tuple[str, int]]
    resistances: list[str]
    immunities: list[str]
    vulnerabilities: list[str]
    excluded_actions: list[str]
    disagreements: list[str]
    creature_type: str
    size: str = "Medium"
    reactions: list[ReactionRecord] = field(default_factory=list)
    spell_save_dc: int = 0
    condition_immunities: list[str] = field(default_factory=list)
    senses: dict = field(default_factory=dict)
    auras: list[dict] = field(default_factory=list)
    starts_invisible: bool = False
    pack_tactics: bool = False

    def to_dict(self) -> dict:
        data = asdict(self)
        data["multiattack"] = [list(part) for part in self.multiattack]
        return data


def parse_range(text: str) -> tuple[str, int]:
    lowered = (text or "").strip().lower()
    # A few 2024 records glue the component line onto the range, as in "Touch Component: V, S".
    lowered = lowered.split(" component", 1)[0].strip()
    if lowered == "self":
        return "self", 0
    if lowered == "touch":
        return "touch", 5
    match = re.search(r"(\d+)\s*feet", lowered)
    if match:
        return "ranged", int(match.group(1))
    if "unlimited" in lowered or lowered == "sight":
        return "special", 1000
    return "special", 0


def parse_speed(speed: dict | None) -> int:
    """Highest listed speed. The grid has no altitude, so fly is ground speed."""
    values: list[int] = []
    for key in ("walk", "fly", "swim", "climb", "burrow"):
        raw = (speed or {}).get(key)
        if not raw:
            continue
        match = re.search(r"(\d+)", str(raw))
        if match:
            values.append(int(match.group(1)))
    return max(values) if values else 0


def parse_ac(armor_class) -> int:
    if isinstance(armor_class, list) and armor_class:
        return int(armor_class[0]["value"])
    if isinstance(armor_class, dict):
        return int(armor_class["value"])
    return int(armor_class)


def _typed_name(damage_type) -> str:
    if isinstance(damage_type, dict):
        return str(damage_type.get("index") or damage_type.get("name") or "untyped").lower()
    if damage_type:
        return str(damage_type).lower()
    return "untyped"


def _as_list(value) -> list:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return value
    return []


def _text(value) -> str:
    if isinstance(value, list):
        return "\n".join(str(item) for item in value if item)
    if value is None:
        return ""
    return str(value)


def _clean_table(table) -> dict[str, str]:
    cleaned = {}
    for key, value in (table or {}).items():
        if value is None:
            continue
        cleaned[str(key)] = str(value).replace(" ", "")
    return cleaned


def _damage_specs(raw_damage) -> list[dict]:
    specs = []
    for part in _as_list(raw_damage):
        if not isinstance(part, dict):
            continue
        by_slot = _clean_table(part.get("damage_at_slot_level"))
        by_level = _clean_table(part.get("damage_at_character_level"))
        dice = part.get("damage_dice")
        if not by_slot and not by_level and not dice:
            continue
        specs.append(
            {
                "type": _typed_name(part.get("damage_type")),
                "by_slot": by_slot,
                "by_level": by_level,
                "dice": dice.replace(" ", "") if isinstance(dice, str) else None,
            }
        )
    return specs


def _has_structured_damage(specs: list[dict]) -> bool:
    return any(spec["by_slot"] or spec["by_level"] or spec["dice"] for spec in specs)


def _reach_and_range(desc: str) -> tuple[int | None, int | None]:
    reach = None
    normal = None
    reach_match = re.search(r"reach (\d+) ft", desc or "", re.I)
    if reach_match:
        reach = int(reach_match.group(1))
    range_match = re.search(r"range (\d+)(?:/\d+)? ft", desc or "", re.I)
    if range_match:
        normal = int(range_match.group(1))
    return reach, normal


def _success(dc: dict) -> str:
    return str(dc.get("dc_success") or dc.get("success_type") or "none").lower()


def _half_text(desc: str) -> bool:
    text = _text(desc).lower()
    return "half as much" in text or "success: half" in text


def _disagrees(desc: str, success: str) -> bool:
    text = _text(desc).lower()
    half = _half_text(text)
    if success == "none" and half:
        return True
    return success == "half" and not half and ("success: none" in text or "no damage" in text)


_AREA_PHRASES = (
    (re.compile(r"(\d+)-foot-long(?:,\s*\d+-foot-wide)?\s+line", re.I), "line"),
    (re.compile(r"(\d+)-foot-radius sphere", re.I), "sphere"),
    (re.compile(r"(\d+)-foot radius", re.I), "sphere"),
    (re.compile(r"(\d+)-foot emanation", re.I), "sphere"),
    (re.compile(r"(\d+)-foot cone", re.I), "cone"),
    (re.compile(r"(\d+)-foot cube", re.I), "cube"),
    (re.compile(r"(\d+)-foot line", re.I), "line"),
    (re.compile(r"(\d+) foot line", re.I), "line"),
)


def area_phrase(desc: str) -> tuple[str | None, int | None]:
    """Monster actions store area size in prose. Spells use area_of_effect instead."""
    for pattern, kind in _AREA_PHRASES:
        match = pattern.search(desc or "")
        if match:
            return kind, int(match.group(1))
    return None, None


def _usage(action: dict) -> tuple[int | None, int | None]:
    usage = action.get("usage") or {}
    kind = str(usage.get("type") or "")
    if kind == "recharge on roll":
        return int(usage["min_value"]), None
    if "per day" in kind:
        return None, int(usage.get("times") or 0)
    return None, None


_SAVE_ABILITY = re.compile(
    r"(strength|dexterity|constitution|intelligence|wisdom|charisma)\s+saving throw",
    re.I,
)
_HEAL_PROSE = re.compile(
    r"regains? (?:a number of )?hit points equal to (\d+d\d+)(?:\s+plus your spellcasting ability modifier)?",
    re.I,
)
_MOD_DAMAGE = re.compile(
    r"([A-Za-z]+) damage equal to (\d+d\d+) plus your spellcasting ability modifier",
    re.I,
)
_DART_COUNT = re.compile(r"\b(one|two|three|four|five|\d+)\b[\w\s,]{0,40}\bdarts?\b", re.I)
_DART_DICE = re.compile(r"(\d+)d(\d+)\s*\+\s*(\d+)")
_UPCAST = re.compile(r"increases by (\d+)d(\d+).*?(?:above|higher than)\s+(\d+)", re.I | re.S)
_CANTRIP_PAIR = re.compile(r"(\d+)(?:st|nd|rd|th)?(?:\s+level)?\s*\((\d+d\d+)\)", re.I)
_RIDER = re.compile(r"next time you hit", re.I)
_COUNT_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
_MAKES_ATTACKS = re.compile(
    r"makes (one|two|three|four|five|six|\d+) ([A-Za-z][\w'/-]*(?:\s+or\s+[A-Za-z][\w'/-]*)?) attacks",
    re.I,
)


def spell_activation(casting_time: str) -> str | None:
    """An action, bonus action, or reaction. Longer casts return None."""
    text = (casting_time or "").strip().lower()
    if text.startswith("reaction"):
        return "reaction"
    if text.startswith("bonus action") or text.startswith("1 bonus action"):
        return "bonus"
    if text.startswith("action") or text.startswith("1 action"):
        return "action"
    return None


def _bump_dice(expr: str, add_count: int, add_sides: int) -> str | None:
    text = expr.replace(" ", "")
    suffix = ""
    for token in ("+MOD", "-MOD"):
        if text.endswith(token):
            suffix = token
            text = text[: -len(token)]
            break
    match = re.fullmatch(r"(\d+)d(\d+)([+-]\d+)?", text)
    if not match or int(match.group(2)) != add_sides:
        return None
    count = int(match.group(1)) + add_count
    return f"{count}d{add_sides}{match.group(3) or ''}{suffix}"


def _expand_table(table: dict[str, str], prose: str) -> dict[str, str]:
    """Fill higher slots from a sentence such as 'increases by 1d6 above 3'."""
    match = _UPCAST.search(prose or "")
    if not match or not table:
        return table
    add_n, add_sides, above = int(match.group(1)), int(match.group(2)), int(match.group(3))
    eligible = [int(key) for key in table if key.isdigit() and int(key) <= above]
    if not eligible:
        return table
    base_level = max(eligible)
    base_expr = table[str(base_level)]
    expanded = dict(table)
    for level in range(above + 1, 10):
        if str(level) in expanded:
            continue
        updated = _bump_dice(base_expr, add_n * (level - base_level), add_sides)
        if updated is None:
            break
        expanded[str(level)] = updated
    return expanded


def _apply_cantrip(specs: list[dict], description: str) -> None:
    if not specs or specs[0].get("by_level"):
        return
    lowered = description.lower()
    if "cantrip upgrade" not in lowered and "when you reach" not in lowered:
        return
    section = description[lowered.find("cantrip upgrade") :] if "cantrip upgrade" in lowered else description
    base = None
    if specs[0].get("by_slot", {}).get("0"):
        base = specs[0]["by_slot"]["0"]
    elif specs[0].get("dice"):
        base = specs[0]["dice"]
    table: dict[str, str] = {}
    if base:
        table["1"] = base.replace(" ", "")
    for level, expr in _CANTRIP_PAIR.findall(section):
        table[str(int(level))] = expr.replace(" ", "")
    if len(table) > 1 or (table and "1" in table and any(int(key) > 1 for key in table)):
        specs[0]["by_level"] = table


def _heal_table(description: str, prose: str, level: int) -> dict[str, str]:
    match = _HEAL_PROSE.search(description)
    if not match:
        return {}
    expr = match.group(1).replace(" ", "")
    if "spellcasting ability modifier" in match.group(0).lower():
        expr = f"{expr}+MOD"
    return _expand_table({str(max(level, 1)): expr}, prose)


def _missile_table(description: str, prose: str) -> dict[str, str] | None:
    if "dart" not in description.lower():
        return None
    count_match = _DART_COUNT.search(description)
    dice_match = _DART_DICE.search(description)
    if not count_match or not dice_match:
        return None
    raw_count = count_match.group(1).lower()
    darts = _COUNT_WORDS.get(raw_count)
    if darts is None and raw_count.isdigit():
        darts = int(raw_count)
    if not darts:
        return None
    per_die = int(dice_match.group(1))
    sides = int(dice_match.group(2))
    bonus = int(dice_match.group(3))
    adds = bool(re.search(r"one more dart|1 more dart|additional dart", prose, re.I))
    table = {}
    for slot in range(1, 10):
        total = darts + (slot - 1 if adds else 0)
        table[str(slot)] = f"{total * per_die}d{sides}+{total * bonus}"
    return table


def _mod_damage_spec(description: str, prose: str, level: int) -> dict | None:
    match = _MOD_DAMAGE.search(description)
    if not match:
        return None
    expr = f"{match.group(2)}+MOD"
    return {
        "type": match.group(1).lower(),
        "by_slot": _expand_table({str(max(level, 1)): expr}, prose),
        "by_level": {},
        "dice": None,
    }


def classify_spell(raw: dict) -> SpellRecord:
    description = _text(raw.get("description") if raw.get("description") is not None else raw.get("desc"))
    higher = _text(raw.get("higher_level"))
    prose = f"{description}\n{higher}"
    specs = _damage_specs(raw.get("damage"))
    _apply_cantrip(specs, description)
    for spec in specs:
        if spec.get("by_slot"):
            spec["by_slot"] = _expand_table(spec["by_slot"], prose)
    healing = _clean_table(raw.get("heal_at_slot_level"))
    level = int(raw.get("level") or 0)
    if healing:
        healing = _expand_table(healing, prose)
    else:
        healing = _heal_table(description, prose, level)
    if not _has_structured_damage(specs):
        missiles = _missile_table(description, prose)
        if missiles:
            specs = [{"type": "force", "by_slot": missiles, "by_level": {}, "dice": None}]
        else:
            mod_spec = _mod_damage_spec(description, prose, level)
            if mod_spec:
                specs = [mod_spec]
    has_damage = _has_structured_damage(specs)
    has_heal = bool(healing)
    dc = raw.get("dc") or {}
    area = raw.get("area_of_effect") or {}
    area_type = str(area.get("type") or "").lower() or None
    area_size = int(area["size"]) if area.get("size") else None
    if area_type is None:
        area_type, area_size = area_phrase(description)
    range_mode, range_feet = parse_range(str(raw.get("range") or ""))
    attack_type = raw.get("attack_type")
    classes = [item.get("name", "") for item in raw.get("classes") or [] if isinstance(item, dict)]
    activation = spell_activation(str(raw.get("casting_time") or ""))
    save_match = _SAVE_ABILITY.search(description)
    if dc:
        dc_ability = (dc.get("dc_type") or {}).get("index")
        dc_success = _success(dc)
    elif save_match and has_damage:
        dc_ability = save_match.group(1).lower()
        dc_success = "half" if _half_text(description) else "none"
    else:
        dc_ability = None
        dc_success = None
    common = dict(
        index=raw.get("index", ""),
        name=raw.get("name", ""),
        level=level,
        casting_time=str(raw.get("casting_time") or ""),
        duration=str(raw.get("duration") or ""),
        concentration=bool(raw.get("concentration")),
        range_mode=range_mode,
        range_feet=range_feet,
        area_type=area_type,
        area_size=area_size,
        attack_type=attack_type,
        dc_ability=dc_ability,
        dc_success=dc_success,
        damage=specs,
        healing=healing,
        classes=classes,
        activation=activation or "other",
        has_components=bool(raw.get("components")),
        verbal="V" in {str(item).upper() for item in (raw.get("components") or [])},
        needs_sight="you can see" in description.lower() or "that you can see" in description.lower(),
    )
    reason = None
    kind = "excluded"
    if activation is None:
        reason = EXCLUDED_CAST
    elif str(raw.get("index") or "") in {"shield", "counterspell"}:
        kind = "reaction"
    elif area_type and area_type not in KNOWN_AREAS:
        reason = EXCLUDED_AREA
    elif _RIDER.search(description):
        reason = EXCLUDED_RIDER
    elif has_heal and has_damage:
        reason = EXCLUDED_BOTH
    elif has_heal:
        kind = "heal"
    elif attack_type and has_damage:
        kind = "attack"
    elif dc_ability and has_damage:
        kind = "save"
    elif has_damage:
        kind = "auto"
    else:
        reason = EXCLUDED_FIELDS
    ongoing = (
        kind != "excluded"
        and common["concentration"]
        and common["duration"] != "Instantaneous"
        and area_type in {"sphere", "cylinder"}
        and range_mode == "self"
    )
    record = SpellRecord(kind=kind, ongoing=ongoing, exclude_reason=reason, **common)
    if record.index in NOT_AURA:
        record.ongoing = False
        record.area_type = None
        record.area_size = None
    spec = zone_spec(record.index)
    if spec is not None and record.activation in {"action", "bonus"}:
        record.kind = "zone"
        record.ongoing = False
        record.exclude_reason = None
        record.area_type = spec.area_type
        record.area_size = spec.area_size
        if spec.save:
            record.dc_ability = spec.save
            record.dc_success = spec.success
    found = affliction(record.index)
    if found and record.activation in {"action", "bonus"} and spec is None:
        if found.get("concentration"):
            record.concentration = True
        if found.get("area") in KNOWN_AREAS:
            record.area_type = found["area"]
            record.area_size = int(found.get("area_feet") or 0)
        if found.get("mode") == "heal":
            record.kind = "heal"
            record.healing = dict(found.get("healing") or {})
            record.exclude_reason = None
        elif found.get("mode") == "cast":
            record.kind = "afflict"
            record.exclude_reason = None
            if found.get("save") and not record.dc_ability:
                record.dc_ability = found["save"]
                record.dc_success = "none"
        elif found.get("special") == "prismatic":
            record.kind = "afflict"
            record.exclude_reason = None
        elif found.get("mode") == "rider" and record.kind == "excluded" and found.get("damage"):
            record.kind = "save" if found.get("save") else "auto"
            record.exclude_reason = None
            record.damage = [{
                "type": found.get("damage_type") or "untyped",
                "dice": found["damage"],
                "by_slot": {},
                "by_level": {},
            }]
            if found.get("save"):
                record.dc_ability = found["save"]
                record.dc_success = "none"
    return record


def _save_bonuses(raw: dict) -> dict[str, int]:
    bonuses: dict[str, int] = {}
    for prof in raw.get("proficiencies") or []:
        index = (prof.get("proficiency") or {}).get("index") or ""
        if not index.startswith("saving-throw-"):
            continue
        short = index.removeprefix("saving-throw-")
        full = ABILITY_FROM_SHORT.get(short)
        if full:
            bonuses[full] = int(prof["value"])
    for ability in ABILITIES:
        bonuses.setdefault(ability, ability_mod(int(raw[ability])))
    return bonuses


def _damage_names(items) -> list[str]:
    names = []
    for item in items or []:
        if isinstance(item, dict):
            text = item.get("index") or item.get("name") or ""
        else:
            text = item
        cleaned = str(text).strip().lower()
        if cleaned:
            names.append(cleaned)
    return names


def _challenge(value) -> float:
    if value is None or value == "":
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        return float(numerator) / float(denominator)
    return float(text)


def _damage_pairs(action: dict) -> list[dict]:
    pairs = [
        {"type": spec["type"], "expr": spec["dice"]}
        for spec in _damage_specs(action.get("damage"))
        if spec.get("dice")
    ]
    desc = _text(action.get("desc")).lower()
    if len(pairs) > 1 and "if the attack roll had advantage" in desc:
        return pairs[:1]
    return pairs


def _multiattack_rows(action: dict) -> list[dict]:
    options = ((action.get("action_options") or {}).get("from") or {}).get("options")
    if isinstance(options, list) and options:
        return [row for row in options if isinstance(row, dict)]
    rows = action.get("actions")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    return []


def _multiattack(action: dict) -> tuple[list[tuple[str, int]], list[str]]:
    parts: list[tuple[str, int]] = []
    excluded: list[str] = []
    for row in _multiattack_rows(action):
        name = str(row.get("action_name") or "").strip()
        try:
            count = int(row.get("count"))
        except (TypeError, ValueError):
            excluded.append(name or "Multiattack")
            continue
        if not name:
            excluded.append("Multiattack")
            continue
        parts.append((name, count))
    match = _MAKES_ATTACKS.search(_text(action.get("desc")))
    if not match:
        return parts, excluded
    raw_count = match.group(1).lower()
    sentence_count = _COUNT_WORDS[raw_count] if raw_count in _COUNT_WORDS else int(raw_count)
    sentence_name = " ".join(match.group(2).split())
    if " or " in sentence_name.lower():
        return parts, excluded
    kept: list[tuple[str, int]] = []
    found = False
    for name, count in parts:
        if name.lower() == sentence_name.lower():
            kept.append((name, max(count, sentence_count)))
            found = True
        else:
            excluded.append(name)
    if not found:
        kept.append((sentence_name, sentence_count))
    return kept, excluded


def _measure(desc: str) -> tuple[int | None, int | None]:
    reach, normal = _reach_and_range(desc)
    if reach is None and normal is None:
        within = re.search(r"within (\d+) feet", desc or "", re.I)
        if within:
            normal = int(within.group(1))
    return reach, normal


_AC_RESPONSE = re.compile(
    r"^the .+ adds (\d+) to its ac against (?:that|the) attack, possibly causing (?:it|the attack) to miss\.?$",
    re.I,
)
_CAST_RESPONSE = re.compile(
    r"^the .+ casts (?:counterspell or shield|shield or counterspell) in response to the spell's trigger, "
    r"using the same spellcasting ability as spellcasting\.?$",
    re.I,
)
_SPELL_DC = re.compile(r"spell save DC (\d+)", re.I)


def _normalize(text: str) -> str:
    cleaned = (text or "").replace("\u2019", "'").replace("\u2018", "'")
    return re.sub(r"\s+", " ", cleaned).strip()


def _trigger_and_response(desc: str) -> tuple[str, str]:
    text = _normalize(desc)
    match = re.search(r"\bresponse:\s*", text, re.I)
    if not match:
        return "", text
    return text[: match.start()].strip(), text[match.end() :].strip()


def _parse_reaction(action: dict) -> list[ReactionRecord]:
    """Offer a reaction only when the whole response is one supported pattern."""
    name = str(action.get("name") or "Unnamed")
    desc = _text(action.get("desc"))
    trigger, response = _trigger_and_response(desc)
    flat = _normalize(f"{trigger} {response}").lower()
    _, uses = _usage(action)
    ac_match = _AC_RESPONSE.match(response)
    if ac_match:
        return [
            ReactionRecord(
                name=name,
                effect="ac",
                triggers=["hit"],
                bonus=int(ac_match.group(1)),
                melee_only="melee attack" in flat,
                needs_weapon="holding a weapon" in flat,
                text=desc,
            )
        ]
    if _CAST_RESPONSE.match(response):
        pool = name if uses is not None else ""
        shared = int(uses) if uses is not None else None
        return [
            ReactionRecord(
                name="Shield",
                effect="spell",
                triggers=["hit", "missile"],
                pool=pool,
                uses=shared,
                spell_index="shield",
                text=desc,
            ),
            ReactionRecord(
                name="Counterspell",
                effect="spell",
                triggers=["counterspell"],
                range_feet=60,
                pool=pool,
                uses=shared,
                spell_index="counterspell",
                text=desc,
            ),
        ]
    return [ReactionRecord(name=name, effect="unresolved", text=desc)]


def _spell_save_dc(raw: dict) -> int:
    for key in ("actions", "special_abilities", "bonus_actions", "reactions"):
        for action in raw.get(key) or []:
            if not isinstance(action, dict):
                continue
            match = _SPELL_DC.search(_text(action.get("desc")))
            if match:
                return int(match.group(1))
    return 0


def _append_action(bucket: dict, action: dict, activation: str) -> None:
    name = action.get("name") or "Unnamed"
    if action.get("multiattack_type") or action.get("action_options") or name == "Multiattack":
        parts, skipped = _multiattack(action)
        bucket["multiattack"].extend(parts)
        bucket["excluded"].extend(skipped)
        return
    if name == "Nimble Escape":
        bucket["attacks"].append(AttackOption(name=name, resolution="disengage", activation=activation))
        return
    pairs = _damage_pairs(action)
    desc = _text(action.get("desc"))
    reach, normal = _measure(desc)
    recharge_min, uses = _usage(action)
    dc = action.get("dc") or {}
    rider = parse_rider(desc)
    if reach is not None:
        rider["reach"] = reach
    if action.get("attack_bonus") is not None and pairs:
        if reach is None and normal is None:
            reach = 5
        bucket["attacks"].append(
            AttackOption(
                name=name,
                resolution="attack",
                attack_bonus=int(action["attack_bonus"]),
                reach_feet=reach,
                range_feet=normal,
                damage=pairs,
                recharge_min=recharge_min,
                uses=uses,
                activation=activation,
                rider=rider,
                needs_sight=bool(rider.get("needs_sight")),
            )
        )
        return
    save_text = bool(rider.get("save_condition") or rider.get("first"))
    if dc.get("dc_value") is not None and (pairs or activation == "bonus" or save_text):
        success = _success(dc)
        disagrees = _disagrees(desc, success)
        if disagrees:
            bucket["disagreements"].append(name)
        area_type, area_size = area_phrase(desc)
        if (
            area_type in {"sphere", "cube"}
            and normal
            and re.search(r"at a point|centered on that point", desc, re.I)
        ):
            rider["aimed"] = True
        bucket["attacks"].append(
            AttackOption(
                name=name,
                resolution="save",
                reach_feet=reach,
                range_feet=normal,
                damage=pairs,
                dc_ability=(dc.get("dc_type") or {}).get("index"),
                dc_value=int(dc["dc_value"]),
                dc_success=success,
                recharge_min=recharge_min,
                uses=uses,
                area_type=area_type,
                area_size=area_size,
                disagrees=disagrees,
                activation=activation,
                rider=rider,
                needs_sight=bool(rider.get("needs_sight")),
            )
        )
        return
    if rider.get("become") == "invisible" and activation == "bonus":
        bucket["attacks"].append(
            AttackOption(name=name, resolution="invisible", activation=activation, rider=rider)
        )
        return
    if save_text and rider.get("save_dc"):
        bucket["attacks"].append(
            AttackOption(
                name=name,
                resolution="save",
                reach_feet=reach,
                range_feet=normal,
                damage=pairs,
                dc_ability=rider.get("save_ability"),
                dc_value=int(rider["save_dc"]),
                dc_success="none",
                recharge_min=recharge_min,
                uses=uses,
                activation=activation,
                rider=rider,
            )
        )
        return
    bucket["excluded"].append(name)


def classify_monster(raw: dict) -> MonsterRecord:
    bucket: dict = {"attacks": [], "multiattack": [], "excluded": [], "disagreements": [], "reactions": []}
    for action in raw.get("actions") or []:
        if isinstance(action, dict):
            _append_action(bucket, action, "action")
    for action in raw.get("bonus_actions") or []:
        if isinstance(action, dict):
            _append_action(bucket, action, "bonus")
    for action in raw.get("reactions") or []:
        if not isinstance(action, dict):
            continue
        for record in _parse_reaction(action):
            bucket["reactions"].append(record)
            if record.effect == "unresolved" and record.name not in bucket["excluded"]:
                bucket["excluded"].append(record.name)
    abilities = {ability: int(raw[ability]) for ability in ABILITIES}
    auras = []
    starts_invisible = False
    pack_tactics = False
    for action in raw.get("special_abilities") or []:
        if not isinstance(action, dict):
            continue
        parsed = parse_special(_text(action.get("desc")))
        if parsed.get("pack_tactics"):
            pack_tactics = True
        if parsed.get("starts_invisible"):
            starts_invisible = True
        if parsed.get("aura"):
            auras.append(parsed["aura"])
    senses = {}
    for key, value in (raw.get("senses") or {}).items():
        match = re.search(r"(\d+)", str(value))
        if match and key in {"blindsight", "truesight", "tremorsense", "darkvision"}:
            senses[key] = int(match.group(1))
    immunities = []
    for item in raw.get("condition_immunities") or []:
        if isinstance(item, dict):
            immunities.append(str(item.get("index") or item.get("name") or "").lower())
        elif item:
            immunities.append(str(item).lower())
    return MonsterRecord(
        index=raw.get("index", ""),
        name=raw.get("name", ""),
        ac=parse_ac(raw.get("armor_class")),
        hp=int(raw.get("hit_points") or 1),
        speed=parse_speed(raw.get("speed")),
        abilities=abilities,
        save_bonuses=_save_bonuses(raw),
        challenge_rating=_challenge(raw.get("challenge_rating")),
        attacks=bucket["attacks"],
        multiattack=bucket["multiattack"],
        resistances=_damage_names(raw.get("damage_resistances")),
        immunities=_damage_names(raw.get("damage_immunities")),
        vulnerabilities=_damage_names(raw.get("damage_vulnerabilities")),
        excluded_actions=bucket["excluded"],
        disagreements=bucket["disagreements"],
        creature_type=str(raw.get("type") or ""),
        size=str(raw.get("size") or "Medium"),
        reactions=bucket["reactions"],
        spell_save_dc=_spell_save_dc(raw),
        condition_immunities=immunities,
        senses=senses,
        auras=auras,
        starts_invisible=starts_invisible,
        pack_tactics=pack_tactics,
    )


def lookup_expression(table: dict[str, str], level: int) -> str | None:
    eligible = [int(key) for key in table if int(key) <= level]
    if not eligible:
        return None
    return table[str(max(eligible))]


def spell_damage_parts(spell: SpellRecord, slot_level: int, caster_level: int) -> list[tuple[str, str]]:
    parts: list[tuple[str, str]] = []
    for spec in spell.damage:
        expr = None
        if spell.level == 0 and spec.get("by_level"):
            expr = lookup_expression(spec["by_level"], caster_level)
        elif spec.get("by_slot"):
            expr = lookup_expression(spec["by_slot"], slot_level)
        elif spec.get("by_level"):
            expr = lookup_expression(spec["by_level"], caster_level)
        elif spec.get("dice"):
            expr = spec["dice"]
        if expr:
            parts.append((spec["type"], expr.replace(" ", "")))
    return parts


def spell_heal_expression(spell: SpellRecord, slot_level: int, modifier: int) -> str | None:
    expr = lookup_expression(spell.healing, slot_level)
    if expr is None:
        return None
    from src.dice import substitute_mod

    return substitute_mod(expr, modifier)


def coverage(spells: list[SpellRecord], monsters: list[MonsterRecord]) -> dict:
    spell_counts: dict[str, int] = {}
    reasons: dict[str, int] = {}
    for spell in spells:
        spell_counts[spell.kind] = spell_counts.get(spell.kind, 0) + 1
        if spell.exclude_reason:
            reasons[spell.exclude_reason] = reasons.get(spell.exclude_reason, 0) + 1
    with_attack = sum(1 for monster in monsters if monster.attacks)
    disagreements = sum(len(monster.disagreements) for monster in monsters)
    return {
        "spell_total": len(spells),
        "spells_by_kind": spell_counts,
        "exclude_reasons": reasons,
        "monster_total": len(monsters),
        "monsters_with_resolvable_action": with_attack,
        "monsters_without_resolvable_action": len(monsters) - with_attack,
        "save_text_disagreements": disagreements,
    }
