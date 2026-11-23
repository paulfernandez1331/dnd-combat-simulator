"""Import one public D&D Beyond character.

D&D Beyond does not publish a supported API. The character app reads a public
sheet from character-service. A private sheet comes back unauthorized. This
module fetches the one id the user pasted and keeps that JSON under data/imports.
Sample characters are the public sheets already saved in data/samples. Scores,
armor, weapons, and actions come from the structured fields on that sheet.
"""

from __future__ import annotations

import json
import math
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from src.api_client import USER_AGENT, CatalogError
from src.bestiary import AttackOption, ReactionRecord, SpellRecord, ability_mod
from src.combatant import Combatant, blank_abilities, save_bonuses_for
from src.feats import DAMAGE_TYPES, QUIET, STYLES, feat_key
from src.paths import IMPORTS, SAMPLES

CHARACTER_SERVICE = "https://character-service.dndbeyond.com/character/v5/character/{character_id}?includeCustomItems=true"
SHEET_URL = "https://www.dndbeyond.com/characters/{character_id}"
PORTRAIT_BYTES = 3_000_000
_ID = re.compile(r"characters/(\d+)")
_PORTRAIT_NAME = re.compile(r"^\d+\.portrait$")
_PORTRAIT_HOSTS = {"dndbeyond.com", "www.dndbeyond.com"}
_PORTRAIT_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
_STAT_NAMES = {
    1: "strength",
    2: "dexterity",
    3: "constitution",
    4: "intelligence",
    5: "wisdom",
    6: "charisma",
}
_MASTERY = {"sap", "slow", "vex", "nick", "graze", "push", "topple", "cleave"}
PRIVATE = (
    "That character is not public. Set Character Privacy to Public on D&D Beyond and try the link again."
)
PASTE = "Paste a public D&D Beyond character link or a numeric id."


def parse_character_id(text: str) -> int | None:
    raw = (text or "").strip()
    if raw.isdigit():
        return int(raw)
    match = _ID.search(raw)
    if match:
        return int(match.group(1))
    return None


def fetch_public_character(
    character_id: int,
    *,
    fetcher=None,
    imports_dir: Path | None = None,
    image_fetcher=None,
) -> dict:
    url = CHARACTER_SERVICE.format(character_id=int(character_id))
    payload = fetcher(url) if fetcher is not None else _get_json(url)
    character = _unwrap(payload)
    folder = imports_dir or IMPORTS
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{int(character_id)}.json").write_text(json.dumps(character), encoding="utf-8")
    save_character_portrait(character, folder=folder, fetcher=image_fetcher)
    return character


def _unarmed_attack(abilities, proficiency, profile, attacks) -> AttackOption:
    """Tavern Brawler is 1d4. Unarmed Fighting is 1d6, or 1d8 with nothing in hand."""
    if profile.get("unarmed_fighting"):
        bare = not profile.get("has_shield") and not attacks
        die = "1d8" if bare else "1d6"
    else:
        die = "1d4"
    modifier = ability_mod(abilities.get("strength", 10))
    expression = f"{die}{modifier:+d}" if modifier else die
    return AttackOption(
        name="Unarmed Strike",
        resolution="attack",
        attack_bonus=int(proficiency) + modifier,
        reach_feet=5,
        damage=[{"type": "bludgeoning", "expr": expression}],
        ability="strength",
    )


def _keep_free_casts(profile, spells, uses_max, uses_left) -> None:
    """One free casting this fight when a feature action or the grant itself names a spell the fight can roll."""
    free = {}
    for item in profile.pop("free_cast_names", []) or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "")
        try:
            uses = int(item.get("uses") or 0)
        except (TypeError, ValueError):
            uses = 0
        if uses <= 0 or not name:
            continue
        if "left" in item:
            try:
                left = int(item.get("left") or 0)
            except (TypeError, ValueError):
                left = 0
        else:
            left = uses
        for spell in spells:
            if spell.name.lower() != name.lower():
                continue
            if spell.kind == "excluded" or spell.index in {"misty-step", "invisibility"}:
                continue
            uses_max[spell.name] = uses
            uses_left[spell.name] = max(0, left)
            free[spell.index] = spell.name
            break
    if free:
        profile["free_spells"] = free


def public_character(
    character: dict,
    catalog,
    position: tuple[int, int],
    creature_id: str,
    imports_dir: Path | None = None,
) -> Combatant:
    abilities = _abilities(character)
    level = sum(int(item.get("level") or 0) for item in _classes(character))
    level = max(level, 1)
    proficiency = math.ceil(level / 4) + 1
    class_name, subclass = _class_label(character)
    species = _species(character)
    speed, speed_note = _speed(character)
    max_hp, hp_note = _hit_points(character, abilities, level)
    ac, ac_note = _armor_class(character, abilities)
    attacks, attack_notes = _weapons(character, abilities, proficiency)
    attacks = _scale_martial_arts(character, abilities, proficiency, attacks)
    spells, spell_notes, attack_bonus, save_dc, casting = _spells(character, catalog, abilities, proficiency)
    slots = _slots(character)
    profile, uses_max, uses_left, extra, feature_notes, reactions = _combat_profile(character, proficiency)
    if not any((attack.rider or {}).get("unarmed") for attack in attacks):
        if profile.get("tavern_brawler") or profile.get("unarmed_fighting"):
            attacks = [*attacks, _unarmed_attack(abilities, proficiency, profile, attacks)]
    granted_free = _granted_free_casts(character, abilities)
    if granted_free:
        profile["free_cast_names"] = [*(profile.get("free_cast_names") or []), *granted_free]
    _keep_free_casts(profile, spells, uses_max, uses_left)
    if profile.get("free_spells"):
        granted_feats = ("magic initiate", "fey touched", "shadow touched")
        feature_notes = [
            name for name in feature_notes
            if not feat_key(name).startswith(granted_feats)
        ]
    known = {attack.name.lower() for attack in attacks} | {spell.name.lower() for spell in spells}
    known |= {item.name.lower() for item in reactions if item.effect != "unresolved"}
    for heal in profile.get("bonus_heals") or []:
        known.add(str(heal.get("name") or "").split(":", 1)[0].strip().lower())
    if uses_max.get("Action Surge"):
        known.add("action surge")
    if profile.get("tactical_shift"):
        known.add("tactical shift")
    for name in profile.get("resolved_names") or []:
        known.add(str(name).lower())
        known.add(feat_key(str(name)))
    unresolved = []
    for note in [*feature_notes, speed_note, hp_note, ac_note, *attack_notes, *spell_notes]:
        if not note:
            continue
        head = note.split(":", 1)[0].strip()
        if head.lower() in known or feat_key(head) in known or note in unresolved:
            continue
        unresolved.append(note)
    profile.update({
        "class_name": class_name,
        "subclass": subclass,
        "level": level,
        "proficiency": proficiency,
        "species": species,
        "ddb_id": int(character.get("id") or 0),
        "ddb_url": SHEET_URL.format(character_id=int(character.get("id") or 0)),
        "unresolved": unresolved,
        "excluded_features": [],
    })
    portrait = stored_portrait(character, imports_dir or IMPORTS)
    if portrait:
        profile["portrait"] = portrait
    return Combatant(
        id=creature_id,
        name=str(character.get("name") or "Imported character"),
        side="party",
        ac=ac,
        max_hp=max_hp,
        hp=max_hp,
        speed=speed,
        abilities=abilities,
        save_bonuses=save_bonuses_for(abilities, _save_proficiencies(character), proficiency),
        position=position,
        attacks=attacks,
        spells=spells,
        spell_attack_bonus=attack_bonus,
        spell_save_dc=save_dc,
        spellcasting_mod=casting,
        caster_level=level,
        slot_max=slots,
        slots=dict(slots),
        attacks_per_action=extra,
        uses_left=uses_left,
        uses_max=uses_max,
        creature_type=species,
        traits=profile,
        reactions=reactions,
    )


def sample_characters() -> list[tuple[str, dict]]:
    if not SAMPLES.exists():
        return []
    found = []
    for path in sorted(SAMPLES.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        character = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(character, dict) or not character.get("name"):
            character = payload
        if isinstance(character, dict) and character.get("name"):
            found.append((path.stem, character))
    return found


def _unwrap(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise CatalogError("D&D Beyond did not return a character sheet.")
    status = int(payload.get("_status") or 0)
    data = payload.get("data")
    server = ""
    if isinstance(data, dict):
        server = str(data.get("serverMessage") or "")
    message = str(payload.get("message") or "")
    if status == 403 or "Unauthorized" in server or "Unauthorized" in message:
        raise CatalogError(PRIVATE)
    if payload.get("success") is False or (isinstance(data, dict) and "name" not in data and server):
        raise CatalogError(server or message or "D&D Beyond did not return that character.")
    character = data if isinstance(data, dict) and data.get("name") else payload
    if not isinstance(character, dict) or not character.get("name"):
        raise CatalogError("D&D Beyond did not return a character sheet.")
    return character


def _get_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = {"success": False, "message": body[:240]}
        if isinstance(payload, dict):
            payload["_status"] = exc.code
            return payload
        raise CatalogError(PRIVATE if exc.code == 403 else "D&D Beyond did not return that character.") from exc
    except urllib.error.URLError as exc:
        raise CatalogError(f"Could not reach D&D Beyond: {exc.reason}") from exc


def portrait_source(character: dict) -> str | None:
    """The character-sheet portrait. Backdrop art and the account picture are not this field."""
    decorations = character.get("decorations") if isinstance(character, dict) else None
    if not isinstance(decorations, dict):
        return None
    url = decorations.get("avatarUrl")
    if not isinstance(url, str):
        return None
    url = url.strip()
    return url or None


def allowed_portrait_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    host = (parsed.hostname or "").lower()
    return parsed.scheme == "https" and host in _PORTRAIT_HOSTS and parsed.path.startswith("/avatars/")


def save_character_portrait(character: dict, *, folder: Path, fetcher=None) -> str | None:
    """Copy a public sheet portrait next to the import. A bad download leaves the character intact."""
    character_id = int(character.get("id") or 0)
    path = folder / f"{character_id}.portrait"
    url = portrait_source(character)
    if character_id <= 0 or not url or not allowed_portrait_url(url):
        if path.is_file():
            path.unlink()
        return None
    try:
        kind, data = fetcher(url) if fetcher is not None else _get_image(url)
    except (OSError, urllib.error.URLError, CatalogError, TypeError, ValueError):
        return _portrait_name(path)
    if not isinstance(data, bytes):
        return _portrait_name(path)
    header = str(kind or "").split(";", 1)[0].strip().lower()
    if header == "image/jpg":
        header = "image/jpeg"
    sniffed = _image_kind(data)
    if header not in _PORTRAIT_TYPES or sniffed != header or len(data) > PORTRAIT_BYTES:
        return _portrait_name(path)
    folder.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path.name


def stored_portrait(character: dict, folder: Path) -> str | None:
    character_id = int(character.get("id") or 0)
    if character_id <= 0:
        return None
    return _portrait_name(folder / f"{character_id}.portrait")


def _portrait_name(path: Path) -> str | None:
    if not _PORTRAIT_NAME.fullmatch(path.name) or not path.is_file():
        return None
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size <= 0 or size > PORTRAIT_BYTES:
        return None
    return path.name


def _image_kind(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


class _PortraitRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not allowed_portrait_url(newurl):
            return None
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _get_image(url: str) -> tuple[str, bytes]:
    if not allowed_portrait_url(url):
        raise CatalogError("That portrait is not a D&D Beyond avatar.")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "image/*"})
    opener = urllib.request.build_opener(_PortraitRedirect)
    try:
        with opener.open(request, timeout=30) as response:
            final = response.geturl()
            if not allowed_portrait_url(final):
                raise CatalogError("That portrait is not a D&D Beyond avatar.")
            kind = response.headers.get("Content-Type") or ""
            data = response.read(PORTRAIT_BYTES + 1)
    except urllib.error.HTTPError as exc:
        raise CatalogError("D&D Beyond did not return that portrait.") from exc
    return kind, data


def _classes(character: dict) -> list[dict]:
    return [item for item in character.get("classes") or [] if isinstance(item, dict)]


def _class_label(character: dict) -> tuple[str, str]:
    rows = _classes(character)
    if not rows:
        return "", ""
    first = rows[0]
    definition = first.get("definition") or {}
    subclass = first.get("subclassDefinition") or {}
    names = [str((item.get("definition") or {}).get("name") or "") for item in rows]
    names = [name for name in names if name]
    return " / ".join(names), str(subclass.get("name") or "")


def _species(character: dict) -> str:
    race = character.get("race") or {}
    if not isinstance(race, dict):
        return ""
    return str(race.get("fullName") or race.get("baseName") or race.get("name") or "")


def _abilities(character: dict) -> dict[str, int]:
    scores = blank_abilities()
    base = {int(item.get("id") or 0): int(item.get("value") or 0) for item in character.get("stats") or [] if isinstance(item, dict)}
    bonus = {int(item.get("id") or 0): int(item.get("value") or 0) for item in character.get("bonusStats") or [] if isinstance(item, dict)}
    override = {
        int(item.get("id") or 0): item.get("value")
        for item in character.get("overrideStats") or []
        if isinstance(item, dict)
    }
    for stat_id, name in _STAT_NAMES.items():
        replacement = override.get(stat_id)
        if replacement is not None:
            scores[name] = int(replacement)
        else:
            scores[name] = int(base.get(stat_id, 10)) + int(bonus.get(stat_id, 0))
    for row in _modifiers(character):
        if str(row.get("type") or "") != "bonus" or not _modifier_counts(character, row):
            continue
        subtype = str(row.get("subType") or "")
        if not subtype.endswith("-score"):
            continue
        ability = subtype[: -len("-score")]
        if ability not in scores:
            continue
        scores[ability] += int(row.get("value") or 0)
    return scores


def _modifiers(character: dict) -> list[dict]:
    groups = character.get("modifiers") or {}
    rows = []
    if isinstance(groups, dict):
        for value in groups.values():
            if isinstance(value, list):
                rows.extend(item for item in value if isinstance(item, dict))
    return rows


def _choice_rows(character: dict) -> list[dict]:
    choices = character.get("choices") or {}
    if not isinstance(choices, dict):
        return []
    rows = []
    for key, value in choices.items():
        if key in {"choiceDefinitions", "definitionKeyNameMap"} or not isinstance(value, list):
            continue
        rows.extend(item for item in value if isinstance(item, dict))
    return rows


def _option_label(character: dict, option_id) -> str:
    if option_id is None:
        return ""
    definitions = (character.get("choices") or {}).get("choiceDefinitions") or []
    if not isinstance(definitions, list):
        return ""
    for definition in definitions:
        if not isinstance(definition, dict):
            continue
        for option in definition.get("options") or []:
            if isinstance(option, dict) and str(option.get("id")) == str(option_id):
                return str(option.get("label") or "")
    return ""


def _chosen_score(character: dict, row: dict) -> bool:
    subtype = str(row.get("subType") or "")
    if not subtype.endswith("-score"):
        return False
    ability = subtype[: -len("-score")]
    try:
        value = int(row.get("value") or 0)
    except (TypeError, ValueError):
        return False
    if value <= 0:
        return False
    component = row.get("componentId")
    choices = [
        choice for choice in _choice_rows(character)
        if str(choice.get("componentId")) == str(component)
    ]
    for choice in choices:
        label = str(choice.get("label") or "").replace(" ", "")
        if f"+{value}" not in label:
            continue
        option = _option_label(character, choice.get("optionValue")).lower()
        if option.startswith(ability):
            return True
    # Speedy's choice says "Dexterity or Constitution" and has no "+1" in the label.
    labeled = any(f"+{value}" in str(choice.get("label") or "").replace(" ", "") for choice in choices)
    if labeled:
        return False
    for choice in choices:
        option = _option_label(character, choice.get("optionValue")).lower()
        if option.startswith(ability):
            return True
    return False


def _modifier_counts(character: dict, row: dict) -> bool:
    if row.get("isGranted") is not False:
        return True
    return _chosen_score(character, row)


def _granted_row(character: dict, kind: str, subtype: str) -> dict | None:
    found = None
    for row in _modifiers(character):
        if str(row.get("type") or "") == kind and str(row.get("subType") or "") == subtype and row.get("isGranted") is not False:
            found = row
    return found


def _save_proficiencies(character: dict) -> list[str]:
    found = []
    for row in _modifiers(character):
        if str(row.get("type") or "") != "proficiency" or not _modifier_counts(character, row):
            continue
        subtype = str(row.get("subType") or "")
        for name in _STAT_NAMES.values():
            if subtype.startswith(name) and "saving" in subtype and name not in found:
                found.append(name)
    return found


def _modifier_sum(character: dict, subtype: str) -> int:
    total = 0
    for row in _modifiers(character):
        if str(row.get("type") or "") == "bonus" and str(row.get("subType") or "") == subtype and _modifier_counts(character, row):
            total += int(row.get("value") or 0)
    return total


def _hit_points(character: dict, abilities: dict[str, int], level: int) -> tuple[int, str]:
    override = character.get("overrideHitPoints")
    if override is not None:
        return max(1, int(override)), ""
    if character.get("baseHitPoints") is None:
        return 1, "Hit points were not in the JSON."
    base = int(character.get("baseHitPoints") or 0) + int(character.get("bonusHitPoints") or 0)
    base += ability_mod(abilities["constitution"]) * level
    base += _modifier_sum(character, "hit-points")
    base += _modifier_sum(character, "hit-points-per-level") * level
    return max(1, base), ""


def _speed(character: dict) -> tuple[int, str]:
    race = character.get("race") if isinstance(character.get("race"), dict) else {}
    speeds = race.get("weightSpeeds") if isinstance(race.get("weightSpeeds"), dict) else {}
    normal = speeds.get("normal") if isinstance(speeds.get("normal"), dict) else speeds
    walk = normal.get("walk") if isinstance(normal, dict) else None
    note = ""
    if not isinstance(walk, int) or walk <= 0:
        walk = 30
        note = "Speed was not in the JSON. The sheet uses 30 feet."
    heavy = _wearing_heavy(character)
    extra = 0
    for row in _modifiers(character):
        if str(row.get("type") or "") != "bonus" or str(row.get("subType") or "") != "speed":
            continue
        if row.get("isGranted") is False:
            continue
        restriction = str(row.get("restriction") or "").lower()
        if heavy and "heavy" in restriction:
            continue
        extra += int(row.get("value") or 0)
    names = _feat_names(character)
    if "speedy" in names and not _component_bonus(character, "speed", _feat_component(character, "speedy")):
        extra += 10
    gear = _gear(character)
    if not gear["has_shield"] and not gear["armor"]:
        for row in _modifiers(character):
            if str(row.get("type") or "") != "bonus" or str(row.get("subType") or "") != "unarmored-movement":
                continue
            if row.get("isGranted") is False:
                continue
            extra += int(row.get("value") or 0)
    return walk + extra, note


def _wearing_heavy(character: dict) -> bool:
    for item in _inventory(character):
        if not item.get("equipped"):
            continue
        definition = item.get("definition") or {}
        if _item_kind(definition) != "armor":
            continue
        label = str(definition.get("armorType") or definition.get("baseArmorName") or "").lower()
        if definition.get("armorTypeId") == 3 or "heavy" in label:
            return True
    return False


def _feat_names(character: dict) -> set[str]:
    names = set()
    for feat in character.get("feats") or []:
        if not isinstance(feat, dict):
            continue
        definition = feat.get("definition") if isinstance(feat.get("definition"), dict) else feat
        if definition.get("name"):
            names.add(feat_key(str(definition.get("name"))))
    for name, _identifier, _level in _feature_records(character):
        names.add(feat_key(name))
    for choice in _choice_rows(character):
        label = _option_label(character, choice.get("optionValue")) or str(choice.get("label") or "")
        if label:
            names.add(feat_key(label))
    return names


def _gear(character: dict) -> dict:
    shield = False
    armor = ""
    one_handed = 0
    melee = 0
    for item in _inventory(character):
        if not item.get("equipped"):
            continue
        definition = item.get("definition") or {}
        kind = _item_kind(definition)
        if kind == "shield":
            shield = True
            continue
        if kind == "armor":
            type_id = definition.get("armorTypeId")
            label = str(definition.get("armorType") or definition.get("baseArmorName") or "").lower()
            if type_id == 3 or "heavy" in label:
                armor = "heavy"
            elif type_id == 2 or "medium" in label:
                armor = "medium"
            elif type_id == 1 or "light" in label or definition.get("armorClass"):
                armor = armor or "light"
            continue
        if kind != "weapon":
            continue
        if int(definition.get("attackType") or 0) == 2 and "thrown" not in _property_names(definition):
            continue
        if int(definition.get("attackType") or 0) == 2:
            continue
        melee += 1
        if "two-handed" not in _property_names(definition) and "two handed" not in _property_names(definition):
            one_handed += 1
    return {
        "has_shield": shield,
        "armor": armor,
        "one_melee": melee == 1 and one_handed == 1,
    }


def _component_bonus(character: dict, subtype: str, component_id) -> bool:
    if component_id is None:
        return False
    for row in _modifiers(character):
        if str(row.get("type") or "") != "bonus" or str(row.get("subType") or "") != subtype:
            continue
        if row.get("isGranted") is False:
            continue
        if str(row.get("componentId")) != str(component_id):
            continue
        try:
            if int(row.get("value") or 0) != 0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _feat_component(character: dict, feat_name: str):
    for feat in character.get("feats") or []:
        if not isinstance(feat, dict):
            continue
        definition = feat.get("definition") if isinstance(feat.get("definition"), dict) else feat
        if feat_key(str(definition.get("name") or "")) == feat_name:
            return definition.get("id")
    for name, identifier, _level in _feature_records(character):
        if feat_key(name) == feat_name and identifier:
            return identifier
    return None


def _armor_class(character: dict, abilities: dict[str, int]) -> tuple[int, str]:
    dex = ability_mod(abilities["dexterity"])
    names = _feat_names(character)
    gear = _gear(character)
    armor = None
    shield = gear["has_shield"]
    for item in _inventory(character):
        if not item.get("equipped"):
            continue
        definition = item.get("definition") or {}
        if _item_kind(definition) == "armor" and definition.get("armorClass"):
            armor = definition
    bonus = _modifier_sum(character, "armor-class") + _modifier_sum(character, "ac")
    unarmored = _granted_row(character, "set", "unarmored-armor-class")
    if armor is not None:
        base = int(armor.get("armorClass") or 10)
        type_id = armor.get("armorTypeId")
        label = str(armor.get("armorType") or armor.get("baseArmorName") or "").lower()
        if type_id == 2 or "medium" in label:
            cap = 2
            if "medium armor master" in names and abilities.get("dexterity", 10) >= 16:
                cap = 3
            base += min(dex, cap)
        elif type_id == 3 or "heavy" in label:
            pass
        else:
            base += dex
        if shield:
            base += 2
        if (
            "defense" in names
            and gear["armor"] in {"light", "medium", "heavy"}
            and not _component_bonus(character, "armor-class", _feat_component(character, "defense"))
            and not _component_bonus(character, "ac", _feat_component(character, "defense"))
        ):
            base += 1
        return base + bonus, ""
    shield_bonus = 2 if shield else 0
    if unarmored is not None:
        stat_id = int(unarmored.get("statId") or 0)
        extra_name = _STAT_NAMES.get(stat_id)
        extra = ability_mod(abilities[extra_name]) if extra_name else 0
        return 10 + dex + extra + shield_bonus + bonus, ""
    return 10 + dex + shield_bonus + bonus, ""


def _inventory(character: dict) -> list[dict]:
    return [item for item in character.get("inventory") or [] if isinstance(item, dict)]


def _is_shield(definition: dict) -> bool:
    """A D&D Beyond shield uses filterType Armor. Its type, base name, or armorTypeId marks it."""
    filter_type = str(definition.get("filterType") or "").lower()
    type_name = str(definition.get("type") or "").lower()
    base = str(definition.get("baseArmorName") or "").lower()
    if "shield" in filter_type or "shield" in type_name or base == "shield":
        return True
    return definition.get("armorTypeId") == 4


def _item_kind(definition: dict) -> str:
    if _is_shield(definition):
        return "shield"
    label = str(definition.get("filterType") or definition.get("type") or "").lower()
    if "armor" in label:
        return "armor"
    if "weapon" in label:
        return "weapon"
    return label


def _weapons(character: dict, abilities: dict[str, int], proficiency: int) -> tuple[list[AttackOption], list[str]]:
    notes = []
    equipped = []
    stored = []
    for item in _inventory(character):
        definition = item.get("definition") or {}
        if _item_kind(definition) != "weapon":
            continue
        if item.get("equipped"):
            equipped.append(item)
        else:
            stored.append(item)
    chosen = equipped or stored
    attacks = []
    for item in chosen:
        attack, note = _weapon_attack(item, abilities, proficiency)
        if attack is None:
            if note:
                notes.append(note)
            continue
        attacks.append(attack)
    for item in stored if equipped else []:
        definition = item.get("definition") or {}
        name = _item_name(character, item)
        notes.append(f"{name}: carried, not equipped")
    return attacks, notes


def _item_name(character: dict, item: dict) -> str:
    definition = item.get("definition") or {}
    custom = ""
    for value in character.get("characterValues") or []:
        if not isinstance(value, dict) or int(value.get("typeId") or 0) != 8:
            continue
        if str(value.get("valueId")) == str(item.get("id")) and str(value.get("value") or "").strip():
            custom = str(value.get("value")).strip()
    return custom or str(definition.get("name") or "Weapon")


def _weapon_attack(item: dict, abilities: dict[str, int], proficiency: int) -> tuple[AttackOption | None, str]:
    definition = item.get("definition") or {}
    name = str(definition.get("name") or "Weapon")
    dice = _dice_string(definition.get("damage"))
    fixed = definition.get("fixedDamage")
    properties = _property_names(definition)
    finesse = "finesse" in properties
    light = "light" in properties
    ranged = int(definition.get("attackType") or 0) == 2
    ability = "dexterity" if ranged else "strength"
    if finesse and ability_mod(abilities["dexterity"]) > ability_mod(abilities["strength"]):
        ability = "dexterity"
    modifier = ability_mod(abilities[ability])
    stored = definition.get("attackBonus", definition.get("toHit"))
    bonus = int(stored) if isinstance(stored, int) else proficiency + modifier
    if dice:
        expr = dice if re.search(r"[+-]\d+$", dice) else (f"{dice}{modifier:+d}" if modifier else dice)
    elif isinstance(fixed, int):
        expr = str(int(fixed))
    else:
        return None, f"{name}: no damage dice in the JSON"
    damage_type = definition.get("damageType")
    if isinstance(damage_type, dict):
        damage_type = damage_type.get("name") or damage_type.get("index")
    reach = 10 if "reach" in properties else 5
    long_range = definition.get("longRange") or definition.get("range")
    return AttackOption(
        name=name,
        resolution="attack",
        attack_bonus=bonus,
        reach_feet=None if ranged else reach,
        range_feet=int(long_range) if ranged and long_range else None,
        damage=[{"type": str(damage_type or "untyped").lower(), "expr": expr}],
        mastery=next((name for name in _MASTERY if name in properties), None),
        light=light,
        finesse=finesse,
        heavy="heavy" in properties,
        two_handed="two-handed" in properties or "two handed" in properties,
        versatile="versatile" in properties,
        thrown="thrown" in properties,
        crossbow="crossbow" in name.lower(),
        ability=ability,
        rider={"monk_weapon": True} if definition.get("isMonkWeapon") else {},
    ), ""


def _class_level(character: dict, name: str) -> int:
    for row in _classes(character):
        definition = row.get("definition") if isinstance(row.get("definition"), dict) else {}
        if feat_key(str(definition.get("name") or "")) == feat_key(name):
            return int(row.get("level") or 0)
    return 0


def _martial_sides(level: int) -> int:
    if level >= 17:
        return 12
    if level >= 11:
        return 10
    if level >= 5:
        return 8
    return 6


def _monk_qualifies(character: dict) -> bool:
    """No armor, no shield, and every equipped weapon is a monk weapon."""
    gear = _gear(character)
    if gear["has_shield"] or gear["armor"]:
        return False
    weapons = []
    for item in _inventory(character):
        if not item.get("equipped"):
            continue
        definition = item.get("definition") or {}
        if _item_kind(definition) != "weapon":
            continue
        weapons.append(definition)
    return all(bool(definition.get("isMonkWeapon")) for definition in weapons)


def _free_hands(character: dict) -> int:
    used = 0
    for item in _inventory(character):
        if not item.get("equipped"):
            continue
        definition = item.get("definition") or {}
        kind = _item_kind(definition)
        if kind == "shield":
            used += 1
            continue
        if kind != "weapon":
            continue
        props = _property_names(definition)
        if "two-handed" in props or "two handed" in props:
            used += 2
        else:
            used += 1
    return max(0, 2 - used)


def _die_sides(expression: str) -> int:
    match = re.search(r"\d*d(\d+)", expression or "")
    return int(match.group(1)) if match else 0


def _expr_with_die(expression: str, sides: int, modifier: int) -> str:
    """Use the martial arts die when it is larger than the weapon die."""
    weapon = _die_sides(expression)
    use = sides if sides > weapon else weapon
    count = 1
    found = re.search(r"(\d*)d\d+", expression or "")
    if found and weapon >= sides and found.group(1):
        count = int(found.group(1))
    base = f"{count}d{use}"
    return f"{base}{modifier:+d}" if modifier else base


def _scale_martial_arts(character: dict, abilities: dict[str, int], proficiency: int, attacks: list[AttackOption]) -> list[AttackOption]:
    level = _class_level(character, "monk")
    if level <= 0 or "martial arts" not in _feat_names(character) or not _monk_qualifies(character):
        return attacks
    sides = _martial_sides(level)
    names = _feat_names(character)
    if "unarmed fighting" in names:
        bare = not attacks and not _gear(character)["has_shield"]
        sides = max(sides, 8 if bare else 6)
    dexterity = ability_mod(abilities.get("dexterity", 10))
    strength = ability_mod(abilities.get("strength", 10))
    ability = "dexterity" if dexterity >= strength else "strength"
    modifier = ability_mod(abilities.get(ability, 10))
    scaled = []
    for attack in attacks:
        if not (attack.rider or {}).get("monk_weapon"):
            scaled.append(attack)
            continue
        damage = list(attack.damage or [])
        if damage:
            kind = str(damage[0].get("type") or "bludgeoning")
            expr = _expr_with_die(str(damage[0].get("expr") or ""), sides, modifier)
            damage = [{"type": kind, "expr": expr}, *damage[1:]]
        attack.damage = damage
        attack.attack_bonus = proficiency + modifier
        attack.ability = ability
        attack.rider = {**(attack.rider or {}), "monk_weapon": True}
        scaled.append(attack)
    die = f"1d{sides}{modifier:+d}" if modifier else f"1d{sides}"
    scaled.append(AttackOption(
        name="Unarmed Strike",
        resolution="attack",
        attack_bonus=proficiency + modifier,
        reach_feet=5,
        damage=[{"type": "bludgeoning", "expr": die}],
        ability=ability,
        rider={"unarmed": True, "monk_weapon": True},
    ))
    dc = 8 + proficiency + modifier
    choice = ["strength", "dexterity"]
    scaled.append(AttackOption(
        name="Shove",
        resolution="save",
        reach_feet=5,
        dc_value=dc,
        dc_ability=ability,
        dc_success="none",
        ability=ability,
        rider={"shove": "push", "squares": 1, "save_choice": choice, "size_step": 1, "unarmed": True, "monk_weapon": True},
    ))
    scaled.append(AttackOption(
        name="Shove Prone",
        resolution="save",
        reach_feet=5,
        dc_value=dc,
        dc_ability=ability,
        dc_success="none",
        ability=ability,
        rider={"shove": "prone", "save_choice": choice, "size_step": 1, "unarmed": True, "monk_weapon": True},
    ))
    if _free_hands(character) >= 1:
        scaled.append(AttackOption(
            name="Grapple",
            resolution="save",
            reach_feet=5,
            dc_value=dc,
            dc_ability=ability,
            dc_success="none",
            ability=ability,
            rider={"grapple": True, "save_choice": choice, "size_step": 1, "escape_dc": dc, "unarmed": True, "monk_weapon": True},
        ))
    return scaled


def _dice_string(damage) -> str:
    if isinstance(damage, str):
        return damage.replace(" ", "")
    if not isinstance(damage, dict):
        return ""
    text = damage.get("diceString") or damage.get("dice")
    if text:
        return str(text).replace(" ", "")
    count = damage.get("diceCount")
    sides = damage.get("diceValue")
    if count and sides:
        return f"{int(count)}d{int(sides)}"
    return ""


def _property_names(definition: dict) -> set[str]:
    names = set()
    for prop in definition.get("properties") or []:
        if isinstance(prop, str):
            names.add(prop.lower())
        elif isinstance(prop, dict):
            label = str(prop.get("name") or prop.get("index") or "").lower()
            if label:
                names.add(label)
    return names


def _spells(character, catalog, abilities, proficiency):
    notes = []
    found: list[SpellRecord] = []
    seen = set()
    casting_id = _spellcasting_ability(character)
    casting = ability_mod(abilities[_STAT_NAMES[casting_id]]) if casting_id in _STAT_NAMES else 0
    attack_bonus = proficiency + casting + _modifier_sum(character, "spell-attacks") if casting_id else 0
    save_dc = 8 + proficiency + casting + _modifier_sum(character, "spell-save-dc") if casting_id else 0
    entries = _spell_entries(character)
    if not casting_id and entries:
        notes.append("Spell attack bonus was not in the JSON.")
    for name, prepared in entries:
        index = _spell_index(name)
        if not prepared:
            notes.append(f"{name}: on the sheet, not prepared")
            continue
        spell = catalog.spells.get(index) if catalog is not None else None
        if spell is None:
            notes.append(f"{name}: not in the local spell cache")
            continue
        if spell.kind == "excluded" or spell.exclude_reason:
            notes.append(f"{spell.name}: {spell.exclude_reason or 'the engine does not roll this spell'}")
            continue
        if index in seen:
            continue
        seen.add(index)
        found.append(SpellRecord.from_dict(spell.to_dict()))
    return found, notes, attack_bonus, save_dc, casting


def _spellcasting_ability(character: dict) -> int | None:
    for row in _classes(character):
        definition = row.get("definition") or {}
        ability = definition.get("spellCastingAbilityId")
        if ability:
            return int(ability)
    for row in _modifiers(character):
        if str(row.get("subType") or "") == "spellcasting-ability" and row.get("value"):
            return int(row["value"])
    # A feat grant names the ability chosen for that feature when the class has none.
    for _bucket, spell in _grant_spells(character):
        ability = spell.get("spellCastingAbilityId")
        if ability:
            return int(ability)
    return None


def _feature_grant(spell: dict) -> bool:
    """A class-bucket spell from a feature, not the prepared list.

    Optional class features and subclass choices store the spell on
    ``spells.class`` with the feature's definition id in ``componentId``.
    The prepared list lives in ``classSpells`` and leaves that id at 0.
    """
    try:
        return int(spell.get("componentId") or 0) != 0
    except (TypeError, ValueError):
        return bool(spell.get("componentId"))


def _grant_spells(character: dict):
    """Spells from a feat, species, background, item, or class feature."""
    container = character.get("spells") or {}
    if not isinstance(container, dict):
        return
    for bucket, value in container.items():
        if not isinstance(value, list):
            continue
        class_list = str(bucket).lower() == "class"
        for spell in value:
            if not isinstance(spell, dict):
                continue
            if class_list and not _feature_grant(spell):
                continue
            yield str(bucket), spell


def _granted_free_casts(character: dict, abilities: dict[str, int]) -> list[dict]:
    """Castings that do not spend a slot. A stat-modifier pool counts, as Bardic Inspiration does."""
    found = []
    for _bucket, spell in _grant_spells(character):
        if spell.get("usesSpellSlot") is not False:
            continue
        definition = spell.get("definition") if isinstance(spell.get("definition"), dict) else spell
        name = str(definition.get("name") or "")
        limited = spell.get("limitedUse") if isinstance(spell.get("limitedUse"), dict) else {}
        count = _use_count(limited, abilities)
        if not count or not name:
            continue
        try:
            spent = int(limited.get("numberUsed") or 0)
        except (TypeError, ValueError):
            spent = 0
        found.append({"name": name, "uses": count, "left": max(0, count - spent)})
    return found


def _spell_entries(character: dict) -> list[tuple[str, bool]]:
    rows = []
    container = character.get("spells") or {}
    if isinstance(container, dict):
        for bucket, value in container.items():
            if not isinstance(value, list):
                continue
            # The class bucket is the prepared list, except a row whose componentId
            # names a class feature. Every other bucket is a feature grant.
            class_list = str(bucket).lower() == "class"
            for spell in value:
                if not isinstance(spell, dict):
                    continue
                granted = (not class_list) or _feature_grant(spell)
                rows.extend(_named_spells([spell], known_style=False, granted=granted))
    classes = {str(row.get("id")): row for row in _classes(character) if row.get("id") is not None}
    for group in character.get("classSpells") or []:
        if not isinstance(group, dict) or not isinstance(group.get("spells"), list):
            continue
        owner = classes.get(str(group.get("characterClassId"))) or {}
        definition = owner.get("definition") if isinstance(owner.get("definition"), dict) else {}
        # A bard's chosen spells are countsAsKnownSpell with prepared false.
        # A class that prepares still requires the prepared flag on a leveled spell.
        # Its cantrips stay countsAsKnownSpell with prepared false.
        rows.extend(_named_spells(group["spells"], known_style=_known_style(definition) if owner else False))
    chosen: dict[str, tuple[str, bool]] = {}
    order: list[str] = []
    for name, prepared in rows:
        key = name.lower()
        if key not in chosen:
            order.append(key)
            chosen[key] = (name, prepared)
        elif prepared and not chosen[key][1]:
            # A feature grant wins when the class copy was left unprepared.
            chosen[key] = (chosen[key][0], True)
    return [chosen[key] for key in order]


def _named_spells(bucket: list, known_style: bool, *, granted: bool = False) -> list[tuple[str, bool]]:
    rows = []
    for spell in bucket:
        if not isinstance(spell, dict):
            continue
        definition = spell.get("definition") if isinstance(spell.get("definition"), dict) else spell
        name = definition.get("name")
        if not name:
            continue
        rows.append((str(name), _spell_available(spell, known_style, granted=granted)))
    return rows


def _known_style(definition: dict) -> bool:
    if definition.get("spellPrepareType"):
        return False
    rules = definition.get("spellRules") if isinstance(definition.get("spellRules"), dict) else {}
    return not (rules.get("levelPreparedSpellMaxes") or [])


def _known_cantrip(spell: dict) -> bool:
    """A preparing class still knows its cantrips. They are stored unprepared."""
    if not spell.get("countsAsKnownSpell"):
        return False
    definition = spell.get("definition") if isinstance(spell.get("definition"), dict) else {}
    raw = definition.get("level")
    if raw is None:
        raw = spell.get("level")
    try:
        return int(raw) == 0
    except (TypeError, ValueError):
        return False


def _spell_available(spell: dict, known_style: bool, *, granted: bool = False) -> bool:
    if granted:
        return True
    prepared = spell.get("prepared")
    always = bool(spell.get("alwaysPrepared"))
    if known_style:
        return prepared is True or always or bool(spell.get("countsAsKnownSpell"))
    if prepared is not False or always:
        return True
    return _known_cantrip(spell)


def _spell_index(name: str) -> str:
    text = name.lower().replace("'", "").replace("’", "")
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-")


def _slots(character: dict) -> dict[int, int]:
    listed = _counted_slots(_slot_rows(character.get("spellSlots")) + _slot_rows(character.get("pactMagic")))
    if listed:
        return listed
    if not _spell_slots_unset(character):
        return {}
    return _class_table_slots(character)


def _slot_rows(raw) -> list[dict]:
    rows = []
    if isinstance(raw, dict):
        for level, value in raw.items():
            if isinstance(value, dict):
                rows.append({"level": level, **value})
            else:
                rows.append({"level": level, "available": value})
    elif isinstance(raw, list):
        rows.extend(item for item in raw if isinstance(item, dict))
    return rows


def _counted_slots(rows: list[dict]) -> dict[int, int]:
    slots: dict[int, int] = {}
    for row in rows:
        try:
            level = int(row.get("level") or 0)
        except (TypeError, ValueError):
            continue
        count = row.get("available", row.get("slots", row.get("max")))
        if level > 0 and count:
            slots[level] = slots.get(level, 0) + int(count)
    return slots


def _spell_slots_unset(character: dict) -> bool:
    for row in _slot_rows(character.get("spellSlots")):
        try:
            used = int(row.get("used") or 0)
            available = int(row.get("available", row.get("slots", row.get("max"))) or 0)
        except (TypeError, ValueError):
            return False
        if used or available:
            return False
    return True


def _class_table_slots(character: dict) -> dict[int, int]:
    casters = []
    for row in _classes(character):
        definition = row.get("definition") if isinstance(row.get("definition"), dict) else {}
        rules = definition.get("spellRules") if isinstance(definition.get("spellRules"), dict) else {}
        table = rules.get("levelSpellSlots")
        if isinstance(table, list) and table:
            casters.append((row, table))
    if len(casters) != 1:
        return {}
    row, table = casters[0]
    try:
        level = int(row.get("level") or 0)
    except (TypeError, ValueError):
        return {}
    if level < 0 or level >= len(table) or not isinstance(table[level], list):
        return {}
    slots: dict[int, int] = {}
    for index, count in enumerate(table[level], start=1):
        try:
            number = int(count or 0)
        except (TypeError, ValueError):
            continue
        if number > 0:
            slots[index] = number
    return slots


_QUIET_FEATURES = {
    "extra attack",
    "rage",
    "unarmored defense",
    "weapon mastery",
    "danger sense",
    "reckless attack",
    "fast movement",
    "ability score improvement",
    "divine fury",
    "warrior of the gods",
    "barbarian subclass",
    "core barbarian traits",
    "great weapon master",
    "tough",
    "hew",
    "heavy weapon mastery",
    "cutting words",
} | set(QUIET)


def _feature_records(character: dict) -> list[tuple[str, str, int]]:
    records = []
    for row in _classes(character):
        level = int(row.get("level") or 0)
        sources = list(row.get("classFeatures") or [])
        subclass = row.get("subclassDefinition") or {}
        if isinstance(subclass, dict):
            sources.extend(subclass.get("classFeatures") or [])
        for feature in sources:
            if not isinstance(feature, dict):
                continue
            definition = feature.get("definition") if isinstance(feature.get("definition"), dict) else feature
            required = int(definition.get("requiredLevel") or feature.get("requiredLevel") or 0)
            name = str(definition.get("name") or feature.get("name") or "")
            if not name or required > level:
                continue
            identifier = definition.get("id", feature.get("id"))
            records.append((name, "" if identifier is None else str(identifier), level))
    return records


def _plain(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text or "")).strip()


def _sheet_feature(name: str) -> bool:
    lowered = name.lower()
    if lowered in _QUIET_FEATURES or "weapon mastery" in lowered or "ability score" in lowered:
        return False
    return True


def _action_rows(character: dict):
    groups = character.get("actions") or {}
    if not isinstance(groups, dict):
        return
    for rows in groups.values():
        if not isinstance(rows, list):
            continue
        for action in rows:
            if isinstance(action, dict):
                yield action


def _rage_bonus(level: int) -> int:
    if level >= 16:
        return 4
    if level >= 9:
        return 3
    return 2


def _self_heal(text: str) -> bool:
    lowered = text.lower()
    points = "hit point" in lowered or re.search(r"\bhp\b", lowered) is not None
    return "heal" in lowered or "regain" in lowered and points


_POOL_NAME = re.compile(
    r"expend (?:one|a) use of ([A-Za-z][^.;,]*?)(?:\s+and\b|[.;,]|$)",
    re.I,
)
_EXTRA_REACTION = re.compile(
    r"\b(teleports?|splits? into|swap places|redirects?|frightened|blinded|restrained|incapacitated)\b",
    re.I,
)


def _sheet_subtract(name: str, die: str, snippet: str, action: dict) -> ReactionRecord | None:
    """A reaction action that only subtracts its die from an attack or damage roll."""
    text = snippet or ""
    if not die or not re.search(r"\bsubtract\b", text, re.I):
        return None
    if not re.search(r"attack roll|damage roll", text, re.I):
        return None
    if _EXTRA_REACTION.search(text) or re.search(r"\badds \d+ to\b|\bcasts?\b", text, re.I):
        return None
    found = _POOL_NAME.search(text)
    pool = found.group(1).strip() if found else name
    span = action.get("range") if isinstance(action.get("range"), dict) else {}
    try:
        reach = int(span.get("range") or 60)
    except (TypeError, ValueError):
        reach = 60
    return ReactionRecord(
        name=name or "Reaction",
        effect="subtract",
        triggers=["hit", "damage"],
        die=die,
        range_feet=reach or 60,
        pool=pool,
        text=text,
    )


def _use_count(limited: dict, abilities: dict[str, int]) -> int | None:
    """A listed maximum, or a stat-modifier pool such as Bardic Inspiration."""
    if not limited:
        return None
    try:
        count = int(limited.get("maxUses") or 0)
    except (TypeError, ValueError):
        count = 0
    if count > 0:
        return count
    try:
        stat_id = int(limited.get("statModifierUsesId"))
    except (TypeError, ValueError):
        return None
    name = _STAT_NAMES.get(stat_id)
    if name is None:
        return None
    return max(1, ability_mod(abilities.get(name, 10)))


def _any_mastery(character: dict) -> bool:
    for item in _inventory(character):
        if not item.get("equipped"):
            continue
        definition = item.get("definition") or {}
        if _item_kind(definition) != "weapon":
            continue
        if any(name in _property_names(definition) for name in _MASTERY):
            return True
    return False


def _chosen_damage_type(character: dict) -> str:
    for choice in _choice_rows(character):
        label = feat_key(_option_label(character, choice.get("optionValue")) or str(choice.get("label") or ""))
        if label in DAMAGE_TYPES:
            return label
    return ""


def _apply_sheet_feats(character, traits, uses_max, uses_left, abilities, notes) -> None:
    """Turn feat and fighting-style names into traits. Scores and hit points stay as the sheet stored them."""
    names = _feat_names(character)
    gear = _gear(character)
    traits["has_shield"] = gear["has_shield"]
    if gear["armor"]:
        traits["armor"] = gear["armor"]
    if gear["one_melee"]:
        traits["one_melee"] = True
    level = sum(int(item.get("level") or 0) for item in _classes(character))
    proficiency = math.ceil(max(level, 1) / 4) + 1
    if "great weapon master" in names:
        traits["gwm"] = True
        traits["hew"] = True
    flags = {
        "alert": "alert",
        "savage attacker": "savage_attacker",
        "healer": "healer",
        "archery": "archery",
        "dueling": "dueling",
        "great weapon fighting": "great_weapon_fighting",
        "two weapon fighting": "two_weapon_fighting",
        "crossbow expert": "crossbow_expert",
        "thrown weapon fighting": "thrown_weapon_fighting",
        "interception": "interception",
        "protection": "protection",
        "charger": "charger",
        "crusher": "crusher",
        "piercer": "piercer",
        "slasher": "slasher",
        "defensive duelist": "defensive_duelist",
        "dual wielder": "dual_wielder",
        "heavy armor master": "heavy_armor_master",
        "mage slayer": "mage_slayer",
        "poisoner": "poisoner",
        "polearm master": "polearm_master",
        "sentinel": "sentinel",
        "shield master": "shield_master",
        "spell sniper": "spell_sniper",
        "war caster": "war_caster",
        "speedy": "speedy",
        "tavern brawler": "tavern_brawler",
        "unarmed fighting": "unarmed_fighting",
        "lucky": "lucky",
    }
    for name, key in flags.items():
        if name in names:
            traits[key] = True
    if traits.get("alert"):
        traits["initiative_bonus"] = proficiency
    if traits.get("lucky") and "Luck" not in uses_max:
        uses_max["Luck"] = proficiency
        uses_left["Luck"] = proficiency
    if traits.get("mage_slayer") and "Guarded Mind" not in uses_max:
        uses_max["Guarded Mind"] = 1
        uses_left["Guarded Mind"] = 1
    if "elemental adept" in names:
        chosen = _chosen_damage_type(character)
        if chosen:
            traits["elemental_adept"] = chosen
        else:
            notes.append("Elemental Adept is on the sheet, and the damage type was not named.")
    if traits.get("savage_attacker"):
        notes.append("Savage Attacker rolls the weapon's damage dice twice on the first hit each turn and keeps the higher.")
    if traits.get("interception"):
        for action in _action_rows(character):
            if feat_key(str(action.get("name") or "")) != "interception":
                continue
            dice = action.get("dice") if isinstance(action.get("dice"), dict) else {}
            die = str(dice.get("diceString") or "")
            if die:
                traits["interception_die"] = die
                break
    if traits.get("telekinetic") or "telekinetic" in names:
        ability = _telekinetic_ability(character, abilities)
        traits["telekinetic"] = _telekinetic_dc(character, abilities, proficiency)
        traits["telekinetic_ability"] = ability
        notes.append(f"Telekinetic Shove uses {ability.title()} (DC {traits['telekinetic']}).")
    if traits.get("unarmed_fighting"):
        traits["unarmed_reroll"] = True
    if any(token in names for token in ("magic initiate", "fey touched", "shadow touched")):
        free = []
        owned_ids = _owned_ids(character)
        for action in _action_rows(character):
            if str(action.get("componentId") or "") not in owned_ids:
                continue
            action_name = str(action.get("name") or "")
            if feat_key(action_name) in {"magic initiate", "fey touched", "shadow touched"}:
                continue
            limited = action.get("limitedUse") if isinstance(action.get("limitedUse"), dict) else {}
            try:
                count = int(limited.get("maxUses") or 0)
            except (TypeError, ValueError):
                count = 0
            activation = action.get("activation") if isinstance(action.get("activation"), dict) else {}
            try:
                activation_type = int(activation.get("activationType") or 0)
            except (TypeError, ValueError):
                activation_type = 0
            if count > 0 and action_name and activation_type == 1:
                free.append({"name": action_name, "uses": count})
        if free:
            traits["free_cast_names"] = free


def _telekinetic_ability(character, abilities) -> str:
    casting = _spellcasting_ability(character)
    if casting in _STAT_NAMES:
        return _STAT_NAMES[casting]
    best = "intelligence"
    score = -1
    for name in ("intelligence", "wisdom", "charisma"):
        value = int(abilities.get(name, 10))
        if value > score:
            score = value
            best = name
    return best


def _telekinetic_dc(character, abilities, proficiency: int) -> int:
    casting = _spellcasting_ability(character)
    if casting in _STAT_NAMES:
        modifier = ability_mod(abilities.get(_STAT_NAMES[casting], 10))
    else:
        modifier = max(ability_mod(abilities.get(name, 10)) for name in ("intelligence", "wisdom", "charisma"))
    return 8 + proficiency + modifier


def _owned_ids(character: dict) -> set[str]:
    owned = set()
    for _name, identifier, _level in _feature_records(character):
        if identifier:
            owned.add(identifier)
    for feat in character.get("feats") or []:
        if not isinstance(feat, dict):
            continue
        definition = feat.get("definition") if isinstance(feat.get("definition"), dict) else feat
        if definition.get("id") is not None:
            owned.add(str(definition.get("id")))
    return owned


def _resolve_name(traits: dict, *names: str) -> None:
    found = traits.setdefault("resolved_names", [])
    for name in names:
        if name and name not in found:
            found.append(name)


def _monk_state(traits: dict, character: dict, abilities: dict[str, int], proficiency: int) -> dict:
    current = traits.get("monk")
    if isinstance(current, dict):
        return current
    level = _class_level(character, "monk")
    dexterity = ability_mod(abilities.get("dexterity", 10))
    strength = ability_mod(abilities.get("strength", 10))
    wisdom = ability_mod(abilities.get("wisdom", 10))
    monk = {
        "level": level,
        "die": _martial_sides(level) if level else 6,
        "ability": "dexterity" if dexterity >= strength else "strength",
        "qualifies": _monk_qualifies(character) and level > 0,
        "save_dc": 8 + proficiency + wisdom,
        "bonus_unarmed": bool(level),
    }
    traits["monk"] = monk
    return monk


def _focus_pool_name(name: str) -> str:
    key = feat_key(name)
    if key in {"focus points", "ki", "ki points"}:
        return name
    return ""


def _signed_die(dice: dict, bonus: int) -> str:
    try:
        count = int(dice.get("diceCount") or 1)
        sides = int(dice.get("diceValue") or 0)
    except (TypeError, ValueError):
        return ""
    if sides <= 0:
        return ""
    base = f"{count}d{sides}"
    return f"{base}{bonus:+d}" if bonus else base


def _absorb_monk_action(character, action, name, snippet, die, maximum, spent, activation_type, abilities, proficiency, traits, uses_max, uses_left, reactions) -> bool:
    """Read one monk action off the sheet. True means the generic loop should skip it."""
    key = feat_key(name)
    text = snippet.lower()
    if _focus_pool_name(name) and maximum:
        count = int(maximum)
        uses_max[name] = count
        uses_left[name] = max(0, count - spent)
        monk = _monk_state(traits, character, abilities, proficiency)
        monk["focus"] = name
        _resolve_name(traits, name, "Monk's Focus", "Ki")
        return True
    if key == "flurry of blows":
        monk = _monk_state(traits, character, abilities, proficiency)
        monk["flurry"] = True
        _resolve_name(traits, name)
        return True
    if key == "patient defense":
        monk = _monk_state(traits, character, abilities, proficiency)
        monk["patient"] = True
        monk["patient_focused"] = "expend" in text
        _resolve_name(traits, name)
        return True
    if key == "step of the wind":
        monk = _monk_state(traits, character, abilities, proficiency)
        monk["step"] = True
        monk["step_focused"] = "expend" in text
        _resolve_name(traits, name)
        return True
    if key == "stunning strike":
        monk = _monk_state(traits, character, abilities, proficiency)
        monk["stunning"] = True
        _resolve_name(traits, name)
        return True
    if key.startswith("flurry of blows"):
        technique = key.split(":", 1)[1].strip() if ":" in key else ""
        if technique in {"addle", "push", "topple"}:
            monk = _monk_state(traits, character, abilities, proficiency)
            techniques = list(monk.get("open_hand") or [])
            if technique not in techniques:
                techniques.append(technique)
            monk["open_hand"] = techniques
            _resolve_name(traits, name, "Open Hand Technique")
            return True
    if key == "uncanny metabolism" and maximum:
        count = int(maximum)
        uses_max[name] = count
        uses_left[name] = max(0, count - spent)
        monk = _monk_state(traits, character, abilities, proficiency)
        monk["metabolism"] = {"name": name, "die": die or "1d8"}
        _resolve_name(traits, name)
        return True
    if key == "deflect attack" and activation_type == 4:
        dice = action.get("dice") if isinstance(action.get("dice"), dict) else {}
        try:
            fixed = int(dice.get("fixedValue"))
        except (TypeError, ValueError):
            fixed = int(_monk_state(traits, character, abilities, proficiency).get("level") or 0)
        bonus = ability_mod(abilities.get("dexterity", 10)) + fixed
        expression = _signed_die(dice, bonus) or f"1d10{bonus:+d}"
        monk = _monk_state(traits, character, abilities, proficiency)
        monk["deflect"] = {"die": f"1d{int(dice.get('diceValue') or 10)}", "bonus": bonus, "expr": expression}
        reactions.append(ReactionRecord(
            name=name or "Deflect Attack",
            effect="deflect",
            triggers=["damage"],
            die=expression,
            text=snippet,
        ))
        _resolve_name(traits, name, "Deflect Attacks", "Deflect Attack")
        return True
    if "redirect" in key and "deflect" in key:
        dice = action.get("dice") if isinstance(action.get("dice"), dict) else {}
        bonus = ability_mod(abilities.get("dexterity", 10))
        expression = _signed_die(dice, bonus) or f"2d8{bonus:+d}"
        spans = [int(item) for item in re.findall(r"within (\d+) ft", snippet, re.I)]
        melee = spans[0] if spans else 5
        ranged = spans[1] if len(spans) > 1 else (spans[0] if spans else 60)
        monk = _monk_state(traits, character, abilities, proficiency)
        monk["redirect"] = {"expr": expression, "melee": melee, "ranged": ranged}
        _resolve_name(traits, name, "Deflect Attack: Redirect Attack")
        return True
    if key == "slow fall":
        return False
    return False


def _combat_profile(character: dict, proficiency: int) -> tuple[dict, dict, dict, int, list[str], list[ReactionRecord]]:
    records = _feature_records(character)
    labels = []
    owned = set()
    owner_level = {}
    for name, identifier, level in records:
        if identifier:
            owned.add(identifier)
        key = name.lower()
        owner_level.setdefault(key, level)
        if name not in labels:
            labels.append(name)
    for feat in character.get("feats") or []:
        if not isinstance(feat, dict):
            continue
        definition = feat.get("definition") if isinstance(feat.get("definition"), dict) else feat
        name = str(definition.get("name") or "")
        identifier = definition.get("id")
        if identifier is not None:
            owned.add(str(identifier))
        if name and name not in labels:
            labels.append(name)
    traits: dict = {}
    uses_max: dict[str, int] = {}
    uses_left: dict[str, int] = {}
    reactions: list[ReactionRecord] = []
    limited_rows: list[tuple[str, dict, int]] = []
    abilities = _abilities(character)
    lowered = {name.lower() for name in labels}
    if "reckless attack" in lowered:
        traits["reckless_feature"] = True
    if "danger sense" in lowered:
        traits["danger_sense"] = True
    if "divine fury" in lowered:
        level = int(owner_level.get("divine fury") or 1)
        traits["divine_fury"] = {"die": "1d6", "flat": level // 2, "type": "radiant"}
    if "great weapon master" in lowered:
        traits["gwm"] = True
    extra_row = _granted_row(character, "set", "extra-attacks")
    attacks = 1
    if extra_row is not None and extra_row.get("value"):
        attacks = 1 + int(extra_row["value"])
    elif "extra attack" in lowered:
        attacks = 2
    for action in _action_rows(character):
        if str(action.get("componentId") or "") not in owned:
            continue
        name = str(action.get("name") or "")
        snippet = _plain(str(action.get("snippet") or ""))
        limited = action.get("limitedUse") if isinstance(action.get("limitedUse"), dict) else {}
        maximum = limited.get("maxUses")
        spent = int(limited.get("numberUsed") or 0)
        activation = action.get("activation") if isinstance(action.get("activation"), dict) else {}
        dice = action.get("dice") if isinstance(action.get("dice"), dict) else {}
        die = str(dice.get("diceString") or "")
        if name and limited:
            limited_rows.append((name, limited, spent))
        try:
            activation_type = int(activation.get("activationType") or 0)
        except (TypeError, ValueError):
            activation_type = 0
        if _absorb_monk_action(
            character, action, name, snippet, die, maximum, spent, activation_type,
            abilities, proficiency, traits, uses_max, uses_left, reactions,
        ):
            continue
        if name.lower().startswith("rage") and maximum:
            count = int(maximum)
            uses_max["Rage"] = count
            uses_left["Rage"] = max(0, count - spent)
            traits["rage_bonus"] = _rage_bonus(int(owner_level.get("rage") or 1))
            continue
        if name.lower() == "hew":
            traits["hew"] = True
            continue
        if int(activation.get("activationType") or 0) == 4:
            record = _sheet_subtract(name, die, snippet, action)
            if record is None and name.lower() == "cutting words":
                span = action.get("range") if isinstance(action.get("range"), dict) else {}
                try:
                    reach = int(span.get("range") or 60)
                except (TypeError, ValueError):
                    reach = 60
                record = ReactionRecord(
                    name=name or "Cutting Words",
                    effect="subtract",
                    triggers=["hit", "damage"],
                    die=die or "1d8",
                    range_feet=reach or 60,
                    pool="Bardic Inspiration",
                    text=snippet,
                )
            if record is not None:
                reactions.append(record)
                if name.lower() == "cutting words":
                    traits["cutting_words"] = {
                        "die": record.die or "1d8",
                        "range": int(record.range_feet or 60),
                        "pool": record.pool or "Bardic Inspiration",
                    }
                continue
            reactions.append(ReactionRecord(name=name or "Reaction", effect="unresolved", text=snippet))
            continue
        if name.lower() == "bardic inspiration":
            if "Bardic Inspiration" not in uses_max:
                count = _use_count(limited, abilities)
                if count:
                    uses_max["Bardic Inspiration"] = count
                    uses_left["Bardic Inspiration"] = max(0, count - spent)
            continue
        if "{{proficiency}}" in snippet and "heavy" in snippet.lower():
            traits["gwm"] = True
        if name.lower() == "action surge" and maximum:
            count = int(maximum)
            uses_max["Action Surge"] = count
            uses_left["Action Surge"] = max(0, count - spent)
            continue
        if "tactical shift" in name.lower():
            traits["tactical_shift"] = True
            continue
        if int(activation.get("activationType") or 0) == 3 and die and maximum and _self_heal(snippet):
            heal_name = name.split(":", 1)[0].strip() or name
            if any(item.get("name") == heal_name for item in traits.get("bonus_heals") or []):
                continue
            count = int(maximum)
            uses_max[heal_name] = count
            uses_left[heal_name] = max(0, count - spent)
            traits.setdefault("bonus_heals", []).append({"name": heal_name, "die": die})
    for record in reactions:
        pool = record.pool
        if not pool or pool in uses_max or record.effect == "unresolved":
            continue
        for row_name, limited, spent in limited_rows:
            if row_name.lower() != pool.lower():
                continue
            count = _use_count(limited, abilities)
            if count:
                record.pool = row_name
                uses_max[row_name] = count
                uses_left[row_name] = max(0, count - spent)
            break
    feature = traits.get("cutting_words")
    if isinstance(feature, dict):
        for record in reactions:
            if record.name.lower() == "cutting words" and record.effect == "subtract":
                feature["die"] = record.die or feature.get("die") or "1d8"
                feature["range"] = int(record.range_feet or feature.get("range") or 60)
                feature["pool"] = record.pool or feature.get("pool") or "Bardic Inspiration"
    for record in reactions:
        if record.effect == "unresolved" and record.name and record.name not in labels and _sheet_feature(record.name):
            labels.append(record.name)
    notes = []
    if traits.get("divine_fury"):
        fury = traits["divine_fury"]
        notes.append(
            f"Divine Fury adds {fury['die']}+{fury['flat']} {fury['type']} on the first hit while raging. "
            "The sheet can choose necrotic or radiant on each hit; this fight uses radiant."
        )
    if traits.get("gwm"):
        notes.append("A Heavy weapon hit on the Attack action adds the proficiency bonus as damage.")
    if traits.get("hew"):
        notes.append("Hew offers one bonus attack after a melee critical hit or a melee hit that drops a creature to 0.")
    _apply_sheet_feats(character, traits, uses_max, uses_left, abilities, notes)
    if _class_level(character, "monk") and "martial arts" in _feat_names(character):
        _resolve_name(traits, "Martial Arts")
        if _monk_qualifies(character):
            _resolve_name(traits, "Unarmored Movement")
    traits["sheet_notes"] = [note for note in notes]
    known_feats = _feat_names(character)
    style_chosen = any(name in STYLES for name in known_feats)
    feature_notes = []
    resolved = {feat_key(name) for name in traits.get("resolved_names") or []}
    for name in labels:
        if not _sheet_feature(name):
            continue
        if feat_key(name) in resolved:
            continue
        if feat_key(name) == "fighting style" and style_chosen:
            continue
        if feat_key(name) == "weapon master" and _any_mastery(character):
            continue
        if feat_key(name) == "elemental adept" and traits.get("elemental_adept"):
            continue
        feature_notes.append(name)
    return traits, uses_max, uses_left, attacks, feature_notes, reactions
