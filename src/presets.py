"""Built-in monster layouts. The party is whatever public sheets are in data/samples."""

from __future__ import annotations

from src.api_client import Catalog, CatalogError
from src.combatant import from_monster
from src.ddb import public_character, sample_characters
from src.encounter import Encounter


def _party(catalog: Catalog) -> list:
    rows = sample_characters()
    if not rows:
        raise CatalogError("Put a public character JSON file in data/samples before loading a preset.")
    members = []
    for index, (stem, character) in enumerate(rows):
        creature_id = f"ddb-{int(character.get('id') or stem)}"
        members.append(public_character(character, catalog, (2, 5 + index * 2), creature_id))
    return members


def _monster(catalog: Catalog, index: str, creature_id: str, position: tuple[int, int]):
    record = catalog.monsters.get(index)
    if record is None:
        raise CatalogError(f"{index} is not in the local cache. Download the SRD catalog from the Bestiary page.")
    number = creature_id.rsplit("-", 1)[-1]
    return from_monster(record, creature_id=creature_id, name=f"{record.name} {number}", side="enemy", position=position)


PLAY_WIDTH = 20
PLAY_HEIGHT = 20


def four_goblins(catalog: Catalog, seed: int = 7) -> Encounter:
    combatants = [
        *_party(catalog),
        _monster(catalog, "goblin-warrior", "goblin-warrior-1", (8, 4)),
        _monster(catalog, "goblin-warrior", "goblin-warrior-2", (8, 6)),
        _monster(catalog, "goblin-warrior", "goblin-warrior-3", (8, 8)),
        _monster(catalog, "goblin-warrior", "goblin-warrior-4", (9, 5)),
    ]
    return Encounter(combatants, width=PLAY_WIDTH, height=PLAY_HEIGHT, seed=seed, label="Four goblins", enemy_policy="llm")


def bugbear_patrol(catalog: Catalog, seed: int = 11) -> Encounter:
    combatants = [
        *_party(catalog),
        _monster(catalog, "bugbear-warrior", "bugbear-warrior-1", (8, 5)),
        _monster(catalog, "hobgoblin-warrior", "hobgoblin-warrior-1", (9, 4)),
        _monster(catalog, "hobgoblin-warrior", "hobgoblin-warrior-2", (9, 7)),
    ]
    return Encounter(combatants, width=PLAY_WIDTH, height=PLAY_HEIGHT, seed=seed, label="Bugbear patrol", enemy_policy="llm")
