"""Fetch the 5e SRD API and read it back from disk.

The live service is ``https://www.dnd5eapi.co/api/2024``. A missing network
falls through to the cache. The app does not call the API during combat.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

from src.bestiary import MonsterRecord, SpellRecord, classify_monster, classify_spell
from src.paths import CACHE

API_ROOT = "https://www.dnd5eapi.co/api/2024"
USER_AGENT = "dnd-encounter-simulator/1.0 (class project)"


class CatalogError(RuntimeError):
    """The local cache cannot answer the request."""


class Catalog:
    def __init__(self, monsters: dict[str, dict], spells: dict[str, dict]):
        self.monster_raw = monsters
        self.spell_raw = spells
        self.monsters: dict[str, MonsterRecord] = {
            index: classify_monster(raw) for index, raw in monsters.items()
        }
        self.spells: dict[str, SpellRecord] = {
            index: classify_spell(raw) for index, raw in spells.items()
        }

    def monster_names(self) -> list[tuple[str, str]]:
        return sorted(((record.name, index) for index, record in self.monsters.items()), key=lambda item: item[0])

    def spell_names(self, *, resolvable_only: bool = False) -> list[tuple[str, str]]:
        rows = []
        for index, spell in self.spells.items():
            if resolvable_only and spell.kind == "excluded":
                continue
            rows.append((spell.name, index))
        return sorted(rows, key=lambda item: item[0])


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def fetch_json(url: str, timeout: float = 30) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise CatalogError(f"Could not reach {url}: {exc.reason}") from exc


def _load_dir(directory: Path) -> dict[str, dict]:
    records = {}
    if not directory.exists():
        return records
    for path in sorted(directory.glob("*.json")):
        if path.stem == "index":
            continue
        payload = _read_json(path)
        index = payload.get("index") or path.stem
        records[index] = payload
    return records


def load_catalog(cache_dir: Path | None = None) -> Catalog:
    root = cache_dir or CACHE
    monsters = _load_dir(root / "monsters")
    spells = _load_dir(root / "spells")
    if not monsters or not spells:
        raise CatalogError(
            "The local SRD cache is empty. Open the Bestiary page and download the catalog, "
            "or run python -m scripts.refresh_cache."
        )
    return Catalog(monsters, spells)


def cache_record(kind: str, index: str, cache_dir: Path | None = None, fetcher=fetch_json) -> dict:
    root = cache_dir or CACHE
    path = root / kind / f"{index}.json"
    if path.exists():
        return _read_json(path)
    payload = fetcher(f"{API_ROOT}/{kind}/{index}")
    _write_json(path, payload)
    return payload


def refresh_catalog(cache_dir: Path | None = None, fetcher=fetch_json, on_progress=None) -> dict:
    """Download any missing monster and spell records. Existing files stay."""
    root = cache_dir or CACHE
    counts = {"monsters": 0, "spells": 0, "skipped": 0}
    for kind in ("monsters", "spells"):
        index_payload = fetcher(f"{API_ROOT}/{kind}")
        _write_json(root / f"{kind}_index.json", index_payload)
        results = index_payload.get("results") or []
        for position, item in enumerate(results, start=1):
            index = item["index"]
            path = root / kind / f"{index}.json"
            if path.exists():
                counts["skipped"] += 1
            else:
                payload = fetcher(f"{API_ROOT}/{kind}/{index}")
                _write_json(path, payload)
                counts[kind] += 1
            if on_progress:
                on_progress(kind, position, len(results), index)
    return counts
