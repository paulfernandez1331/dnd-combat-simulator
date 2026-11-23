"""Download the SRD monster and spell catalog into data/cache.

The script skips files that are already on disk, so it is safe to rerun.
"""

from src.api_client import refresh_catalog


def main() -> None:
    def report(kind: str, position: int, total: int, index: str) -> None:
        if position == 1 or position == total or position % 25 == 0:
            print(f"{kind}: {position}/{total} ({index})", flush=True)

    counts = refresh_catalog(on_progress=report)
    print(
        f"Done. New monsters: {counts['monsters']}. New spells: {counts['spells']}. "
        f"Already cached: {counts['skipped']}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
