"""Fight the two demo layouts with the heuristic and print the analysis brief.

Each layout is written once, under data/runs/. That folder is gitignored.
Run again only after deleting a layout folder; fights are appended, not replaced.
"""

from src.api_client import load_catalog
from src.bestiary import coverage
from src.eda import load_runs, report_brief
from src.paths import RUNS
from src.presets import bugbear_patrol, four_goblins
from src.simulate import run_batch


def main() -> None:
    catalog = load_catalog()
    jobs = (
        ("four-goblins-heuristic", four_goblins(catalog), 7),
        ("bugbear-patrol-heuristic", bugbear_patrol(catalog), 11),
    )
    for name, encounter, seed in jobs:
        run_dir = RUNS / name
        if (run_dir / "fights.csv").exists():
            print(f"{name}: already saved, left as-is")
            continue
        setup = encounter.to_setup()
        setup["enemy_policy"] = "heuristic"
        result = run_batch(setup, count=30, policy="heuristic", seed=seed, run_dir=run_dir)
        print(f"{name}: {result['completed']} fights")
    fights, turns = load_runs(RUNS)
    names = [name for name, _encounter, _seed in jobs]
    kept = fights[fights["run_id"].isin(names)]
    kept_turns = turns[turns["fight_id"].isin(set(kept["fight_id"]))] if not turns.empty and "fight_id" in turns.columns else turns
    report = coverage(list(catalog.spells.values()), list(catalog.monsters.values()))
    print(report_brief(kept, report, kept_turns))


if __name__ == "__main__":
    main()
