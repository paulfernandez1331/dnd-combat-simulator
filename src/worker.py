"""Run a batch outside the Streamlit process so the page can show a stop button."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.ai import LLMError
from src.paths import ROOT
from src.simulate import run_batch


def main() -> int:
    parser = argparse.ArgumentParser(description="Simulate an encounter and write CSV logs.")
    parser.add_argument("--setup", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--policy", choices=("llm", "heuristic"), default="llm")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    setup = json.loads(Path(args.setup).read_text(encoding="utf-8"))
    run_dir = Path(args.run_dir)
    try:
        result = run_batch(
            setup,
            count=args.count,
            policy=args.policy,
            seed=args.seed,
            run_dir=run_dir,
            should_stop=lambda: (run_dir / "STOP").exists(),
        )
    except LLMError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"Completed {result['completed']} fights.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
