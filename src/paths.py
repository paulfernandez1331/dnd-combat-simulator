"""Project directories."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = DATA / "cache"
RUNS = DATA / "runs"
IMPORTS = DATA / "imports"
SAMPLES = DATA / "samples"
FIXTURES = ROOT / "tests" / "fixtures"
