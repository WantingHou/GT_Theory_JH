#!/usr/bin/env python3
"""Generate JH Figures 2 through 7 from the packaged theoretical states."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jh_theory import build_all


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    for path in build_all(args.output):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
