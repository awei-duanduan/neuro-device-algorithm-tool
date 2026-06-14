#!/usr/bin/env python
from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> int:
    here = Path(__file__).resolve()
    project = here.parents[4]
    target = project / "tools" / "check_env.py"
    if not target.exists():
        print(f"Missing project checker: {target}", file=sys.stderr)
        return 1
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
