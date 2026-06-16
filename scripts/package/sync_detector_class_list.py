#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatial_memory_evaluation.common.labels import (
    DEFAULT_DETECTOR_CLASS_LIST_PATH,
    detector_class_list_mismatch,
    write_default_detector_class_list,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync or check the shared detector class list.")
    parser.add_argument("--path", type=Path, default=DEFAULT_DETECTOR_CLASS_LIST_PATH)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.check:
        mismatch = detector_class_list_mismatch(args.path)
        print(json.dumps(mismatch, indent=2, sort_keys=True))
        return 0 if not mismatch["missing"] and not mismatch["extra"] and mismatch["order_matches"] else 1
    write_default_detector_class_list(args.path)
    print(args.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
