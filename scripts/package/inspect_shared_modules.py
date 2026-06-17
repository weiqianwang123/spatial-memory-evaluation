from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatial_memory_evaluation.shared_modules import get_shared_module_registry


METHODS = ("hovsg", "dualmap", "conceptgraphs", "daaam")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect shared module registry entries.")
    parser.add_argument("--method", choices=METHODS, default=None)
    parser.add_argument("--profile", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--check", action="store_true", help="fail if required modules are unavailable")
    return parser.parse_args()


def main(args: argparse.Namespace) -> int:
    registry = get_shared_module_registry()
    if args.method is None:
        payload = {
            "profiles": {
                profile: {
                    method: registry.method_metadata(method, profile) for method in METHODS
                }
                for profile in ("smoke", "formal")
            }
        }
        missing: list[str] = []
    else:
        payload = registry.method_metadata(args.method, args.profile)
        missing = registry.check_method(args.method, args.profile)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.check and missing:
        print(f"missing required shared modules: {', '.join(missing)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
