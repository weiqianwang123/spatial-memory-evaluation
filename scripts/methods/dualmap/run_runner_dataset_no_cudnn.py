from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    import torch

    torch.backends.cudnn.enabled = False
    print("[dualmap wrapper] torch.backends.cudnn.enabled=False", file=sys.stderr)

    script_path = Path("applications/runner_dataset.py").resolve()
    if not script_path.exists():
        raise FileNotFoundError(
            "DualMap applications/runner_dataset.py was not found. Run this "
            "launcher with cwd set to the DualMap repo root."
        )
    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()
