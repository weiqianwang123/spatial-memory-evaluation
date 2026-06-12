from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable, List, Sequence


BASE_URL = "http://kaldir.vc.in.tum.de/scannet/"
TOS_URL = BASE_URL + "ScanNet_TOS.pdf"
RELEASE = "v2/scans"
V1_RELEASE = "v1/scans"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download only the ScanNet .sens files required by the OpenEQA "
            "scannet-v0 split."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/open-eqa-v0.json"),
        help="OpenEQA dataset JSON used when --scenes-file is absent.",
    )
    parser.add_argument(
        "--scenes-file",
        type=Path,
        default=Path("data/openeqa-scannet-required-scenes.txt"),
        help="Optional newline-separated scene id list.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/raw/scannet"),
        help="ScanNet root to create, compatible with data/scannet/extract-frames.py.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Download only the first N scenes, useful for a smoke test.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned downloads without fetching files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing .sens files.",
    )
    parser.add_argument(
        "--agree-tos",
        action="store_true",
        help="Confirm that you have accepted the ScanNet Terms of Use.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retry count for each file.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> int:
    scenes = _load_scene_ids(args.scenes_file, args.dataset)
    if args.limit is not None:
        scenes = scenes[: args.limit]
    if not scenes:
        raise ValueError("No OpenEQA ScanNet scenes found.")

    if not args.dry_run:
        _confirm_tos(args.agree_tos)
    release_scans = set(_fetch_lines(BASE_URL + RELEASE + ".txt", args.timeout))
    release_test_scans = set(_fetch_lines(BASE_URL + RELEASE + "_test.txt", args.timeout))

    planned = []
    missing = []
    for scene_id in scenes:
        if scene_id in release_scans:
            # Official ScanNet v2 uses v1 .sens streams for non-test scans.
            url = f"{BASE_URL}{V1_RELEASE}/{scene_id}/{scene_id}.sens"
            out_file = args.out_dir / "scans" / scene_id / f"{scene_id}.sens"
        elif scene_id in release_test_scans:
            # Match the official downloader: test scan ids are listed in
            # v2/scans_test.txt, while the files live under v2/scans/.
            url = f"{BASE_URL}{RELEASE}/{scene_id}/{scene_id}.sens"
            out_file = args.out_dir / "scans_test" / scene_id / f"{scene_id}.sens"
        else:
            missing.append(scene_id)
            continue
        planned.append((scene_id, url, out_file))

    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(f"Scenes not found in ScanNet v2 release lists: {preview}")

    print(f"OpenEQA ScanNet scenes: {len(planned)}")
    print(f"ScanNet output root:     {args.out_dir}")
    if args.dry_run:
        for scene_id, url, out_file in planned:
            status = "exists" if out_file.exists() else "missing"
            print(f"{status:7s} {scene_id} -> {out_file}")
            print(f"        {url}")
        return 0

    downloaded = 0
    skipped = 0
    for index, (scene_id, url, out_file) in enumerate(planned, start=1):
        print(f"[{index}/{len(planned)}] {scene_id}")
        result = _download_file(
            url=url,
            out_file=out_file,
            force=args.force,
            retries=args.retries,
            timeout=args.timeout,
        )
        if result == "downloaded":
            downloaded += 1
        else:
            skipped += 1

    print(f"Downloaded: {downloaded}")
    print(f"Skipped:    {skipped}")
    return 0


def _confirm_tos(agree_tos: bool) -> None:
    if agree_tos:
        return
    print("ScanNet Terms of Use:")
    print(TOS_URL)
    reply = input("Type 'agree' if you have accepted the terms and want to continue: ")
    if reply.strip().lower() != "agree":
        raise SystemExit("Aborted before downloading ScanNet data.")


def _load_scene_ids(scenes_file: Path, dataset: Path) -> List[str]:
    if scenes_file.exists():
        return _unique_ordered(
            line.strip() for line in scenes_file.read_text().splitlines() if line.strip()
        )

    data = json.loads(dataset.read_text())
    scene_ids = []
    for item in data:
        episode_history = str(item.get("episode_history", ""))
        if not episode_history.startswith("scannet-v0/"):
            continue
        scene_ids.append(_scene_id_from_episode(episode_history))
    return _unique_ordered(scene_ids)


def _scene_id_from_episode(episode_history: str) -> str:
    episode = episode_history.split("/")[-1]
    return episode.split("scannet-")[-1]


def _unique_ordered(items: Iterable[str]) -> List[str]:
    seen = set()
    ordered = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _fetch_lines(url: str, timeout: int) -> Sequence[str]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        lines = []
        for line in response.readlines():
            value = line.decode("utf-8").strip()
            if value:
                lines.append(value)
        return lines


def _download_file(
    *,
    url: str,
    out_file: Path,
    force: bool,
    retries: int,
    timeout: int,
) -> str:
    if out_file.exists() and not force:
        print(f"  skip existing {out_file}")
        return "skipped"

    out_file.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=out_file.parent,
                prefix=out_file.name + ".",
                suffix=".part",
                delete=False,
            ) as tmp:
                tmp_path = Path(tmp.name)
                print(f"  {url}")
                with urllib.request.urlopen(url, timeout=timeout) as response:
                    shutil.copyfileobj(response, tmp)
            tmp_path.replace(out_file)
            print(f"  wrote {out_file}")
            return "downloaded"
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last_error = error
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
            if attempt < retries:
                wait_s = min(30, 2**attempt)
                print(f"  attempt {attempt} failed: {error}; retrying in {wait_s}s")
                time.sleep(wait_s)
    raise RuntimeError(f"Failed to download {url}: {last_error}")


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
