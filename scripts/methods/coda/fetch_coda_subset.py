#!/usr/bin/env python
"""Selectively fetch ONLY what Track 4 (OC-NaVQA / ReMEmbR) needs from a CODa
sequence zip, via HTTP range requests — avoiding the full ~45-120 GB/seq download.

CODa per-sequence zips (web.corral.tacc.utexas.edu) are ZIP64 with a central
directory and the server supports byte ranges. We parse the CD, then fetch only:
  - poses/dense_global/<seq>.txt   (the GLOBAL poses OC-NaVQA's position GT is in)
  - poses/dense/<seq>.txt          (fallback / local)
  - timestamps/<seq>.txt
  - calibrations/<seq>/*           (camera intrinsics/extrinsics)
  - metadata/<seq>.json
  - 2d_rect/cam0/<seq>/*.png        SPARSELY — every Nth frame (default ~1 / 3 s),
    since the ReMEmbR memory only captions ~1 frame / 3 s.

This turns a 45 GB sequence into ~1-2 GB (sparse RGB) of exactly the needed data.

Output layout (mirrors what the ReMEmbR adapter + CODa loader expect):
  <out>/<seq>/poses/dense_global/<seq>.txt
  <out>/<seq>/poses/dense/<seq>.txt
  <out>/<seq>/timestamps/<seq>.txt
  <out>/<seq>/calibrations/<seq>/*
  <out>/<seq>/metadata/<seq>.json
  <out>/<seq>/2d_rect/cam0/<seq>/<sparse frames>.png

Usage:
  python fetch_coda_subset.py --seq 0 --out /data/.../coda --rgb-stride-seconds 3
"""
from __future__ import annotations
import argparse, os, struct, zlib, sys, time
from pathlib import Path
import requests

BASE = "https://web.corral.tacc.utexas.edu/texasrobotics/web_CODa/sequences/{seq}.zip"

# One persistent session per worker (keep-alive — the TACC server times out under
# many fresh concurrent connections, so we reuse connections and keep concurrency low).
import threading
_tls = threading.local()
def _session() -> requests.Session:
    s = getattr(_tls, "s", None)
    if s is None:
        s = requests.Session()
        s.headers["User-Agent"] = "coda-subset/1.0"
        _tls.s = s
    return s

def _get(url, headers, timeout, retries=5):
    last = None
    for a in range(retries):
        try:
            r = _session().get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            last = e
            time.sleep(min(30, 2 ** a))   # backoff 1,2,4,8,16
    raise last


def _u(fmt, b, o):
    return struct.unpack(fmt, b[o:o + struct.calcsize(fmt)])[0]


def fetch_central_directory(url: str, total: int) -> bytes:
    """ZIP64: read EOCD64 from the tail, then stream the central directory."""
    tail = _get(url, {"Range": f"bytes={total-1_000_000}-{total-1}"}, timeout=120).content
    z = tail.rfind(b"PK\x06\x06")
    if z < 0:
        raise RuntimeError("ZIP64 EOCD not found")
    cd_size = _u("<Q", tail, z + 40)
    cd_off = _u("<Q", tail, z + 48)
    return _get(url, {"Range": f"bytes={cd_off}-{cd_off+cd_size-1}"}, timeout=600).content


def parse_entries(cd: bytes):
    """Yield dicts {name, method, comp, uncomp, lho} for every central-dir entry."""
    off = 0
    while True:
        i = cd.find(b"PK\x01\x02", off)
        if i < 0:
            break
        method = _u("<H", cd, i + 10)
        comp = _u("<I", cd, i + 20)
        uncomp = _u("<I", cd, i + 24)
        nlen = _u("<H", cd, i + 28)
        elen = _u("<H", cd, i + 30)
        clen = _u("<H", cd, i + 32)
        lho = _u("<I", cd, i + 42)
        name = cd[i + 46:i + 46 + nlen].decode("utf-8", "replace")
        extra = cd[i + 46 + nlen:i + 46 + nlen + elen]
        if 0xffffffff in (comp, uncomp, lho):
            j = 0
            while j + 4 <= len(extra):
                eid, esz = struct.unpack("<HH", extra[j:j + 4]); vals = extra[j + 4:j + 4 + esz]; k = 0
                if eid == 1:
                    if uncomp == 0xffffffff: uncomp = _u("<Q", vals, k); k += 8
                    if comp == 0xffffffff: comp = _u("<Q", vals, k); k += 8
                    if lho == 0xffffffff: lho = _u("<Q", vals, k); k += 8
                j += 4 + esz
        yield {"name": name, "method": method, "comp": comp, "uncomp": uncomp, "lho": lho}
        off = i + 46 + nlen + elen + clen


def fetch_member(url: str, e: dict) -> bytes:
    """Range-fetch one member's data (local header + payload) and inflate.

    One combined range request (local header + payload) to halve round-trips, with
    retry/backoff via the shared keep-alive session.
    """
    # local header is 30 bytes + name + extra; name/extra lens vary, so over-fetch a
    # small slack (256B) for the header, then slice precisely.
    end = e["lho"] + 30 + 256 + e["comp"] + 64
    blob = _get(url, {"Range": f"bytes={e['lho']}-{end}"}, timeout=300).content
    ln = _u("<H", blob, 26); le = _u("<H", blob, 28)
    start = 30 + ln + le
    raw = blob[start:start + e["comp"]]
    return zlib.decompress(raw, -15) if e["method"] == 8 else raw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--rgb-stride-seconds", type=float, default=3.0,
                    help="keep ~1 RGB frame per this many seconds (0 = ALL frames)")
    ap.add_argument("--cam-fps", type=float, default=10.0, help="CODa cam nominal fps")
    ap.add_argument("--cameras", default="cam0",
                    help="comma list of cameras to fetch RGB for, e.g. 'cam0' (ReMEmbR) "
                    "or 'cam0,cam1' (DAAAM needs both for stereo depth)")
    ap.add_argument("--workers", type=int, default=4,
                    help="parallel member fetches (keep LOW — TACC server times out under high concurrency)")
    args = ap.parse_args()
    cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]
    seq = str(args.seq)
    url = BASE.format(seq=seq)
    out = args.out / seq
    out.mkdir(parents=True, exist_ok=True)

    total = int(requests.head(url, allow_redirects=True, timeout=30).headers["Content-Length"])
    # Cache the central directory to disk so a relaunch doesn't re-fetch it.
    cd_cache = out / f".cd_{seq}.bin"
    if cd_cache.exists() and cd_cache.stat().st_size > 1000:
        cd = cd_cache.read_bytes()
        print(f"[seq {seq}] zip {total/1e9:.1f} GB — central directory from cache", flush=True)
    else:
        print(f"[seq {seq}] zip {total/1e9:.1f} GB — reading central directory ...", flush=True)
        cd = fetch_central_directory(url, total)
        cd_cache.write_bytes(cd)
    entries = list(parse_entries(cd))
    print(f"[seq {seq}] {len(entries)} entries", flush=True)

    by_name = {e["name"]: e for e in entries}
    # 1) small required files (all of them): poses, timestamps, calib, metadata, 3d bbox GT
    small = [n for n in by_name if (
        n.startswith("poses/dense_global/") or n.startswith("poses/dense/")
        or n.startswith("poses/imu/") or n.startswith("timestamps/")
        or n.startswith(f"calibrations/{seq}/") or n.startswith("metadata/")
        or n.startswith("3d_bbox/"))]
    # 2) RGB for requested cameras (sparse or full)
    rgb_keep = []
    step = max(1, int(round(args.rgb_stride_seconds * args.cam_fps))) if (args.rgb_stride_seconds and args.rgb_stride_seconds > 0) else 1
    for cam in cameras:
        frames = sorted([n for n in by_name if n.startswith(f"2d_rect/{cam}/{seq}/") and n.endswith(".png")],
                        key=lambda n: float(n.rsplit("_", 1)[-1][:-4]))
        keep = frames[::step] if step > 1 else frames
        rgb_keep += keep
        print(f"[seq {seq}] {cam}: {len(frames)} frames -> keep {len(keep)} (stride {args.rgb_stride_seconds}s)", flush=True)
    targets = small + rgb_keep
    print(f"[seq {seq}] small files: {len(small)} | RGB to fetch: {len(rgb_keep)} | total {len(targets)}", flush=True)

    # parallel fetch with resume (skip existing non-empty files)
    from concurrent.futures import ThreadPoolExecutor
    import threading
    todo = [n for n in targets if not ((out / n).exists() and (out / n).stat().st_size > 0)]
    print(f"[seq {seq}] {len(targets)-len(todo)} already present, fetching {len(todo)}", flush=True)
    cnt = {"done": 0, "fail": 0}
    lock = threading.Lock()

    def grab(n):
        dst = out / n
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            dst.write_bytes(fetch_member(url, by_name[n]))
            with lock:
                cnt["done"] += 1
                if cnt["done"] % 200 == 0:
                    print(f"  [seq {seq}] {cnt['done']}/{len(todo)} fetched", flush=True)
        except Exception as ex:
            with lock:
                cnt["fail"] += 1
            print(f"  [seq {seq}] FAIL {n}: {ex}", file=sys.stderr, flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(grab, todo))
    print(f"[seq {seq}] DONE fetched={cnt['done']} fail={cnt['fail']} (of {len(todo)}) -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
