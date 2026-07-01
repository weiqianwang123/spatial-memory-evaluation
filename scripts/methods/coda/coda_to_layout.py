"""Convert a downloaded CODa sequence into the standard posed-RGB 'layout' shape
that our ReMEmbR / DAAAM adapters consume, for Track 4 (OC-NaVQA).

CODa raw (on NAS, from fetch/download tooling), per sequence <seq>:
  2d_rect/cam0/<seq>/2d_rect_cam0_<seq>_<frameidx>.png   (rectified RGB)
  poses/dense_global/<seq>.txt   lines: "ts tx ty tz qx qy qz qw"  (world frame; 1/frame)
  poses/dense/<seq>.txt          (local fallback, same format)
  timestamps/<seq>.txt           (one epoch per frame; row i <-> pose row i <-> frame i)

Output layout (mirrors data/scannet_layouts/<scene>/layout/):
  <out>/color/<frameidx>.jpg     (symlink or copy of the cam0 png)
  <out>/pose/<frameidx>.txt      (4x4 world->camera-ish matrix from the quaternion)
  <out>/timestamps.json          {frameidx: epoch_seconds}   (REAL time for OC-NaVQA
                                  time/duration questions)
  <out>/layout_summary.json

We use dense_global poses (OC-NaVQA position GT is in the global frame). Pose row =
world pose of the robot at that frame; we write it as a 4x4 [R|t] so the ReMEmbR
builder's _pose_position_yaw reads position=t and yaw from R. (For ReMEmbR's caption
memory only position+yaw are used; full extrinsics to camera are not needed for the
text path — DAAAM, which needs camera extrinsics + depth, will use calibrations/ +
its own loader.)

Usage:
  python coda_to_layout.py --coda-seq-dir /data/.../coda/seqs/0 --seq 0 \
      --out data/coda_layouts/0 --rgb-stride-seconds 3
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path


def quat_to_R(qx, qy, qz, qw):
    n = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw) or 1.0
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    return [
        [1 - 2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),     2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),     1 - 2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),     2*(qy*qz+qx*qw),     1 - 2*(qx*qx+qy*qy)],
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coda-seq-dir", type=Path, required=True, help="<coda>/seqs/<seq>")
    ap.add_argument("--seq", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rgb-stride-seconds", type=float, default=3.0,
                    help="emit ~1 frame per this many seconds (ReMEmbR captions ~1/3s)")
    ap.add_argument("--cam-fps", type=float, default=10.0)
    ap.add_argument("--pose", choices=("dense_global", "dense"), default="dense_global")
    ap.add_argument("--copy", action="store_true", help="copy RGB instead of symlink")
    args = ap.parse_args()
    seq = args.seq
    d = args.coda_seq_dir

    pose_file = d / "poses" / args.pose / f"{seq}.txt"
    ts_file = d / "timestamps" / f"{seq}.txt"
    cam0 = d / "2d_rect" / "cam0" / seq
    for p in (pose_file, ts_file, cam0):
        if not p.exists():
            raise FileNotFoundError(f"missing CODa input: {p}")

    poses = [ln.split() for ln in pose_file.read_text().splitlines() if ln.strip()]
    times = [float(x) for x in ts_file.read_text().splitlines() if x.strip()]
    # cam0 frames indexed by the trailing integer in the filename
    frames = {}
    for f in cam0.glob(f"2d_rect_cam0_{seq}_*.png"):
        try:
            frames[int(f.stem.rsplit("_", 1)[-1])] = f
        except ValueError:
            continue
    n = min(len(poses), len(times))
    print(f"[seq {seq}] poses {len(poses)} | timestamps {len(times)} | cam0 frames {len(frames)}")

    step = max(1, int(round(args.rgb_stride_seconds * args.cam_fps))) if args.rgb_stride_seconds > 0 else 1
    color_dir = args.out / "color"; pose_dir = args.out / "pose"
    color_dir.mkdir(parents=True, exist_ok=True); pose_dir.mkdir(parents=True, exist_ok=True)
    ts_map = {}
    kept = 0
    for i in range(0, n, step):
        if i not in frames:
            continue
        fid = f"{i:06d}"
        # RGB
        dst_img = color_dir / f"{fid}.jpg"
        if not dst_img.exists():
            if args.copy:
                import shutil; shutil.copy2(frames[i], dst_img)
            else:
                if dst_img.is_symlink(): dst_img.unlink()
                dst_img.symlink_to(frames[i].resolve())
        # pose -> 4x4 matrix
        row = poses[i]
        tx, ty, tz = float(row[1]), float(row[2]), float(row[3])
        qx, qy, qz, qw = float(row[4]), float(row[5]), float(row[6]), float(row[7])
        R = quat_to_R(qx, qy, qz, qw)
        mat = [[R[0][0], R[0][1], R[0][2], tx],
               [R[1][0], R[1][1], R[1][2], ty],
               [R[2][0], R[2][1], R[2][2], tz],
               [0.0, 0.0, 0.0, 1.0]]
        (pose_dir / f"{fid}.txt").write_text(
            "\n".join(" ".join(f"{v:.6f}" for v in r) for r in mat) + "\n")
        ts_map[fid] = times[i]
        kept += 1
    (args.out / "timestamps.json").write_text(json.dumps(ts_map, indent=0))
    (args.out / "layout_summary.json").write_text(json.dumps({
        "dataset": "coda", "sequence": seq, "frames": kept,
        "rgb_stride_seconds": args.rgb_stride_seconds, "pose_source": args.pose,
        "start_epoch": times[0] if times else None, "end_epoch": times[n-1] if n else None,
    }, indent=2))
    print(f"[seq {seq}] layout -> {args.out} | kept {kept} frames (stride {args.rgb_stride_seconds}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
