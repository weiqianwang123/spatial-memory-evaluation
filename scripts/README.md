# Scripts

Keep scripts small and grouped by purpose.

- `package/`: package-level utilities shared by all methods.
- `methods/<method>/`: method-specific exporters, smoke tests, or one-off
  native-output conversion scripts.
- `build_track1_data.py` / `evaluate_track1.py`: Track 1 (object-level location
  query + build cost) data and eval.
- `build_track2_data.py` / `evaluate_track2.py`: Track 2 (ScanRefer referring)
  data and eval (skeleton; `data_unavailable` until ScanRefer is acquired).
- `build_track3_data.py` / `evaluate_track3.py`: Track 3 (OpenEQA general QA,
  ScanNet + HM3D) data and eval.
- `agent_designed/run_baseline.py`: agent-designed memory baseline harness CLI.
- `tools/`: small inspection utilities such as RGB sequence replay.

Do not add new root-level scripts unless they are truly repository-wide entry
points.

Claude worktree helpers:

```bash
python scripts/tools/run_claude_task.py 01 --worktree /home/robin_wang/spatial-memory-evaluation-task01 --stream-json --background
python scripts/tools/claude_session_dashboard.py --watch 5
python scripts/tools/claude_session_dashboard.py --html /tmp/claude_sessions.html --watch 10
```

Session metadata and logs are written under each worktree's
`.claude/session_logs/`, which is git ignored. With `--stream-json`, raw Claude
events are written to `.stream.jsonl` and the dashboard summarizes recent
activity from that file.
