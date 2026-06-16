# Scripts

Keep scripts small and grouped by purpose.

- `package/`: package-level utilities shared by all methods.
- `methods/<method>/`: method-specific exporters, smoke tests, or one-off
  native-output conversion scripts.
- `build_track1_data.py` / `evaluate_track1.py`: formal Track 1 data and eval.
- `build_track2_queries.py` / `evaluate_track2.py`: formal Track 2 data and eval.

Do not add new root-level scripts unless they are truly repository-wide entry
points.
