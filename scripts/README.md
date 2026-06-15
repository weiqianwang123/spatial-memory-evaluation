# Scripts

Keep scripts small and grouped by purpose.

- `package/`: package-level utilities shared by all methods.
- `methods/<method>/`: method-specific exporters, smoke tests, or one-off
  native-output conversion scripts.

Do not add new root-level scripts unless they are truly repository-wide entry
points.
