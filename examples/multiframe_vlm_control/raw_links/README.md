# Raw Links (ablation/control input)

`sampled_frames.jsonl` indexes the raw sampled frames + pose/time text the
multi-frame VLM control reads at query time.

This is **ablation/control input only**. It is not exported spatial memory:

- It carries no object inventory, no object labels, and no 3D object positions.
- It must never feed a Track 1/2 fixed object-memory API.
- Object-memory fixed-API evaluation disables raw-frame access.

`manifest.explicit_memory` is `false`. See `../schema.md`.
