# No-Explicit-Memory Controls — 3-Track Results

Two **no-explicit-memory controls** bound how well an LLM/VLM does *without* any
built object/scene-graph memory, so the object-memory methods (DAAAM, ClawS) and
caption memory (ReMEmbR) can be read against a floor. Both keep
`explicit_memory = false` and declare every fixed-API track `invalid`; they are
run only through `--mode tool_llm` and must never be promoted to object-memory
baselines. Generated packages/results are gitignored under `memories/`/`results/`.

Build scripts:
- LLM-with-captions (`caption_control`): `scripts/methods/remembr/build_caption_control_package.py`
  (captions reused from the ReMEmbR caption memory; native tools
  `retrieve_from_text`/`retrieve_from_position`/`retrieve_from_time`).
- Multi-frame VLM (`raw_frame_control`): `scripts/methods/multiframe_vlm/build_control_package.py`
  (12 sampled raw frames + pose/time; native tool `retrieve_frames` hands the
  multimodal agent the frame image paths).

Same scenes/benchmarks as the methods: Track 1 ScanNet++ `036bce3393` (37 q),
Track 2 ScanEnts3D `scene0207_00` (15-q subset), Track 3 OpenEQA `scene0709_00`
(13 q, LLM-Match judge). Agent LLM = local Claude CLI (Bedrock, Opus 4.8).

## LLM-with-captions control (`remembr_captions`)

The ReMEmbR `NonAgent` path: an LLM answering from VILA-style frame captions, no
object memory. Caption memory has no per-object 3D output.

| Track | Result |
|---|---|
| 1 — object location (37 q) | success@5=0.162, success@1=0.0, MRR=0.444, first-hit 1.20 m (found: blanket, cabinet, curtain, machine, sofa, table) |
| 2 — referring (15-q subset) | referring_acc@1=0.87, acc@0.25m=acc@0.5m=0.0, dist 2.19 m |
| 3 — OpenEQA QA (13 q) | **LLM-Match=0.81**, 13/13 answered (attribute/state/world-knowledge 1.0; spatial 0.5) |

Reading: captions alone are weak at object localization (Track 1 success@5=0.16)
and emit no object position (Track 2 distance = 0), but strong at name-level
recognition and open-ended QA — Track 3 LLM-Match 0.81 actually exceeds both the
ReMEmbR retrieval agent (0.65) and DAAAM (0.60) on this scene, because dumping all
captions into the prompt answers recognition/attribute/world-knowledge questions
directly without a lossy retrieval/grounding step. This is the expected control
signal: for general QA on a small scene, a no-memory caption context is a strong
floor; for spatial localization it is weak.

## Multi-frame VLM control (`multiframe_vlm`)

The ReMEmbR `VLMNonAgent` path: a multimodal VLM reasoning over 12 sampled raw
frames (+ pose/time), no built memory.

| Track | Result |
|---|---|
| 1 — object location (37 q) | _(pending — results/multiframe_vlm/track1-tool_llm/)_ |
| 2 — referring (15-q subset) | _(pending — results/multiframe_vlm/track2-tool_llm/)_ |
| 3 — OpenEQA QA (13 q) | _(pending — results/multiframe_vlm/track3-tool_llm-judged/)_ |

### Reproduce

```bash
# LLM-with-captions control (reuses ReMEmbR captions per scene)
python scripts/methods/remembr/build_caption_control_package.py \
  --captions-json data/caption_control_inputs/036bce3393.json \
  --dataset scannetpp --scene-id 036bce3393 --run-id captions-036bce3393
PKG=$(pwd)/memories/remembr_captions/scannetpp/036bce3393/captions-036bce3393
python scripts/evaluate_track1.py "$PKG" --scene-id 036bce3393 --mode tool_llm \
  --llm-command 'claude -p "$(cat {prompt_path})" --output-format text --permission-mode bypassPermissions > {output_path}' \
  --output "$(pwd)/results/remembr_captions/track1-tool_llm/captions-036bce3393/eval_summary.json"

# Multi-frame VLM control (samples raw frames from a prepared layout)
python scripts/methods/multiframe_vlm/build_control_package.py \
  --frames-dir data/daaam_layouts/scannet_scene0709_00/<run>/rgb \
  --pose-dir   data/daaam_layouts/scannet_scene0709_00/<run>/pose \
  --dataset openeqa --scene-id scene0709_00 --num-frames 12
```

## Unified 10-scene ScanNet results (2026-06-25)

Full results in `.codex/scannet_10scene_results.md`. Multi-frame VLM control
(raw_frame_control, explicit_memory=false): the lower-bound "answer from a handful
of raw frames, no built memory" baseline.
- T1 success@5 0.053, T2 acc@0.5m 0.008 (no object memory -> ~0 localization),
  proximity_top1@3m 0.888. T3 OpenEQA LLM-Match 0.337.
- Build: 23 raw frames/scene (stride 18, uniform across scene = ReMEmbR cadence),
  retrieve_frames hands all sampled frames + pose to the multimodal agent.
- Confirms the value of explicit memory: raw-frame VLM trails both geometric (T1/T2)
  and caption (T3) memory.
