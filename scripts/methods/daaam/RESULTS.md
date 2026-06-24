# DAAAM 3-Track Adaptation Results

DAAAM is adapted as **scene-graph object memory + LLM tool-calling**: build a
DAAAM/Hydra Dynamic Scene Graph (DSG) from a posed RGB-D scene, package the
native object/background inventory, then evaluate via the `tool_llm` path where
an LLM calls DAAAM's native `get_matching_subjects` tool over the packaged DSG
memory and produces the answer/prediction. This mirrors the ReMEmbR adaptation
(see `scripts/methods/remembr/RESULTS.md`) but over geometric object memory
instead of caption memory. The agent LLM uses the local Claude CLI; the DSG is
built by DAAAM's real pipeline (FastSAM/SAM + DAM grounding + Hydra), not a
stand-in. Generated packages/results live under gitignored `memories/` and
`results/`.

Capabilities: family `scene_graph`, `agent_access.mode = tool_llm`, native tool
`get_matching_subjects` over `memory/native/` (DSG + corrections +
object_positions + background_objects). The deterministic fixed-API object table
is also exported, but the 3-track comparison here is the tool-LLM path.

Native DSG build env (cuDNN + hydra_python): see
`memory/daaam-native-build-env.md`. ScanNet scenes (Track 2/3) use
`scripts/methods/daaam/extract_sens_frames.py` + `export_scannet_layout.py` to
prepare the RGB-D layout, since DAAAM's native exporter is ScanNet++-only.

## Track 1 — Object-Level Location Query (ScanNet++ scene 036bce3393)

- Memory: DAAAM DSG built from the prepared ScanNet++ layout (FastSAM + DAM
  grounding + Hydra), packaged at
  `memories/daaam/scannetpp/036bce3393/daaam-track1-036bce3393` (71 merged
  objects with 3D positions + bboxes; built from the existing native DSG).
- Eval: `--mode tool_llm`, local Claude CLI, max 3 tool calls/query, on the
  37-query detector-coverable split. The LLM calls `get_matching_subjects` over
  the DSG object memory and returns ranked object predictions with 3D positions.
- Result: `success@5 = 0.514`, `success@1 = 0.486`, `recall@1 = 0.151`,
  `recall@5 = 0.328`, `mrr = 0.974`, **`mean_first_hit_distance = 0.32 m`**,
  mean latency ~231 s/query (multi-step tool loop + large DSG candidate set).
- Reading: DAAAM's geometric scene-graph memory localizes found objects very
  precisely (**0.32 m** first-hit error vs ReMEmbR caption memory's 1.30 m), and
  the high MRR (0.97) shows the right object is almost always rank-1 when present.
  success@5 ~0.51 reflects that roughly half the detector-coverable categories
  are present/nameable in the DAM free-text labels; misses are objects DAM did
  not ground or labeled with a description that did not lexically match.

### Reproduce (Track 1)

```bash
# Package from an existing native DAAAM DSG (no re-run):
PKG=$(pwd)/memories/daaam/scannetpp/036bce3393/daaam-track1-036bce3393
python scripts/methods/daaam/build_memory_smoke.py --skip-daaam-run \
  --native-output-dir <native-dsg-out-dir> --run-id daaam-track1-036bce3393 \
  --daaam-python /home/robin_wang/miniforge3/envs/daaam/bin/python

python scripts/evaluate_track1.py "$PKG" --scene-id 036bce3393 --mode tool_llm \
  --llm-command 'claude -p "$(cat {prompt_path})" --output-format text --permission-mode bypassPermissions > {output_path}' \
  --output "$(pwd)/results/daaam/track1-tool_llm/daaam-track1-036bce3393/eval_summary.json"
```

## Track 2 — Instance-Level Referring (ScanEnts3D val, scene0207_00)

- Memory: DAAAM DSG built natively from the ScanNet `scene0207_00` `.sens`
  frames (199 frames extracted via `extract_sens_frames.py` →
  `export_scannet_layout.py`), packaged at
  `memories/daaam/scannetpp/scene0207_00/daaam-track2-scene0207_00` (116
  background objects with 3D positions; postprocess skipped due to a
  sentence_transformers regression, so objects land in BACKGROUND_OBJECTS but
  retain DAM labels + positions).
- Benchmark: the same 15-query distinct-object-type subset used for ReMEmbR
  (`benchmarks/track2/scanents3d/scene0207_00_subset15`), scored on object-name
  referring accuracy + distance-based localization (`acc@0.25m`/`acc@0.5m`,
  top-1 predicted position within X m of the GT object center).
- Eval: `--mode tool_llm`, local Claude CLI, max 3 tool calls/query. The LLM
  calls `get_matching_subjects` over the DSG and returns referring predictions
  with 3D positions.
- Result (15-query subset): `referring_acc@1 = 0.40`, **`acc@0.25m = 0.067`,
  `acc@0.5m = 0.20`**, `mean_center_distance_m = 2.28 m`, mean latency
  ~194 s/query.
- Reading: unlike caption memory (ReMEmbR: acc@0.25m = acc@0.5m = 0.0, because it
  emits no per-object position), DAAAM's scene-graph memory emits 3D object
  positions, so the distance metric activates (20% of targets localized within
  0.5 m). Name-level referring (0.40) is lower than ReMEmbR's caption matching
  (0.87) on this subset because DAM free-text labels for many ScanNet 0207
  objects do not lexically match the GT class names (e.g. the window/desk
  descriptions), and the sparse DSG misses some target instances entirely.

## Track 3 — OpenEQA General QA (ScanNet scene0709_00)

- Memory: DAAAM DSG built natively from the OpenEQA `scene0709_00` frames (188
  frames), packaged at
  `memories/daaam/scannetpp/scene0709_00/daaam-track3-scene0709_00` (233
  background objects; DSG = 350 nodes).
- Benchmark: the 13 OpenEQA questions for `scene0709_00` (all 7 categories),
  `benchmarks/track3/openeqa/scene0709_00`.
- Eval: `--mode tool_llm`, local Claude CLI as the answering agent, max 4 tool
  calls/question. The LLM calls `get_matching_subjects` over the DSG object
  memory and returns a short answer + evidence. Scored by an LLM-Match judge
  (local Claude CLI, separate from the answering call; OpenEQA-style 1-5 rating
  mapped to [0,1]).
- Result: 13/13 answered, **LLM-Match = 0.60** (`llm_judge_available = true`),
  mean latency ~156 s/question.
- Reading: comparable to ReMEmbR caption memory (0.65) — DAAAM answers
  recognition/attribute questions from its grounded object descriptions but,
  like caption memory, is weaker on spatial-relation questions, since the fixed
  object-memory tool exposes per-object descriptions + positions rather than
  relational structure.

## Cross-track summary (DAAAM vs ReMEmbR)

| Track | Metric | DAAAM (scene-graph object memory) | ReMEmbR (caption memory) |
|---|---|---|---|
| 1 | success@5 / first-hit dist | 0.51 / **0.32 m** | 0.375 / 1.30 m |
| 2 | referring_acc@1 / acc@0.5m | 0.40 / **0.20** | 0.87 / 0.0 |
| 3 | LLM-Match | 0.60 | 0.65 |

DAAAM's geometric object memory wins on precise localization (Track 1 distance,
Track 2 distance metric activates at all); caption memory wins on name-level
recognition (Track 2 acc@1, Track 3 QA) because its free-text captions match
class names and answer recognition questions directly. This is the expected
geometric-vs-semantic memory trade-off the benchmark is designed to surface.
