# Unified 10-Scene ScanNet Benchmark Results (2026-06-25)

Full evaluation of **5 spatial-memory methods × 10 shared ScanNet scenes × 3 tracks**,
150 tool_llm cells + Track-1 fixed_api. Branch `eval-10scene-unified`. Raw numbers
in `scannet_10scene_results.json` (gitignored `results/` holds per-cell summaries).

**Scenes (shared by all tracks):** scene0015_00, scene0050_00, scene0077_00,
scene0084_00, scene0131_00, scene0193_00, scene0207_00, scene0222_00, scene0256_00,
scene0314_00.

**Eval harness:** per-query INDEPENDENT agent = Claude **haiku** (fastest tier);
Track-3 judge = Claude **sonnet** (LLM-Match). Memory construction is fully LOCAL
(qwen3.5:4b describer/captioner, qwen3-embedding:0.6b) — no Claude in any method's
memory. Query counts: T1 148 detector_coverable, T2 150 (subset15, 15/scene), T3 121.

## Track 1 — Object Location (tool_llm)

| Method (family) | success@5 | success@1 | first-hit (m) | mrr | prox_top1@1m | prox_top1@3m |
|---|---|---|---|---|---|---|
| **DAAAM** (scene_graph) | **0.539** | 0.386 | 0.428 | 0.816 | 0.503 | 0.835 |
| **ClawS** (object_map) | 0.418 | 0.290 | **0.330** | 0.868 | **0.583** | 0.890 |
| ReMEmbR (caption) | 0.094 | 0.045 | 1.368 | 0.694 | 0.060 | 0.920 |
| LLM-with-captions (control) | 0.106 | 0.045 | 1.259 | 0.635 | 0.056 | **0.941** |
| Multi-frame VLM (control) | 0.053 | 0.053 | 0.986 | 1.000 | 0.090 | 0.888 |

**Track 1 — fixed_api (native deterministic query; ClawS/DAAAM only):**

| Method | success@5 | success@1 | first-hit (m) | mrr | latency |
|---|---|---|---|---|---|
| ClawS | **0.472** | 0.325 | 0.334 | 0.862 | **0.6 ms** |
| DAAAM | 0.396 | 0.270 | 0.328 | 0.836 | 4.9 ms |

## Track 2 — Referring (tool_llm, distance-only)

| Method | acc@0.25m | acc@0.5m | acc_top5@0.5m | prox@1m | prox@3m | prox_top5@3m | mean dist (m) |
|---|---|---|---|---|---|---|---|
| **ClawS** | 0.203 | **0.351** | 0.472 | 0.504 | **0.837** | **0.916** | **1.458** |
| **DAAAM** | 0.214 | 0.330 | **0.518** | **0.510** | 0.757 | 0.861 | 1.769 |
| ReMEmbR | 0.000 | 0.000 | 0.000 | 0.037 | 0.900 | 0.987 | 2.065 |
| LLM-with-captions | 0.000 | 0.000 | 0.000 | 0.041 | 0.893 | 0.987 | 2.050 |
| Multi-frame VLM | 0.000 | 0.008 | 0.008 | 0.042 | 0.819 | 0.835 | 2.244 |

## Track 3 — OpenEQA QA (LLM-Match judge, sonnet)

| Method | llm_match | answered_rate |
|---|---|---|
| **LLM-with-captions** (control) | **0.520** | 0.908 |
| **ReMEmbR** (caption) | 0.498 | 0.948 |
| DAAAM | 0.367 | 0.838 |
| ClawS | 0.340 | 0.916 |
| Multi-frame VLM | 0.337 | 0.869 |

## Build cost (per scene, averaged)

| Method | frames used | avg memory | per-frame compute | stored items |
|---|---|---|---|---|
| DAAAM | ALL (~413, full stream) | 21.9 MB | 0.124 s/frame (FastSAM cv + Hydra) | ~173 bg objects |
| ClawS | ALL (~413, stride 1) | 5.1 MB | 0.183 s/frame (incl model-load) | ~44 objects |
| ReMEmbR | ~23 (1 caption / 3s, native cadence) | 0.3 MB | ~5.0 s/frame (qwen caption + embed) | 23 captions |
| LLM-with-captions | reuses ReMEmbR's ~23 captions | 0.7 MB | n/a (reuse) | 23 captions |
| Multi-frame VLM | ~23 raw frames (stride 18) | ~0 (links) | ~0 | 0 (raw frames) |

## Headline findings

1. **Memory type determines capability — a clean localization/recognition split.**
   - **Geometric memory (DAAAM, ClawS) wins localization** (T1/T2): success@5 0.42–0.54,
     first-hit ~0.33–0.43 m, T2 acc@0.5m 0.33–0.35. They build dense 3D object maps.
   - **Caption memory (ReMEmbR, LLM-with-captions) wins QA** (T3): **0.50–0.52 vs
     0.34–0.37**. Natural-language captions answer "what/why/attribute" questions that
     object tables cannot.

2. **The proximity metric vindicates caption memory's coarse spatial sense.** ReMEmbR
   scores ~0 on strict T1/T2 localization (it stores the robot VIEWPOINT, not the object
   center) but **prox_top1@3m ≈ 0.92** — it reliably points to the right *region*, just
   not object-precise coordinates. This is an architectural property, not a frame-count
   or retrieval-quality issue (confirmed: it stays ~0 strict regardless of sampling).

3. **Native deterministic query (fixed_api) is ClawS's distinguishing strength:**
   success@5 0.47 (> its tool_llm 0.42) at **0.6 ms/query** vs ~55 s/query for tool_llm.

4. **DAAAM ≈ ClawS overall on localization**, with a precision/recall trade: ClawS
   localizes tighter (first-hit 0.33 m, prox_top1@1m 0.58) while DAAAM has higher
   top-5 recall (success@5 0.54). On T2 they're near-tied (ClawS better mean-distance
   1.46 m, DAAAM better acc_top5).

## Fairness / fidelity notes (caveats)

- **Frame budget by design, not arbitrary:** DAAAM/ClawS consume the full RGB-D stream
  (online SLAM-style, their native operating point); ReMEmbR samples at its native
  ~1-caption-per-3-seconds cadence (stride 18 on the 6 fps layout); the multi-frame VLM
  control samples the same ~3 s cadence (uniformly across the scene). Same input video,
  each method's native internal sampling.
- **Local-model substitutions (documented):** VILA captioner → qwen3.5:4b (vision,
  ~VILA-3B scale); Milvus vector memory → qwen3-embedding:0.6b cosine over caption
  embeddings (retrieve_from_text is embedding-based, faithful to ReMEmbR's semantic
  retrieval). The shared agent/judge LLM (haiku/sonnet) is the only cross-method
  substitution and is applied uniformly.
- **DAAAM/ClawS T3 answered_rate < 1.0** is a real finding, not a bug: object-table
  memory often cannot directly answer descriptive QA, so the agent exhausts its tool
  budget; the last-step prompt now forces a best-effort answer.
- **No code leakage to the agent:** the prompt contains only metadata + tool schemas +
  the query (no source code); traces confirm the agent emits only tool_call/final and
  never reads the symlinked method source (the symlink exists only so the evaluator can
  execute native tools, e.g. ClawS query_object.py).
- **Track 2 is distance-only** (no name-matching / referring_acc): string overlap is not
  instance grounding and penalized free-text labels.
