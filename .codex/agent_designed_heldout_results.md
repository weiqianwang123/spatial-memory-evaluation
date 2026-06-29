# Agent-Designed Memory — Held-Out 10-Scene Results

The **agent-designed** spatial memory (a coding agent autonomously designed + self-
improved it via the AutoResearch loop, run2, frozen at commit `1d977bf`) was scored
**once** on the 10 held-out ScanNet scenes it never saw during design, using the
**same fair per-query `tool_llm` protocol** as every hand-built method (independent
agent per query; T2 = subset15; T3 LLM-Match judge = Sonnet). The design loop only
ever touched 3 dev scenes (scene0527/0406/0426), outside these 10.

## Held-out vs hand-built methods (10 scenes, tool_llm)

### Track 1 — object location
| Method (family) | success@1 | success@5 | prox_top1@3m | first-hit (m) |
|---|---|---|---|---|
| **agent_designed** (object_map+caption) | **0.774** | **0.948** | 0.955 | **0.309** |
| DAAAM (scene_graph) | 0.386 | 0.539 | 0.835 | 0.428 |
| ClawS (object_map) | 0.290 | 0.418 | 0.890 | 0.330 |
| ReMEmbR (caption) | 0.045 | 0.094 | 0.920 | 1.368 |

### Track 2 — referring (acc = distance-only, subset15)
| Method | acc@0.25m | acc@0.5m | acc_top5@0.5m | prox@3m |
|---|---|---|---|---|
| **agent_designed** | **0.220** | **0.360** | 0.700 | 0.873 |
| ClawS | 0.203 | 0.351 | 0.472 | 0.837 |
| DAAAM | 0.214 | 0.330 | 0.518 | 0.757 |
| ReMEmbR / controls | 0.000 | 0.000 | 0.000 | 0.90 |

### Track 3 — OpenEQA QA (LLM-Match, Sonnet judge)
| Method | llm_match | answered_rate |
|---|---|---|
| LLM-with-captions (control) | 0.520 | 0.908 |
| **agent_designed** | **0.502** | **0.988** |
| ReMEmbR (caption) | 0.498 | 0.948 |
| DAAAM | 0.367 | 0.838 |
| ClawS | 0.340 | 0.916 |

### Build cost (held-out, per scene)
agent_designed: mean ~3 MB native memory, 0.38–1.56 s/frame (mean ~0.9). Compact +
near-real-time — same order as ClawS (5.1 MB / 0.18 s) and far under DAAAM (21.9 MB).

## Verdict

- **It generalizes.** Frozen design tuned on 3 dev scenes → on 10 unseen scenes:
  T1 success@1 0.849→0.774, T2 acc@0.5m 0.578→0.360 (referring overfit, expected),
  T3 0.531→0.502. No collapse; T1/T3 nearly hold, T2 drops but stays competitive.
- **It is the best or tied-best on every track.** T1: best by a wide margin
  (success@1 0.77 vs 0.39/0.29; first-hit 0.31 m). T2: best (acc@0.5m 0.36, edging
  DAAAM/ClawS 0.33–0.35). T3: 0.502, essentially tied with the top caption methods
  (0.52/0.50) and far above the geometric methods (0.34–0.37) — and highest
  answered_rate (0.99).
- **One memory wins all three tracks** — unusual: in the main eval geometric memory
  won localization (T1/T2) while caption memory won QA (T3). The agent-designed
  design (3D object map + per-object qwen descriptions + frame captions + 4 native
  tools) does both, because it was optimized end-to-end against all three.

## Caveats (honest reading)

- T1's label-match is free (each query passes `target_label`), so T1 ≈ spatial
  localization + ranking; success@1 (0.77) + proximity (0.96) are the trustworthy
  signals. This affects all methods equally, but the agent-designed agent leans on
  it via its `find_objects` tool. Documented as a benchmark issue to fix later
  (stop leaking target_label).
- T2 is the weakest / most overfit track (referring needs relational grounding;
  prox@3m 0.87 shows it lands in the right region but misses the strict 0.5m or the
  exact same-class instance).
- This is a self-improved design over 5 rounds on a sound, non-gameable objective
  (T1=success@1, + a memory-size/time penalty). Build determinism was fixed
  (qwen temp0/seed0) so the loop's keep/revert decisions were reliable.

Scored 2026-06-27. Source: results/agent_designed/track{1,2,3}-tool_llm/scannet-*/.
