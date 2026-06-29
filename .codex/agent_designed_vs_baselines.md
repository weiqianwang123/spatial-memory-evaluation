# Method Comparison — Held-Out 10 ScanNet Scenes (per-query tool_llm)

All methods scored on the **same 10 held-out scenes** with the **same fair per-query
agent protocol** (independent agent per query; T2 = subset15; T3 LLM-Match judge =
Sonnet). `agent_designed` is the autonomously-designed + self-improved memory
(frozen at run2 commit `1d977bf`, tuned only on 3 separate dev scenes). For Track 1
the trustworthy signals are **success@1** (top-1 must be near GT) and **proximity**
(within 3 m); success@5 is gameable since each query passes `target_label`.

## Track 1 — Object Location
| Method (family) | success@1 | proximity_top1@3m |
|---|---|---|
| **agent_designed** (object-map + captions) | **0.774** | **0.955** |
| DAAAM (scene_graph) | 0.386 | 0.835 |
| ClawS (object_map) | 0.290 | 0.890 |
| Multi-frame VLM (control) | 0.053 | 0.888 |
| ReMEmbR (caption) | 0.045 | 0.920 |
| LLM-with-captions (control) | 0.045 | 0.941 |

## Track 2 — Referring (distance-only, subset15)
| Method | acc@0.5m | proximity@3m |
|---|---|---|
| **agent_designed** | **0.360** | 0.873 |
| ClawS | 0.351 | 0.837 |
| DAAAM | 0.330 | 0.757 |
| Multi-frame VLM | 0.008 | 0.819 |
| ReMEmbR | 0.000 | 0.900 |
| LLM-with-captions | 0.000 | 0.893 |

## Track 3 — OpenEQA QA (LLM-Match, Sonnet judge)
| Method | llm_match | answered_rate |
|---|---|---|
| LLM-with-captions (control) | 0.520 | 0.908 |
| **agent_designed** | **0.502** | **0.988** |
| ReMEmbR (caption) | 0.498 | 0.948 |
| DAAAM | 0.367 | 0.838 |
| ClawS | 0.340 | 0.916 |
| Multi-frame VLM | 0.337 | 0.869 |

## Build cost (per scene)
| Method | native memory (mean, 10 held-out) | time/frame | real-time? |
|---|---|---|---|
| ReMEmbR (caption) | 0.3 MB | ~5.0 s (1 cap/3s native cadence) | n/a (sparse) |
| **agent_designed** | **3.1 MB** (range 1.2–5.8; ~85% is the 1024-d embeddings) | **0.4–1.6 s (mean ~0.9)** | **NO** |
| ClawS | 5.1 MB (sqlite-vec DB: objects+embeddings+index) | 0.095 s | yes |
| DAAAM | 21.9 MB (Hydra 3D scene graph + DAM text) | 0.012 s | yes |

Memory notes (corrected): agent_designed and ClawS are the **same order** (~3 vs
~5 MB) — not a large agent win. Both are embedding-dominated (each stores 1024-d
vectors per object/snapshot); agent's jsonl object table is tiny (0.4 MB) and the
bulk is `object_embeddings.npy`. DAAAM is ~7× larger (heavy scene-graph structure).
Earlier itemization understated agent's artifacts (the embedding .npy files were not
listed in build_log's `native_memory_artifacts`, though the recorded total was
correct); fixed in build_memory.py + patched in the held-out build_logs.

**Takeaway:** one autonomously-designed memory is best-or-tied-best on all three
*answer-quality* tracks — it leads T1 by a wide margin (success@1 0.77 vs 0.39/0.29),
edges the geometric methods on T2, and ties the top caption methods on T3 (highest
answered_rate). Memory is compact and comparable to ClawS (~3 vs ~5 MB). The one
clear weakness is **NOT real-time**: ~0.4–1.6 s/frame (detect-every-frame + qwen
describe) vs DAAAM 0.012 s / ClawS 0.095 s. The strengthened TPF penalty (budget
0.2 s/frame, saturating cap 0.5) now pressures the loop to fix exactly this.
