# run4 Agent-Designed Memory — Held-Out Results (the successful run)

**Status: FINAL.** run4 is the current agent-designed baseline. Frozen at r13
(commit `64532f1`, dev loop_objective 1.863), built on the 10 held-out ScanNet
scenes it never saw during design, scored once with the same fair per-query
`tool_llm` protocol as every other method (independent agent per query; T2 =
subset15; T3 LLM-Match judge = Sonnet). Date: 2026-07-01.

## Headline: run4 is the first auto-designed memory to beat run2 on held-out total

| Track | **run4** | run2 (prev best) | run3 |
|---|---|---|---|
| T1 success@1 | 0.690 | **0.774** | 0.583 |
| T2 acc@0.5m | **0.426** | 0.360 | 0.336 |
| T3 llm_match | **0.570** | 0.502 | ~0.42 |
| **accuracy_sum** | **1.686** | 1.636 | ~1.34 |
| memory / scene | 4.7 MB | 3.1 MB | ~3 MB |
| build TPF | 0.19 s/f (full pipeline) | ~0.9 s/f | 0.12 s/f* |

\* run3's low TPF was partly the query-time-deferral loophole; run4 builds semantics
into memory at build time (honest 0.19 s/f, near-real-time).

**run4 wins T2 and T3, loses only T1, and its accuracy_sum (1.686) exceeds run2 (1.636).**
This is the first time an auto-designed memory beats the previous hand-tuned best on the
held-out total.

## Per-scene (run4 held-out)

| scene | T1 | T2 | T3 |
|---|---|---|---|
| scene0015 | 1.000 | 0.067 | 0.550 |
| scene0050 | 0.765 | 0.733 | 0.500 |
| scene0077 | 0.400 | 0.533 | 0.462 |
| scene0084 | 0.579 | 0.429 | 0.577 |
| scene0131 | 0.733 | 0.267 | 0.688 |
| scene0193 | 1.000 | 0.667 | 0.714 |
| scene0207 | 0.581 | 0.500 | 0.568 |
| scene0222 | 0.842 | 0.333 | 0.404 |
| scene0256 | 0.333 | 0.333 | 0.577 |
| scene0314 | 0.667 | 0.400 | 0.656 |
| **mean** | **0.690** | **0.426** | **0.570** |

## Why run4 succeeded where run3 failed

run3's failure was overfitting: it looked good on 3 dev scenes but its T2/T3 collapsed
on held-out (dev T2 ~0.62 → held-out 0.336). run4 fixed exactly this:

1. **6 dev scenes instead of 3** (added scene0354/0462/0500, all three tracks) — a wider
   dev set the loop can't memorize as easily.
2. **dev T2 on the held-out-matched 15-query subset (subset15)** — the dev T2 number is now
   the same protocol as held-out, so the loop optimizes the real thing (this was a bug in the
   initial run4 sandbox; fixed + the run restarted with +4h budget).
3. **Harness bug fixed** — `history.jsonl` untracked so `git reset --hard` on a revert no longer
   eats the best-so-far row (the greedy loop's decisions are now correct).
4. **Genuinely good, generalizable design moves** (all discovered blind, no run2/run3 hints):
   - object-centric 3D semantic map: YOLO-World detect → depth back-project → multi-view fuse
     → amortized qwen caption/embed (semantics built into memory, not deferred to query time).
   - **bbox volumetric center** as the predicted position (matches the GT scoring center) — a
     read-the-metric win that lifts BOTH T1 and T2.
   - **spatial-relation re-ranker** for T2 (above/below→z, near/on→xy-dist, between→midpoint) —
     the T2 skill run3 lacked; this held up on held-out (0.51 dev → 0.426 held-out, no collapse).
   - synonym-expanded label coverage for T1; question-type-aware answers for T3.

Generalization gap is small and healthy: dev (T1 0.82 / T2 0.51 / T3 0.53) → held-out
(0.69 / 0.43 / 0.57) — T3 even rose. No run3-style collapse.

## Remaining weakness: T1 coverage

The only track run4 loses is T1 (0.690 vs run2 0.774). Root cause is the same coverage gap
inherited from run3's object map (the stride / MIN_VIEWS build-speed choices drop objects seen
in few frames), which hurts the hardest scenes (scene0077 0.40, scene0256 0.33). run2's slower
but full-coverage build wins T1 there. This is the one clear lever for a future run: keep run4's
T2/T3 machinery but restore full-frame / keep-single-view coverage for T1.

## vs hand-built baselines (held-out, tool_llm)

| Method | T1 | T2 | T3 |
|---|---|---|---|
| **agent-designed (run4)** | **0.690** | **0.426** | **0.570** |
| DAAAM (scene_graph) | 0.386 | 0.330 | 0.367 |
| ClawS (object_map) | 0.290 | 0.351 | 0.340 |
| ReMEmbR (caption) | 0.045 | 0.000 | 0.498 |

run4 is best on all three tracks vs every hand-built baseline, by a wide margin on T1/T2.
(Build-time comparison + its caveats: see the README results table.)
