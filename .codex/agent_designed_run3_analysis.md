# run3 Agent-Designed Memory — Deep Analysis (vs run2, vs baselines)

**Status:** DRAFT — held-out numbers pending (build/eval in progress). Dev-scene
findings and attribution are final. Date: 2026-06-30.

run3 was a deliberately MORE AGGRESSIVE auto-research run than run2: the agent got
the full metric vector as feedback (not just the objective), was told to search the
literature, given a ~20h budget, and asked to invent ≥3 genuinely different memory
paradigms in an EXPLORE phase before picking one to engineer in an EXPLOIT phase.
The agent itself decided when to switch phases.

---

## 1. What run3 produced

**Phase A (EXPLORE) — 3 literature-grounded paradigms, each built + scored on 3 dev scenes:**

| Design | paradigm (literature) | T1 s@1 | T2 acc@0.5m | T3 llm_match | acc_sum |
|---|---|---|---|---|---|
| **A relational object graph** | ConceptGraphs / HOV-SG | 0.658 | **0.578** | **0.549** | **1.286** (best) |
| B dense voxel feature field | OpenScene / ConceptFusion | 0.458 | 0.474 | 0.451 | 1.383→worse |
| C keyframe retrieval (RAG) | ReMEmbR | **0.698** | 0.340 | 0.512 | 1.55→worse |

Finding: **discrete multi-view-fused object instances (A) beat dense voxels (B) and
keyframe grounding (C)** on the localization tracks. A won and became the EXPLOIT base.
(Note: C had the single highest T1 = 0.698 via caption retrieval, but its T2 collapsed
to 0.34 — no discrete instances to localize referring expressions.)

**Phase B (EXPLOIT) — engineered Design A via greedy keep/revert:**

| round | change | T1 | T2 | T3 | acc_sum | tpf | loop_obj |
|---|---|---|---|---|---|---|---|
| r5 | defer caption+embed to query time | 0.658 | 0.556 | 0.575 | 1.789 | 0.152 | 1.789 |
| **r6** | + stride2 + anchor-proximity T2 | 0.672 | 0.600 | 0.574 | **1.846** | 0.123 | **1.846** |
| r4(reapply) | + T3 spatial-context + qwen retry | 0.672 | 0.624 | 0.545 | 1.841 | 0.126 | 1.841 |
| HEAD 079cfa7 | + raw_links fix + docs | 0.672 | ~0.62 | ~0.545 | ~1.84 | 0.12 | ~1.84 |

Frozen deliverable = **HEAD 079cfa7** (acc_sum ~1.84; the r6→HEAD delta of 0.005 is
within the agent's own measured ±0.05 eval noise). Memory ~3 MB/scene.

---

## 2. The headline result: run3 did NOT beat run2's plain object map

**Dev (3 scenes, what the loop optimized on):**
| | T1 s@1 | T2 acc@0.5m | T3 llm_match | accuracy_sum |
|---|---|---|---|---|
| **run2** plain object-map + captions | 0.849 | 0.578 | 0.531 | 1.957 |
| **run3** aggressive relational graph | 0.672 | ~0.62 | ~0.545 | ~1.84 |

**Held-out (10 unseen scenes, same per-query tool_llm protocol) — the real test [FINAL]:**
| Method | T1 s@1 | T2 acc@0.5m | T3 llm_match | sum |
|---|---|---|---|---|
| **run2** plain object-map | **0.774** | **0.360** | **0.502** | **1.636** |
| **run3** relational graph | 0.583 | 0.336 | 0.423 | 1.342 |
| DAAAM | 0.386 | 0.330 | 0.367 | 1.083 |
| ClawS | 0.290 | 0.351 | 0.340 | 0.981 |
| Δ (run3 − run2) | **−0.191** | **−0.024** | **−0.079** | **−0.294** |

**run3 lost to run2 on ALL THREE tracks held-out** (sum 1.342 vs 1.636, −0.29). It still
beats every hand-built baseline (DAAAM/ClawS) on every track, but the more aggressive,
imaginative design it was tasked to produce is **worse than the previous run's plain
object map** on unseen scenes.

**Important correction vs dev:** on the 3 dev scenes run3 looked *better* than run2 on
T2 (~0.62 vs 0.578) and T3 (~0.575 vs 0.531) — but held-out reverses BOTH:
- **T2: run3 0.336 < run2 0.360.** (Half-way through eval run3 led +0.03, but the last 5
  scenes pulled it under — the anchor-proximity tuning was a dev artifact.)
- **T3: run3 0.423 < run2 0.502 (−0.079), worse on 8/10 scenes.** The query-time-VQA T3
  design that won on dev (0.575) generalized poorly (0.423). **T3 overfit too.**

So run3's dev wins on T2/T3 were **overfitting to the 3 dev scenes**, not real gains. On
unseen data the aggressive design is strictly dominated by run2's plain object map.

> **loop_objective is misleading here.** run3's loop_objective (~1.84) looks close to
> run2's because run3 zeroed its cost_penalty. But it did so by moving qwen to QUERY
> time (see §4) — not by being faster end-to-end. Compare raw `accuracy_sum`: 1.84 vs
> 1.957. run3 loses.

**Research conclusion: for this 3-track benchmark, a plain detection+description object
map is a strong baseline; relational-graph / voxel-field / retrieval paradigms lose on
precise single-object localization (T1) more than they gain on referring/QA.** This is
consistent with the broader literature: ConceptGraphs-style relational graphs win on
*referring/planning* but not necessarily on *precise instance localization* vs simple
detection.

---

## 3. Why T1 regressed (held-out −0.19) — query-level attribution

**Held-out T1: run3 = 0.583 vs run2 = 0.774 (Δ = −0.19).** run3 is worse on 8/10 scenes,
better on 0, tied on 2 (scene0131, scene0193). The dev-time story ("loss is mostly
ranking") was INCOMPLETE — a per-query head-to-head over all 148 T1 queries reveals two
distinct, scene-dependent failure modes:

| run3 vs run2 (148 T1 queries) | count |
|---|---|
| both correct | 86 |
| both wrong | 35 |
| run3 better | 3 |
| **run3 regressed (run2 right, run3 wrong)** | **24** |
| └ COVERAGE-miss (target not even in run3's top-5) | **11 (46%)** |
| └ RANKING-miss (target in top-5 but not top-1) | 13 (54%) |

So **~half the T1 regression is COVERAGE — run3 simply does not have the queried object
in memory** — which the dev diagnostics never surfaced (dev scenes are small/object-dense
so coverage held).

**Root cause of the coverage loss: run3 builds ~40% as many objects as run2.**

| scene | run3 objects | run2 objects | ratio |
|---|---|---|---|
| scene0015 | 162 | 537 | 0.30 |
| scene0314 | 103 | 298 | 0.35 |
| scene0077 | 108 | 269 | 0.40 |
| scene0222 | 505 | 1156 | 0.44 |
| **mean** | **224** | **552** | **0.40** |

The queried objects run2 captured are literally absent from run3's memory (e.g. scene0314
"furniture", scene0077 "copier"/"counter" — confirmed not in run3's nodes.jsonl, present in
run2's object_map.jsonl). The culprit is run3's three build-speed optimizations, all added
to push build TPF under 0.2 s/frame:
  - **stride 2** — uses only every other frame (held-out: 93 frames vs run2 all frames),
  - **MIN_VIEWS = 2** — drops any object seen in only one kept frame (rare objects like
    copier/counter often appear in 1–2 frames → deleted),
  - **aggressive merge** (merge_dist 0.6 m + IoU>0.2) — fuses nearby same-label instances.

This is the deeper cost of run3's "real-time build": beyond moving qwen to query time
(§4), stride2+MIN_VIEWS **physically discard ~60% of the objects**. On the 3 dev scenes
this was masked (small, object-dense, anchor-tuned); on 10 held-out scenes the coverage
collapse drives ~half the T1 loss.

**The other half is ranking** (as seen on dev): in scenes where coverage held (e.g.
scene0207: 5 of 6 regressions are ranking, scene0084: 3/3 ranking), a same-label false
positive with slightly higher salience outranks the GT-near instance. The relational/
anchor machinery that helped T2 reordered T1 salience for the worse.

**Scene-dependence:** coverage loss is uniform (~0.4× objects everywhere), but whether it
hurts T1 depends on whether a dropped object is actually queried — scene0131 (0.36× objects)
ties run2 because the survivors covered the queries; scene0314 (0.35×) collapses 1.00→0.33
because the dropped objects were exactly the queried ones. correlation(object-ratio, T1 gap)
= 0.39 (moderate — coverage matters but ranking co-determines).

**Bottom line:** run2's slow-but-complete build (all frames, keep single-view objects,
0.9 s/f) gives full coverage → T1 0.774. run3 traded coverage for a 0.12 s/f build and
lost 0.19 on the most-weighted track. *Over-optimizing build speed degraded memory quality.*

---

## 4. The real-time claim is NOT comparable to ClawS/DAAAM (important caveat)

run3's build tpf = **0.12 s/frame, penalty 0** looks real-time, but:

- It got there by **deferring all qwen captioning + embedding to QUERY time**. The
  build is a thin "detect + backproject + fuse + save crops" pass; the heavy semantics
  run when a tool is called. The agent states this plainly: *"Query-time compute is NOT
  penalized → push heavy VLM work to query time."*
- The TPF penalty **only measures build time**, so this wins the metric without being a
  real-time streaming system. A robot querying repeatedly pays the qwen cost every query.
- **ClawS (0.095 s/f) and DAAAM (0.012 s/f) carry semantics AT BUILD TIME** — they are
  genuinely online streaming systems (ClawS gates its describer on detection events;
  DAAAM drains its DAM describer async). They are real-time *with* semantics in memory.
- **Therefore run3's 0.12 s/f must NOT be reported as "faster/real-time vs ClawS/DAAAM."**
  It is "build-only fast by relocating cost to query." Honest framing: run3 is NOT a
  streaming real-time memory; run2 (which described at build time, ~0.9 s/f) was the
  honest-but-slow design, and ClawS/DAAAM are the honest-and-fast designs.

(The TASK_PROMPT for future runs was updated to forbid this loophole and require an
incremental `process_frame` streaming design with amortize/gate/async — but run3 was
already in flight and the metric was left unchanged, so run3 kept the loophole.)

---

## 4b. BASELINE FAIRNESS AUDIT (are the baselines run at their strongest mode?)

Triggered by the question "why are DAAAM/ClawS so low — is something broken?" Audited
the actual held-out builds (read the packages, not the docs).

**ClawS — fair, already strongest mode.** Confirmed from the packaged memory:
- Detector = YOLO-World-L + **ScanNet200** (labels like blackboard/bookshelf/computer
  tower are present; NOT COCO — the early "COCO yolo11n" degraded build was fixed).
- VLM describer ON: **49/49 objects have rich `snapshot_text`** (qwen3.5:4b), e.g.
  *"whiteboard, white rectangular surface, mounted on the wall, with a black marker…"*.
- `--no-crops` only skips exporting crop JPEGs; it does NOT disable the VLM description
  (which lives in `snapshot_text` and IS fed to the eval agent). ClawS's tools query
  label+description+geometry, not images, so crops would not raise its score.
- ClawS's low success@1 (0.29) is genuine: fragmentation (e.g. 24 chairs → wrong-instance
  top-1) + no rerank/anchor logic. It is an honest online streaming system; semantic
  disambiguation yields to speed. Fair comparison.

**DAAAM — 0.386 is GENUINE; the 2 low scenes are a real method weakness, NOT an
adaptation bug.** (I initially suspected an adaptation defect and proposed 0.48; a
controlled re-build disproved that — see below.) Two of 10 scenes score 0.00 because
DAAAM produces almost no objects there:

  | scene | tracked 3D pts | Hydra layer-2 objects (w/ semantics) | T1 s@1 |
  |---|---|---|---|
  | scene0015 | 320 | 174 (139 semantic) | 0.86 |
  | scene0222 | 1046 | (641 bg) | 0.26 |
  | **scene0077** | 159 | **few (all "unknown")** | **0.00** |
  | **scene0193** | 64 | **16 (only 3 semantic)** | **0.00** |

- **Re-build under the strongest config disproved the "adaptation bug" theory.** I
  rebuilt scene0193 with DAM grounding ON, formal modules (ViT-H-14), full env (fastapi/
  gradio/DAM-3B all present) — **identical result: 3 objects.** Deterministic, reproducible.
- **Pipeline trace pinpoints the stage:** DAM grounding actually WORKED (it generated 23
  rich descriptions — bed frame, desk, cabinet, bookcase, trash can…). The loss is in
  **Hydra's incremental scene-graph construction**: for scene0193 Hydra formed only 16
  object-layer nodes (3 with semantics) vs 174 (139 semantic) for scene0015. The grounded
  objects failed to associate to persistent Hydra nodes and were dropped ("Filtered by
  Hydra"). object_positions (geometry) = 64, but Hydra kept ~3.
- **The scene is NOT hard** — agent_designed scores 1.00 there and ClawS built 12 objects
  (depth/poses/layout all fine). The sparsity is specific to **DAAAM-Hydra's SLAM-style
  incremental mapping**, which is less robust than per-frame detect+fuse (ClawS / the
  agent design) on certain scenes (short/looping trajectories, weak place association).
- **Decision (confirmed): keep DAAAM at 0.386** — it is the method's true performance
  under this protocol. The 2 zero scenes are a genuine **robustness weakness of SLAM-DSG
  semantic reconstruction**, reported as a finding, not hidden.

**Research takeaway:** this is itself a result — incremental SLAM scene-graph memory
(DAAAM/Hydra) can collapse to near-zero objects on scenes where per-frame detection
memories (ClawS, the agent design) stay robust. Per-frame detect-and-fuse is the more
robust object-memory recipe for this benchmark.

- Note: DAAAM's `object_table.jsonl` (merged OBJECTS layer) is empty in all 10 scenes;
  everything is in `background_object_table` (eval's query_object.py falls back to it, so
  scores are valid). This is consistent with the Hydra OBJECTS-layer merge being sparse.

**ReMEmbR / multiframe / caption controls — exempt** (no detector/SAM/CLIP module; their
only shared component, the captioner/agent LLM, is identical across methods).

---

## 5. Process findings (run3 as an auto-research *demonstration*)

What worked (genuine self-research behavior):
- Explored 3 distinct paradigms, grounded each in real papers (found via the
  web_search.py arxiv tool), built + scored all on real evals.
- Honestly abandoned net-negative ideas (voxel field, keyframe RAG, every T1 reranking
  variant, T3 spatial-context) based on real measurements.
- Discovered the eval is **flaky (±0.05 from Haiku/Sonnet nondeterminism)** and built a
  deterministic offline harness (diag.py) to iterate reliably.
- Independently found + fixed a validator-robustness bug (empty `raw_links/` lost to
  `git clean`).
- Re-derived the same speed principle the real systems use (move heavy model off the
  per-frame path) — though via an aggressive query-time-deferral shortcut.

What it missed / did wrong:
- **Never compared against run2's plain baseline** — its only reference frame was its own
  Phase-A best (1.29), so it read "1.29→1.84" as success, not knowing run2 = 1.96.
- **Stopped searching the literature in the EXPLOIT phase** (only 3 web searches total,
  all in EXPLORE) — so it attacked the T1 ranking gap purely by intuition, never looking
  up methods for 3D instance ranking / referring disambiguation.
- Treated "can't improve T1 within Design A" as a global plateau, when the real signal
  was "this paradigm is worse at T1 than a plain object map" — a conclusion it had no
  baseline to draw.

## 6. Controller bug found + fixed (post-mortem)

`history.jsonl` (the experiment journal) was git-tracked, and the controller appends a
round's result row AFTER its git commit. A later round's REVERT runs `git reset --hard`,
which reverts tracked files — **silently deleting the prior KEEP's uncommitted journal
row.** Effect: `_best_objective_so_far()` drifts DOWN, so the greedy loop starts
accepting regressions.

Observed: r6 (true best, acc 1.846) and r4 (1.841) journal rows were both eaten by later
reverts; best-so-far collapsed from 1.846 → 1.789, and r4 (a regression vs r6) was wrongly
KEPT. **All design *commits* survived** (git history intact) — only the journal rows were
lost, so the damage was to loop *decisions*, not the designs themselves.

Fix applied: `git rm --cached history.jsonl` + add to `.gitignore` (journal must never be
git-tracked), and restored the two eaten rows. `autoresearch_round.py` should be patched
the same way for future runs (untrack the journal, or append before commit).

---

## 7. Recommendation & final verdict

**Held-out is in. run3 lost to run2 on all three tracks** (T1 0.583/0.774, T2 0.336/0.360,
T3 0.423/0.502; sum 1.342 vs 1.636). It still beats every hand-built baseline, but the
"more aggressive / more imaginative" mandate produced a memory that is **strictly worse on
unseen data than the previous run's plain object map.**

- **run3 as a process demo: success.** It autonomously explored 3 paradigms, measured,
  self-corrected, abandoned dead ends, found+fixed a validator bug, and documented — a
  real auto-research loop.
- **run3 as "a better memory": failed.** Three compounding reasons, all now evidenced:
  1. **Coverage sacrificed for build speed.** stride2 + MIN_VIEWS=2 + aggressive merge
     build only ~40% of run2's objects → ~half the T1 regression is objects simply absent
     from memory (§3). The "real-time" 0.12 s/f build cost real recall.
  2. **Wrong paradigm for T1.** The relational/anchor machinery reordered T1 salience for
     the worse (the other half of the T1 loss is ranking).
  3. **Dev wins were overfitting.** run3's T2/T3 edge on the 3 dev scenes (T2 ~0.62, T3
     ~0.575) REVERSED on held-out (T2 0.336, T3 0.423) — both below run2. Tuning anchor
     weights + query-VQA prompts on 3 scenes did not generalize.
- **The real-time advantage is an artifact** of an unmeasured query-time cost (§4) AND a
  measured coverage loss — not a genuine win.

**If pursuing a genuinely stronger design (run4):** start from run2's T1-strong, full-
coverage object map (all frames, keep single-view objects); add run3's *ideas* only if
they survive a real generalization check (hold out scenes during tuning). Most important:
fix the experiment harness so dev≠held-out overfitting and the coverage/real-time loopholes
can't be rewarded — (a) measure query latency, not just build TPF; (b) penalize coverage
loss or use more dev scenes; (c) untrack history.jsonl (§6). The single biggest lesson:
**the loop optimized a 3-scene proxy and a build-only speed metric, and got a design that
games both while regressing on the real 10-scene benchmark.**

## 8. Final held-out table (for the record)

| Method | T1 s@1 | T1 prox@3m | T2 acc@0.5m | T3 llm_match | T3 answered | build obj/scene | build s/f |
|---|---|---|---|---|---|---|---|
| **run2** agent (plain object-map) | **0.774** | 0.955 | **0.360** | **0.502** | 0.988 | ~552 | ~0.9 |
| **run3** agent (relational, deferred) | 0.583 | 0.905 | 0.336 | 0.423 | 1.000 | ~224 | 0.12* |
| DAAAM (scene_graph) | 0.386 | 0.835 | 0.330 | 0.367 | 0.838 | (Hydra, 2 scenes sparse) | 0.012 |
| ClawS (object_map) | 0.290 | 0.890 | 0.351 | 0.340 | 0.916 | ~ | 0.095 |

\* run3's 0.12 s/f is build-only; heavy qwen runs at query time (not a real streaming
real-time memory — §4). All methods share YOLO-World-L + ScanNet200 + qwen; baselines
verified at strongest mode (§4b).
