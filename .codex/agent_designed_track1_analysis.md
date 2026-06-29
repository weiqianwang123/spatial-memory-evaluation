# Why Agent-Designed wins Track 1 success@1 by so much — deep analysis

Held-out T1 success@1: **agent_designed 0.774** vs **DAAAM 0.386** vs **ClawS 0.290**
(per-query tool_llm, 10 scenes). That's ~2× DAAAM and ~2.7× ClawS. This documents
*why*, with evidence from the actual memories and per-query metrics.

## The gap is COVERAGE, not ranking

Decompose success@1 into two failure modes:
1. **ranking loss** = success@5 − success@1 (object is in memory but mis-ranked at #1)
2. **coverage loss** = 1 − success@5 (object is not findable at all in top-5)

| method | success@1 | success@5 | ranking loss (@5−@1) | coverage loss (1−@5) |
|---|---|---|---|---|
| agent_designed | 0.774 | 0.948 | 0.174 | **0.052** |
| DAAAM | 0.386 | 0.539 | 0.153 | **0.461** |
| ClawS | 0.290 | 0.418 | 0.127 | **0.582** |

**Ranking loss is nearly identical across all three (0.13–0.17).** The differentiator
is coverage loss: agent_designed misses the object only 5% of the time, DAAAM 46%,
ClawS 58%. Confirmed by mrr: DAAAM/ClawS have very high mrr (0.83–1.0) on many scenes
yet low success@5 — i.e. *when* they hold the object they rank it #1, but they
frequently don't hold it. So **the win is "having the object in memory", not "ranking
it better."**

## Root cause: how many queried objects each memory actually contains

Distinct queried-label coverage (does the memory contain the asked-for category at all):

| scene | #queries | agent_designed | DAAAM | ClawS |
|---|---|---|---|---|
| scene0077 | 10 | 269 objs → **7/10** | 4 objs → 0/10 | 10 objs → 1/10 |
| scene0131 | 15 | 507 objs → **12/15** | 76 objs → 0/15* | 32 objs → 8/15 |
| scene0222 | 19 | 1156 objs → **19/19** | 641 objs → 0/19* | 92 objs → 13/19 |

(*DAAAM stores labels as free-text DAM descriptions / "unknown" in a different field,
so the literal label-match count understates it — its tool_llm agent matches via the
DAM description text instead. But the *instance count* gap is real and is the driver.)

The agent-designed memory holds **10–60× more object instances** than ClawS and far
more usable detections than DAAAM. With the right object physically present and
localized, the per-query agent just has to retrieve it.

## Why the agent-designed memory has so much more coverage — 3 design choices

From `sm_core.py` (the frozen design) vs the hand-built pipelines:

1. **Low detector confidence floor + every frame.** `det_conf = 0.10` and it runs
   YOLO-World-L on *every* posed frame with the full **scannet200** open-vocab
   prompt (`set_classes`). It deliberately favors recall — keep weak detections,
   merge later. ClawS uses a visual-trigger/ByteTrack gate that fires the detector
   far less often (≈10–92 confirmed tracks/scene), so most objects never enter its
   table. DAAAM segments with FastSAM and grounds asynchronously; many objects land
   as background/"unknown" rather than a queryable labeled instance.

2. **`min_obs = 1` — keep single-sighting objects.** It does NOT require an object
   to be seen across many frames. Small/edge/once-glimpsed objects survive. ClawS's
   track-confirmation and DAAAM's observation thresholds drop exactly these.

3. **Per-label merge at `merge_dist = 0.6 m`.** Cross-frame detections of the same
   category within 0.6 m collapse to one instance (weighted centroid, wall-behind
   culling). This is what lets it keep a *high-recall* detection stream without the
   map exploding into noise — it dedups into clean instances (e.g. 1156 raw → still
   queryable) while preserving coverage.

Net: the agent chose a **high-recall detect-everything-then-merge** strategy, whereas
ClawS/DAAAM chose **high-precision gated** strategies that under-populate the map.

## Important caveats (so the number is read honestly)

- **Label-match is free** (each T1 query passes `target_label`; the agent's
  `find_objects` tool emits the queried label), so T1 ≈ *spatial localization +
  coverage*, not naming. success@1 here means "is a correctly-located instance of
  the asked category ranked #1." The agent benefits from this exactly because its
  coverage is high — it almost always *has* a correctly-located candidate.
- **success@5 is even more coverage-dominated** and should not be over-read.
- The same high-recall choice has a cost the benchmark *does* capture elsewhere:
  more instances → larger memory (still ~3 MB mean, within budget) and the build
  runs the detector on every frame (~0.9 s/frame). Form-neutral cost metrics, not
  a dedup metric, are how we account for that.

## One-line answer

Agent-designed wins T1 success@1 because it **populates its memory with far more of
the queried objects** (coverage loss 5% vs 46–58%) via a deliberate high-recall
"detect-every-frame at low confidence (scannet200), keep single sightings, then
merge" pipeline — while DAAAM/ClawS use gated/triggered detection that leaves most
asked-for objects out of the map. Ranking quality is essentially equal across all
three; coverage is the whole story.

Analysis date 2026-06-29. Source: results/{agent_designed,daaam,claws}/track1-tool_llm/.
