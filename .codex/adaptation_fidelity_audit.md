# Adaptation Fidelity & Fairness Audit (2026-06-24)

Audit of the 5 runnable adaptations (SpatialRAG/ClawS, DAAAM, Multi-frame VLM,
ReMEmbR, ReMEmbR-caption-control) for: (1) faithfulness to the original method,
(2) fairness = shared strongest module when a module is common, (3) shared
detector class list. Plus a Track 2 metric change and detector-list recommendations.

## (A) Track 2 metric change — DONE

Removed name-level `referring_acc@1/@5` and `_name_match` from
`track2/evaluator.py`. Track 2 is now **distance-only**: `acc@0.25m` / `acc@0.5m`
(top-1 predicted 3D position within X m of the GT object center) +
`mean_center_distance_m`. Rationale: string overlap between a predicted label and
the GT class name is not instance grounding, and it unfairly penalized free-text
labels (DAAAM DAM prose, e.g. "dark blue trash bag" for *recycling bin*). The
per-query record still keeps `predicted_label` + `target_object_name` for
inspection, but they are not scored.

## (B) Faithfulness — does the adaptation degrade the original method?

| Method | Native stack | Adaptation | Faithful? |
|---|---|---|---|
| **DAAAM** | FastSAM/SAM + DAM grounding + Hydra DSG; native scene-understanding tools | Runs the REAL pipeline (DSG built natively); tool_llm exposes native `get_matching_subjects`+`get_objects_in_radius`. | Yes — no method logic replaced. Only the agent LLM = Claude CLI (a stronger stand-in for its own LLM agent). |
| **ClawS/SpatialRAG** | YOLO+ByteTrack → depth/pose 3D → sqlite-vec memory + VLM describe | Runs the REAL `SpatialPipeline.process_frame`; native `query_spatial_memory`/`get_entity_anchor`/`retrieve_by_location`/`get_all_objects` exposed; also native `fixed_api`. | Yes after the detector fix below. Earlier build degraded it (COCO yolo11n, VLM off). |
| **ReMEmbR** | VILA captioner + Milvus + LangGraph agent over caption memory | Caption memory in native MemoryItem shape; native `retrieve_from_text/position/time`. Captioner = Claude CLI (VILA not installed). | Yes for the retrieval-agent design; captioner is a documented substitute (VILA unavailable). |
| **ReMEmbR caption control** | `NonAgent` over caption context, no retrieval | Same captions, tool_llm; `explicit_memory=false`. | Yes — it is a control by construction. |
| **Multi-frame VLM control** | `VLMNonAgent` over raw sampled frames | 12 sampled frames + `retrieve_frames`; `explicit_memory=false`. | Yes — control by construction. |

No adaptation replaces the method's core memory/representation; the only
substitution is the **agent/captioner LLM = local Claude CLI (Opus 4.8)**, applied
uniformly to ALL methods (and the LLM-Match judge), so it does not advantage any
one method. Native deterministic paths (ClawS/DAAAM `fixed_api`) are reported too.

## (B) Fairness — shared strongest module when a module is common

The shared-module registry has TWO profiles. **The strongest modules already
exist locally** but the committed builds used the weaker `smoke` profile:

| Module (common across methods) | smoke (used so far) | formal (strongest, available) |
|---|---|---|
| SAM | vit_b `sam_b.pt` | **vit_h `sam_vit_h_4b8939.pth`** ✓ local |
| OpenCLIP | ViT-B-32/laion2b_s34b | **ViT-H-14/laion2b_s32b** ✓ cached |
| OV detector | yolov8s-world | **yolov8l-world** ✓ local |
| DAM / sentence-t5 / DAM-3B | (already strongest) | same |

**Action**: rebuild detector/SAM/CLIP-based memory with
`--shared-module-profile formal` (DAAAM/HOV-SG/ConceptGraphs all resolve it; all
formal modules verified present). ClawS's `build_scannet_memory.py` already
defaults to YOLO-World-L; its CLIP/SAM are not used (it's YOLO+depth, not
SAM/CLIP), and its embedding model (ollama qwen3-embedding:0.6b) / VLM
(qwen3.5:4b) are the strongest **locally pulled** — note the ClawS config's
intended vLLM Qwen3-VL-Embedding-2B / qwen3.5:35b are not served here (recorded as
a local-substitution caveat, applied equally as ClawS's own option).

ReMEmbR / caption / multiframe use NO detector/SAM/CLIP module (captioner/raw
frames), so they are exempt from the shared-detector requirement; their only
shared component (the agent/captioner LLM) is already identical across methods.

## (C) Shared detector class list — one list for all scenes

The detector class list is already a SINGLE shared file read by all Track 1 scenes
and all detector-based methods:
`spatial_memory_evaluation/assets/class_lists/detector_coverable.txt`
(`common/labels.py:DEFAULT_DETECTOR_COVERABLE_LABELS`). ✓ one list, all scenes.

**It is too small (37 labels).** Coverage of Track 2 ScanEnts3D referring targets
(148 distinct GT object names, 9,508 queries):

| Class list | size | distinct T2 targets covered | T2 queries covered |
|---|---|---|---|
| current `detector_coverable` | 37 | 19 / 148 | **44%** (4,182 / 9,508) |
| **ScanNet200** | 200 | 100 / 148 | **95%** (9,050 / 9,508) |
| NYU40 | 40 | ~ | (coarse; superclasses) |

### Recommended options (pick one shared list)

1. **ScanNet200** (RECOMMENDED) — `CLASS_LABELS_200` (`DualMap/config/class_list/
   scannet200_classes.txt`, also `concept-graphs/conceptgraph/scannet200_classes.txt`).
   Covers 95% of Track 2 targets and is the standard ScanNet benchmark vocabulary;
   DualMap & ConceptGraphs already project onto it natively (`sem_seg_eval.py`),
   so it's the most defensible shared OV list. Cost: 200-way prompt is heavier for
   YOLO-World and dilutes per-class detection slightly.
2. **ScanNet200 ∩ benchmark targets** (~100–110 labels) — restrict ScanNet200 to
   labels that actually appear as Track 1 GT and/or Track 2/3 targets. Keeps ~95%
   coverage with a smaller, sharper prompt. Good balance; needs a one-time
   intersection script.
3. **NYU40** (40 labels) — only marginally bigger than today and coarse
   (superclasses like "furniture"), so referring distinctions blur. Not
   recommended except as a coarse-eval ablation.
4. **Keep 37 + report coverage** — only if we intentionally restrict to a
   "detector-coverable" subset and always report the covered-split fraction.
   Current behavior; the 44% T2 coverage is the cost.

Recommendation: adopt **ScanNet200** as the shared OV prompt/eval list (option 1),
optionally pruned to the benchmark-present subset (option 2) to keep prompts
sharp. This becomes the single `class_list` for Track 1 query generation AND every
detector method's `set_classes`/prompt, and the Track 1 `detector_coverable` split
is redefined against it.

## Next actions (not yet done)

- Re-run Track 2 (distance-only) for methods already built.
- Rebuild detector/SAM/CLIP memory with `--shared-module-profile formal`.
- If adopting ScanNet200: add the list to `assets/class_lists/`, repoint
  `common/labels.py` + the registry `detector_class_list.canonical`, regenerate
  Track 1 queries, and rebuild detector methods.
