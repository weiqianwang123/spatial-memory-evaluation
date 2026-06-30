# Track 4 — OC-NaVQA: data form + integration findings

Goal: add OC-NaVQA as Track 4 (an independent track — different scenes, more independent
launch). First milestone: get ReMEmbR's eval running on it; everything except data+query
mirrors tracks 1-3 (same agent stack, package/eval conventions).

## What OC-NaVQA IS
- Long-horizon **video/trajectory QA for a mobile robot** — NOT ScanNet rooms. It is a
  CORRECTED version of ReMEmbR's NaVQA, over the **CODa** dataset (UT Austin campus robot,
  outdoor/indoor driving sequences). DAAAM ships the corrected annotations.
- Source data already local:
  - `/home/robin_wang/DAAAM/data/oc-navqa_data.csv` — 210 Q over **7 CODa sequences**
    (Seq IDs 0,3,4,6,16,21,22; 30 Q each). Columns: UUID, Seq ID, Question, Timestamp-
    with-answer, Type, Text answer, Parsable answer, Category, GT Response, GT FullSeq.
  - `/home/robin_wang/remembr/remembr/data/navqa/<seq>/qa_unfilled.json` — per-sequence
    question templates with id, length_category, start_time, end_time, file_info
    {qa_start_filename, qa_end_filename}. (DAAAM's csv replaces remembr's data.csv.)
- Question **types** (210): position 71, binary 67, text 33, time 30, duration 9.
  Scoring per type (from `remembr/scripts/eval.py: evaluate_output`):
  - position -> L2 distance (m) between predicted xyz and GT object position.
  - binary -> exact yes/no match.
  - time / duration -> absolute error in minutes.
  - text -> free text (LLM/human judge; ReMEmbR uses an LLM judge for these).

## Memory model (what a method consumes)
ReMEmbR memory = list of `MemoryItem(caption, time, position, theta)`, one per ~3s window
along the trajectory. The eval slices items by [start_time, end_time] of the question
(horizon = beginning-of-seq -> current time). Two memory paths:
- **TextMemory / our adapter** needs ONLY the per-window {position, theta, time, caption}
  — i.e. the **captions JSON**. This is the path our ReMEmbR adapter already implements
  (`scripts/methods/remembr/build_memory_package.py`, MemoryItem shape +
  retrieve_from_text/position tools). Captioner is pluggable (--captioner claude).
- VideoMemory/VLM path additionally needs the raw RGB .pkl frames (only for VLM models).

## The gating dependency: captions JSON
- captions JSON = `data/captions/{seq}/captions/{caption_file}.json`, list of
  {file_start, file_end, time, position, theta, caption}.
- It is PRODUCED by captioning the CODa RGB frames (ReMEmbR uses VILA every 3s). It does
  NOT exist locally, and CODa raw data is NOT on the NAS yet.
- CODa layout the loader expects (DAAAM `src/daaam/datasets/loaders/coda.py`): per sequence
  `2d_rect/cam0/<seq>/*.png` (RGB), `poses/dense_global/<seq>/*.txt` (world->os1),
  `calibrations/<seq>/*.yaml`, `timestamps/<seq>.txt`. (We need RGB + dense_global poses +
  timestamps only for the text path — not lidar/depth.)
- Download via coda-devkit (eval.md). Full processed = ~335GB; the 7 seqs RGB+pose subset
  is smaller. Disk: NAS /data has 10P free; /home has 698G free.

## Integration plan (mirror tracks 1-3)
1. Stage CODa raw (7 seqs: RGB cam0 + dense_global poses + timestamps + calib) to NAS:
   `/data/mondo-training-dataset/semantic_mapping/coda/<seq>/...`.
2. Build a Track-4 benchmark dir mirroring track3:
   `benchmarks/track4/oc_navqa/<seq>/{questions.jsonl, answers.jsonl}` derived from
   oc-navqa_data.csv + qa_unfilled.json (fill timestamps->context, parse GT per type).
3. Prepare a CODa "layout" per sequence (color/<frame>.jpg + pose/<frame>.txt + timestamps)
   like the ScanNet layouts, so our ReMEmbR adapter's build_memory_package.py can caption it
   into MemoryItem(caption,time,position,theta) memory.
4. Add a Track-4 evaluator (per-type scoring above) reusing the tool_llm per-query agent
   harness (same Haiku answer agent, Sonnet judge for text) — only data+query+metric differ.
5. First milestone: run the existing ReMEmbR adapter on Track 4.

## Independence (per request)
Track 4 uses entirely different scenes (CODa, not ScanNet) and can launch more
independently — separate benchmark root (benchmarks/track4/oc_navqa), separate layout root
(data/coda_layouts or NAS), separate eval driver. Does NOT touch tracks 1-3 or run4.
