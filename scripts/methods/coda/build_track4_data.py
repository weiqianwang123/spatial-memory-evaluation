#!/usr/bin/env python
"""Build the Track 4 (OC-NaVQA) benchmark from DAAAM's corrected annotations.

Track 4 = long-horizon trajectory QA over CODa robot sequences. Source:
  - DAAAM/data/oc-navqa_data.csv      (210 corrected Q/A over 7 CODa seqs)
  - remembr/.../navqa/<seq>/qa_unfilled.json  (per-Q start_time/end_time/length)

We emit, per sequence, a benchmark dir mirroring Track 3's shape so the eval
harness can reuse the per-query agent loop:
  benchmarks/track4/oc_navqa/<seq>/questions.jsonl   # question_id, question, type, episode_id, (current-time context)
  benchmarks/track4/oc_navqa/<seq>/answers.jsonl     # question_id, type, + type-specific GT

GT per type (scoring handled by the track4 evaluator):
  - binary   : answer in {yes,no}                      -> exact match
  - position : answer_position [x,y,z] (global, from CSV GT Response) -> L2 distance (m)
  - time     : minutes_ago (float)                     -> abs error (min)
  - duration : minutes (float)                         -> abs error (min)
  - text     : free text                               -> LLM-Match judge (like T3)

Timezone: the CSV "Timestamp with answer" H:M:S are in UTC-8; the question date
comes from qa_unfilled start_time (epoch). minutes_ago = (end_time - answer_epoch)/60.
The question text is augmented with start/current time so the agent can reason about
"when/how long" (mirrors ReMEmbR's form_question_jsons).

Usage:
  python build_track4_data.py --out benchmarks/track4/oc_navqa
"""
from __future__ import annotations
import argparse, csv, json, calendar, datetime
from pathlib import Path

CSV = Path("/home/robin_wang/DAAAM/data/oc-navqa_data.csv")
QA_UNFILLED_ROOT = Path("/home/robin_wang/remembr/remembr/data/navqa")
SEQS = ["0", "3", "4", "6", "16", "21", "22"]
CSV_TZ_OFFSET_HOURS = -8  # empirically: CSV H:M:S are UTC-8 (answer lands in [start,end])


def _col(cols, prefix):
    return next(c for c in cols if c.startswith(prefix))


def _answer_epoch(hms: str, q_start_epoch: float) -> float | None:
    """Convert a CSV 'H:M:S' answer timestamp to absolute epoch, using the
    question's UTC date and the known CSV UTC-8 offset."""
    hms = hms.strip()
    if not hms:
        return None
    day_utc = datetime.datetime.utcfromtimestamp(q_start_epoch).strftime("%m/%d/%Y")
    dt = datetime.datetime.strptime(f"{day_utc} {hms}", "%m/%d/%Y %H:%M:%S")
    return calendar.timegm(dt.timetuple()) - CSV_TZ_OFFSET_HOURS * 3600


def _parse_position(gt: str):
    gt = gt.strip().strip("[]")
    if not gt:
        return None
    parts = [float(x) for x in gt.replace(",", " ").split()]
    return parts if len(parts) == 3 else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("benchmarks/track4/oc_navqa"))
    ap.add_argument("--csv", type=Path, default=CSV)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    cols = list(rows[0].keys())
    C = {k: _col(cols, p) for k, p in {
        "type": "Type", "parsable": "Parsable", "gt": "GT Response",
        "text": "Text answer", "ts": "Timestamp", "cat": "Question\nCat"}.items()}
    # 'GT Response' has a FullSeq sibling — make sure we grabbed the non-FullSeq one
    C["gt"] = next(c for c in cols if c.startswith("GT Response") and "FullSeq" not in c)

    stats = {"per_seq": {}, "by_type": {}, "skipped": 0}
    for seq in SEQS:
        qaf = QA_UNFILLED_ROOT / seq / "qa_unfilled.json"
        if not qaf.exists():
            print(f"[seq {seq}] no qa_unfilled.json — skip"); continue
        qa_by = {q["id"]: q for q in json.loads(qaf.read_text())["data"]}
        sub = [r for r in rows if r["UUID"] in qa_by and (r["Question"] or "").strip()]
        qdir = args.out / seq
        qdir.mkdir(parents=True, exist_ok=True)
        questions, answers = [], []
        for r in sub:
            uuid = r["UUID"]; fq = qa_by[uuid]
            qtype = r[C["type"]].strip().lower()
            question = r["Question"].strip()
            start_e, end_e = fq["start_time"], fq["end_time"]
            start_iso = datetime.datetime.utcfromtimestamp(start_e).strftime("%Y-%m-%d %H:%M:%S")
            cur_iso = datetime.datetime.utcfromtimestamp(end_e).strftime("%Y-%m-%d %H:%M:%S")
            # augment question with temporal framing (the agent needs "now" for when/how-long)
            q_aug = (f"You started at time {start_iso}. The current time is {cur_iso}. "
                     f"The context is everything observed from the start until now.\n{question}")
            ans = {"question_id": uuid, "type": qtype, "category": (r[C["cat"]] or "").strip()}
            ok = True
            if qtype == "binary":
                ans["answer"] = r[C["parsable"]].strip().lower()
                ok = ans["answer"] in ("yes", "no")
            elif qtype == "position":
                pos = _parse_position(r[C["gt"]])
                if pos is None:
                    ok = False
                else:
                    ans["answer_position"] = pos
                    ans["answer"] = f"position {pos}"
            elif qtype == "time":
                ae = _answer_epoch(r[C["ts"]], start_e)
                if ae is None:
                    ok = False
                else:
                    ans["minutes_ago"] = round((end_e - ae) / 60.0, 3)
                    ans["answer"] = f"{ans['minutes_ago']} minutes ago"
            elif qtype == "duration":
                try:
                    ans["minutes"] = float(r[C["parsable"]].strip())
                    ans["answer"] = f"{ans['minutes']} minutes"
                except ValueError:
                    ok = False
            elif qtype == "text":
                ans["answer"] = (r[C["text"]] or "").strip()
                ok = bool(ans["answer"])
            else:
                ok = False
            if not ok:
                stats["skipped"] += 1
                continue
            # OC-NaVQA length category (SHORT/MEDIUM/LONG) = the context horizon the
            # question must reason over (seconds from seq start to "now"). Read from
            # qa_unfilled (NOT the id prefix — that is the seq number for most seqs).
            length_cat = fq.get("length_category")
            length_s = fq.get("length")
            ans["length_category"] = length_cat
            questions.append({"question_id": uuid, "question": q_aug, "type": qtype,
                              "category": ans["category"], "episode_id": f"coda-seq{seq}",
                              "length_category": length_cat, "length_seconds": length_s,
                              "raw_question": question})
            answers.append(ans)
            stats["by_type"][qtype] = stats["by_type"].get(qtype, 0) + 1
            stats["by_length"] = stats.get("by_length", {})
            stats["by_length"][length_cat] = stats["by_length"].get(length_cat, 0) + 1
        (qdir / "questions.jsonl").write_text("".join(json.dumps(q) + "\n" for q in questions))
        (qdir / "answers.jsonl").write_text("".join(json.dumps(a) + "\n" for a in answers))
        stats["per_seq"][seq] = len(questions)
        print(f"[seq {seq}] {len(questions)} questions -> {qdir}")
    print(f"\nTotal: {sum(stats['per_seq'].values())} questions across {len(stats['per_seq'])} seqs "
          f"| by_type={stats['by_type']} | by_length={stats.get('by_length',{})} | skipped={stats['skipped']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
