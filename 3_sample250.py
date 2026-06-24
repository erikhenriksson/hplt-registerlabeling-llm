import json
import random

langs = [
    "fin_Latn",
    "eng_Latn",
    "swe_Latn",
    "cmn_Hans",
    "cmn_Hant",
    "fas_Arab",
    "ell_Grek",
    "pol_Latn",
    "deu_Latn",
    "tur_Latn",
]
K = 250  # segments to sample per file
CTX = 5  # context segments before/after
DROP = {"text", "xml", "md"}  # heavy fields to drop

for lang in langs:
    rng = random.Random(0)
    reservoir = []  # holds (line_no, seg_idx)
    n_seen = 0  # total segments seen so far

    # Pass 1: reservoir-sample (line_no, seg_idx) pairs, streaming.
    # Each pair is unique by construction, so the reservoir is already dedup'd
    # on the target segment.
    with open(f"{lang}_50k.jsonl", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            segs = json.loads(line)["text"].split("\n")
            for seg_idx in range(len(segs)):
                if len(reservoir) < K:
                    reservoir.append((line_no, seg_idx))
                else:
                    j = rng.randint(0, n_seen)
                    if j < K:
                        reservoir[j] = (line_no, seg_idx)
                n_seen += 1

    # Group sampled segments by line so pass 2 only parses needed lines.
    wanted = {}
    for line_no, seg_idx in reservoir:
        wanted.setdefault(line_no, []).append(seg_idx)

    # Pass 2: pull target + context + metadata for sampled segments.
    out_records = {}
    with open(f"{lang}_50k.jsonl", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if line_no not in wanted:
                continue
            doc = json.loads(line)
            segs = doc["text"].split("\n")
            n_segs = len(segs)
            seg_langs = doc.get("seg_langs_openlid_v3")
            meta = {k: v for k, v in doc.items() if k not in DROP}

            for seg_idx in wanted[line_no]:
                start = max(0, seg_idx - CTX)
                end = min(n_segs, seg_idx + CTX + 1)
                rec = {
                    "source_file": f"{lang}_50k.jsonl",
                    "line_number": line_no,
                    "id": doc.get("id"),
                    "target_index": seg_idx,
                    "n_segments": n_segs,
                    "context_start_index": start,
                    "context_end_index": end - 1,
                    "target_segment": segs[seg_idx],
                    "doc_head": segs[:CTX],
                    "context_before": segs[start:seg_idx],
                    "context_after": segs[seg_idx + 1 : end],
                    "target_seg_lang": (
                        seg_langs[seg_idx]
                        if isinstance(seg_langs, list) and seg_idx < len(seg_langs)
                        else None
                    ),
                    "meta": meta,
                }
                out_records[(line_no, seg_idx)] = rec

    # Write out in the original sampled order.
    with open(f"data/{lang}_sample_250.jsonl", "w", encoding="utf-8") as out:
        for key in reservoir:
            out.write(json.dumps(out_records[key], ensure_ascii=False) + "\n")

    print(f"{lang}: wrote {len(reservoir)} segments from {n_seen} total", flush=True)
