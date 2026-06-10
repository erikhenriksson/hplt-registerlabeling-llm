#!/usr/bin/env python3
"""
Classify each newline-separated segment of each document into one of the
8 CORE main registers, using surrounding segments as context.

Each generation call labels a small CORE of segments (CORE of them) while
showing RADIUS extra segments on each side as context-only. The cores tile
every document with no gaps and no overlap, so every segment is labeled
exactly once and the output is always N labels for N segments.

Built for very large inputs (millions of docs), including docs of any length
(1 to 10000+ segments):
  - Input is STREAMED, never fully loaded into memory.
  - Output is APPENDED one doc at a time.
  - Re-running the same command RESUMES: docs already in the output are skipped.

Input:  en_sample.jsonl       (one doc per line; uses keys "id" and "text")
Output: en_sample.out.jsonl   (one line per doc: id + list of per-segment labels)
"""

import os

os.environ.setdefault("HF_HOME", os.path.join(os.getcwd(), "hf_cache"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(os.getcwd(), "hf_cache"))

import argparse
import itertools
import json
import time

import sglang as sgl
from transformers import AutoTokenizer

# ---- configuration --------------------------------------------------------

MODEL_PATH = "Qwen/Qwen3.6-35B-A3B"
TP_SIZE = 2
CORE = 4  # how many segments each call actually labels
RADIUS = 4  # how many segments on each side to show as context-only
SEGMENT_CHAR_LIMIT = 500  # first N characters of each segment
DOC_CHUNK = 200  # how many docs to process per batch; bounds memory

# The 8 CORE main registers (Egbert/Biber et al.).
REGISTERS = {
    "NA": "Narrative — recounting events; news reports, blogs, news+opinion, fiction, travel.",
    "OP": "Opinion — author argues a personal viewpoint; reviews, personal opinion blogs, advice.",
    "IN": "Informational description — informs/explains; encyclopedic, research, descriptions of things.",
    "IP": "Informational persuasion — informs while persuading; marketing, descriptions with intent to sell.",
    "HI": "How-to / instructional — tells the reader how to do something; recipes, guides, FAQs.",
    "ID": "Interactive discussion — forum threads, comment sections, Q&A discussions.",
    "LY": "Lyrical — songs, poems, prayers; aesthetic/artistic language.",
    "SP": "Spoken — transcribed speech; interviews, formal speeches, TV/video transcripts.",
}
LABELS = list(REGISTERS.keys())

SYSTEM_PROMPT = (
    "You are a text register classifier using the CORE taxonomy. "
    "You classify segments of a web page into the eight main registers, "
    "using the surrounding segments only as context.\n\n"
    "The eight registers are:\n"
    + "\n".join(f"  {k}: {v}" for k, v in REGISTERS.items())
    + "\n\nRespond with ONLY the two-letter codes for the segments marked "
    "'classify', one per segment, comma-separated, in order. "
    "No explanation, no other text."
)


# ---- small building blocks ------------------------------------------------


def split_segments(text):
    """Newline-separated segments, stripped, empties dropped."""
    return [s.strip() for s in text.split("\n") if s.strip()]


def labels_regex(n):
    """Grammar that permits exactly n comma-separated register codes."""
    one = "(" + "|".join(LABELS) + ")"
    return one + ("," + one) * (n - 1)


def build_prompt(segments, core_start, core_len, ctx_lo, ctx_hi):
    """
    The context window for one core of segments: the CORE itself plus
    RADIUS neighbours on each side, shown as context.
    """
    core_end = core_start + core_len  # exclusive
    lines = []
    for i in range(ctx_lo, ctx_hi):
        tag = "classify" if core_start <= i < core_end else "context"
        lines.append(f"[SEGMENT {i}] ({tag})")
        lines.append(segments[i][:SEGMENT_CHAR_LIMIT])

    window = "\n".join(lines)
    last = core_end - 1
    return (
        f"Below are segments {ctx_lo}-{ctx_hi - 1} of a web page. "
        f"Classify ONLY segments {core_start}-{last} (marked 'classify'). "
        f"Use the others as context.\n\n"
        f"{window}\n\n"
        f"Register codes for segments {core_start}-{last}:"
    )


def chunked(iterable, n):
    """Yield successive n-sized lists from any iterable. Never holds it all."""
    it = iter(iterable)
    while batch := list(itertools.islice(it, n)):
        yield batch


# ---- reading input and resume state ---------------------------------------


def read_docs(path):
    """Yield docs one at a time, with their segments attached. Streams the file."""
    with open(path) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                d["segments"] = split_segments(d.get("text", ""))
                yield d


def ids_already_done(output_path):
    """
    Return the set of doc ids already written to the output.
    A truncated final line (from a previous crash mid-write) is skipped, not fatal.
    """
    done = set()
    if not os.path.exists(output_path):
        return done
    with open(output_path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                pass  # partial/garbled trailing line from an interrupted run
    return done


# ---- the one slightly-tricky box: classify a batch of docs ----------------


def classify_chunk(engine, tok, chunk):
    """
    Classify every segment of every doc in `chunk`.
    Returns one list of register codes per doc, in the same order as `chunk`.

    Cores tile each doc by CORE with no overlap, so every segment is labeled
    exactly once. Each call also sees RADIUS context segments on each side.
    """
    labels = [[None] * len(d["segments"]) for d in chunk]

    # Flatten: one prompt per core, remembering which doc/core it is, plus the
    # per-prompt sampling params (the regex and length depend on the core size).
    prompts, index, sps = [], [], []
    for doc_i, d in enumerate(chunk):
        segs = d["segments"]
        for core_start in range(0, len(segs), CORE):
            core_len = min(CORE, len(segs) - core_start)
            ctx_lo = max(0, core_start - RADIUS)
            ctx_hi = min(len(segs), core_start + core_len + RADIUS)
            chat = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_prompt(segs, core_start, core_len, ctx_lo, ctx_hi),
                },
            ]
            prompts.append(
                tok.apply_chat_template(
                    chat,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
            index.append((doc_i, core_start, core_len))
            sps.append(
                {
                    "temperature": 0.0,
                    "max_new_tokens": 3 * core_len + 4,
                    "regex": labels_regex(core_len),
                }
            )

    outputs = engine.generate(prompts, sps)

    # Regroup: scatter each core's labels back into its doc's list.
    for (doc_i, core_start, core_len), o in zip(index, outputs):
        parts = [p.strip() for p in o["text"].strip().split(",")]
        assert len(parts) == core_len
        labels[doc_i][core_start : core_start + core_len] = parts

    # Integrity check: every segment of every doc got exactly one label.
    for doc_i in range(len(chunk)):
        assert None not in labels[doc_i]
    return labels


# ---- the top-level story, readable straight down --------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="en_sample.jsonl")
    ap.add_argument("--output", default="en_sample.out.jsonl")
    args = ap.parse_args()

    done = ids_already_done(args.output)
    if done:
        print(f"[resume] {len(done)} docs already done; skipping those")

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    engine = sgl.Engine(
        model_path=MODEL_PATH,
        tp_size=TP_SIZE,
        mem_fraction_static=0.90,
        context_length=8192,
        reasoning_parser="qwen3",
        max_running_requests=64,
    )

    # Stream input, skip what's done, process the rest in fixed-size chunks.
    todo = (d for d in read_docs(args.input) if d.get("id") not in done)

    processed = 0
    t0 = time.perf_counter()
    with open(args.output, "a") as out:
        for chunk in chunked(todo, DOC_CHUNK):
            labels = classify_chunk(engine, tok, chunk)
            for d, segment_registers in zip(chunk, labels):
                out.write(
                    json.dumps(
                        {
                            "id": d.get("id"),
                            "segment_registers": segment_registers,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            out.flush()
            processed += len(chunk)
            print(
                f"[gen] {processed} new docs this run, "
                f"{processed / (time.perf_counter() - t0):.1f} docs/s"
            )

    print(f"[done] appended {processed} docs to {args.output}")
    engine.shutdown()


if __name__ == "__main__":
    main()
