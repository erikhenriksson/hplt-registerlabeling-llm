#!/usr/bin/env python3
"""
Classify each newline-separated segment of each document into one of the
8 CORE main registers, using +-5 surrounding segments as context.

Built for very large inputs (millions of docs):
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
CONTEXT_RADIUS = 5  # how many segments on each side to show as context
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
    "You classify a single TARGET segment of a web page into exactly one of "
    "eight main registers, using the surrounding segments only as context.\n\n"
    "The eight registers are:\n"
    + "\n".join(f"  {k}: {v}" for k, v in REGISTERS.items())
    + "\n\nRespond with ONLY the two-letter code of the single best register "
    "for the TARGET segment. No explanation, no other text."
)


# ---- small building blocks ------------------------------------------------


def split_segments(text):
    """Newline-separated segments, stripped, empties dropped."""
    return [s.strip() for s in text.split("\n") if s.strip()]


def build_prompt(segments, target_idx):
    """The context window for one target segment: it plus +-CONTEXT_RADIUS neighbours."""
    lo = max(0, target_idx - CONTEXT_RADIUS)
    hi = min(len(segments), target_idx + CONTEXT_RADIUS + 1)

    lines = []
    for i in range(lo, hi):
        snippet = segments[i][:SEGMENT_CHAR_LIMIT]
        tag = "TARGET — classify THIS" if i == target_idx else "context"
        lines.append(f"[SEGMENT {i}] ({tag})")
        lines.append(snippet)

    window = "\n".join(lines)
    return (
        f"Below are segments {lo}-{hi - 1} of a web page. "
        f"Classify ONLY segment {target_idx} (the TARGET). "
        f"Use the others as context.\n\n"
        f"{window}\n\n"
        f"Register code for segment {target_idx}:"
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


def classify_chunk(engine, tok, sampling_params, chunk):
    """
    Classify every segment of every doc in `chunk`.
    Returns one list of register codes per doc, in the same order as `chunk`.
    """
    # Flatten: one prompt per segment, remembering which (doc, segment) it is.
    prompts, index = [], []
    for doc_i, d in enumerate(chunk):
        for seg_i in range(len(d["segments"])):
            chat = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(d["segments"], seg_i)},
            ]
            prompts.append(
                tok.apply_chat_template(
                    chat,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
            index.append((doc_i, seg_i))

    outputs = engine.generate(prompts, sampling_params)

    # Regroup: scatter the flat outputs back into per-doc lists.
    labels = [[None] * len(d["segments"]) for d in chunk]
    for (doc_i, seg_i), o in zip(index, outputs):
        labels[doc_i][seg_i] = o["text"].strip()
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
        # disable_cuda_graph=True,
    )
    sampling_params = {
        "temperature": 0.0,
        "max_new_tokens": 4,
        "regex": "(" + "|".join(LABELS) + ")",
    }

    # Stream input, skip what's done, process the rest in fixed-size chunks.
    todo = (d for d in read_docs(args.input) if d.get("id") not in done)

    processed = 0
    t0 = time.perf_counter()
    with open(args.output, "a") as out:
        for chunk in chunked(todo, DOC_CHUNK):
            labels = classify_chunk(engine, tok, sampling_params, chunk)
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
