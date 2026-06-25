import os

# ---------------------------------------------------------------------------
# env setup BEFORE imports — keep at top level (same as the working script)
# ---------------------------------------------------------------------------
CACHE_DIR = ".hf_cache"
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_HUB_CACHE"] = os.path.join(CACHE_DIR, "hub")

import argparse
import json

from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

# ===========================================================================
# Configuration
# ===========================================================================
MODEL = "google/gemma-4-26B-A4B-it"

CONTEXT_WINDOW = 5  # +-5 neighbouring segments
CHAR_CAP = 1000  # first N chars of each segment (target + context)
DOC_BATCH = 200  # how many documents to flatten into one generate() call
MAX_MODEL_LEN = 8192

MEDIUM_VALUES = ["written", "transcribed", "cannot_rate"]
TURN_VALUES = ["monologic", "dialogic"]
PURPOSE_VALUES = [
    "explaining",
    "supporting",
    "recounting",
    "directing",
    "evaluating",
    "expressing",
    "promoting",
    "creating",
]


# ===========================================================================
# JSON schema given to vLLM structured outputs.
#
# The grammar guarantees SHAPE and ALLOWED VALUES only. All six fields are
# present and nullable. The cannot_rate -> null logic and dropping of null
# keys is enforced deterministically in Python afterwards (post_process()).
# ===========================================================================
def nullable_enum(values):
    return {"type": ["string", "null"], "enum": values + [None]}


RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "mode_medium": nullable_enum(MEDIUM_VALUES),
        "mode_turn": nullable_enum(TURN_VALUES),
        "field_purpose": nullable_enum(PURPOSE_VALUES),
        "mode_medium_2": nullable_enum(MEDIUM_VALUES),
        "mode_turn_2": nullable_enum(TURN_VALUES),
        "field_purpose_2": nullable_enum(PURPOSE_VALUES),
    },
    "required": [
        "mode_medium",
        "mode_turn",
        "field_purpose",
        "mode_medium_2",
        "mode_turn_2",
        "field_purpose_2",
    ],
}


# ===========================================================================
# Prompt construction
# ===========================================================================
SYSTEM = (
    "You are an expert linguistic annotator. You label one TARGET text segment "
    "at a time along three dimensions, using the surrounding segments only as "
    "context. You respond with a single JSON object and nothing else."
)

CODEBOOK = """\
Identify the KIND of text each segment is — its situation and purpose of use —
not its topic, quality, or correctness. Judge each segment by what it is as
communication, not by whether it is well written, true, or on-topic.

Label the TARGET segment on three dimensions.

mode_medium (ALWAYS give a value). Decide in this order:
  STEP 1 — Is the segment a self-standing piece of communication, conveying a
  message on its own? If NOT, it is cannot_rate, however polished it looks:
  titles, headings, labels, navigation, menus, link lists, boilerplate,
  metadata, captions, dates/timestamps, bare fragments, or garbled text. A
  bullet/list item is cannot_rate UNLESS it is itself a complete statement.
  Test: read the item alone — if it does not convey a full message by itself,
  it is cannot_rate. EXCEPTION: if it is a fragment only because a sentence was
  split across segments, rate it as part of that sentence — this does NOT apply
  to dates, headings, labels, or list items. If mode_medium is cannot_rate, set
  the other five fields to null and stop.
  STEP 2 — Only if self-standing, choose:
    written      — composed as text to be read, even if casual, conversational,
                   or spoken-like in tone (comments, posts, messages, emails).
                   Informal or chatty style does NOT make it transcribed.
    transcribed  — a record of words actually spoken aloud: transcripts,
                   interviews, speeches, captions/subtitles, or text with
                   speaker labels or turn markers.

mode_turn:
  monologic — one author to a general audience (articles, guides, FAQs).
  dialogic  — others can reply (forums, comments, social posts, chats, interviews).

field_purpose:
  explaining — how something IS or WORKS; concepts, definitions, mechanisms.
  supporting — backing a claim with evidence, reasons, sources, or experience.
  recounting — relating specific REAL events in sequence as having actually
               happened (trip reports, incident writeups, true anecdotes).
  directing  — telling the reader how to do something, or what they should do.
  evaluating — reasoned assessment or argument where grounds are given.
  expressing — personal opinion, emotion, or stance as the point itself.
  promoting  — the author marketing their own product, service, or organization.
  creating   — literary or artistic text as primary content: poetry, lyrics,
               scripture, fiction, and narrative prose presented as a crafted
               or imaginative work. Narrative form alone is NOT recounting — if
               it reads as a fictional/literary artifact, it is creating.

Tie-breakers:
  - evaluating vs expressing: grounds given = evaluating; bare reaction = expressing.
  - explaining vs supporting: "how/what is X" = explaining; "why believe / evidence" = supporting.
  - recounting vs creating: events offered as really happened = recounting;
    narrative presented as literary/fiction = creating.
  - recounting vs expressing: events related for their own sake = recounting;
    events used as backstory for a present feeling, opinion, or plea = expressing.

Primary and optional secondary value:
  - The primary fields carry the value that fits best. Most segments have one
    main value per dimension.
  - The secondary fields (mode_medium_2, mode_turn_2, field_purpose_2) are
    OPTIONAL and rare. Set one only when the segment truly does two things on
    that dimension. Otherwise set the secondary to null. The three dimensions
    are independent.

Output a JSON object with exactly these keys:
  mode_medium, mode_turn, field_purpose,
  mode_medium_2, mode_turn_2, field_purpose_2
Use null for any field that does not apply."""


def cap(s):
    return s[:CHAR_CAP]


def build_context_block(segments, target_idx):
    """Render the target segment plus +-CONTEXT_WINDOW neighbours.

    Empty neighbours are included so the positional offsets stay truthful.
    Each segment is truncated to CHAR_CAP characters.
    """
    lo = max(0, target_idx - CONTEXT_WINDOW)
    hi = min(len(segments), target_idx + CONTEXT_WINDOW + 1)
    lines = []
    for i in range(lo, hi):
        offset = i - target_idx
        text = cap(segments[i])
        if offset == 0:
            lines.append(f">>> [TARGET] {text}")
        else:
            tag = f"[{offset:+d}]"
            lines.append(f"    {tag} {text}")
    return "\n".join(lines)


def make_prompt(llm, url, segments, target_idx):
    context_block = build_context_block(segments, target_idx)
    user = (
        f"{CODEBOOK}\n\n"
        f"URL of the page: {url}\n\n"
        f"Segments (the TARGET is marked with >>>; classify the TARGET only):\n"
        f"{context_block}"
    )
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ]
    tok = llm.get_tokenizer()
    try:
        return tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        # Some chat templates don't accept enable_thinking; fall back gracefully.
        return tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


# ===========================================================================
# Post-processing: enforce codebook logic, drop null keys
# ===========================================================================
def _clean_set(entry, medium_key, turn_key, purpose_key):
    """Apply cannot_rate -> null rule to one (primary or secondary) set,
    in place, then return nothing. Null fields are removed later."""
    if entry.get(medium_key) == "cannot_rate":
        entry[turn_key] = None
        entry[purpose_key] = None


def post_process(raw):
    """Take the model's raw dict, enforce logic, drop null keys, return a
    clean entry. The two sets (primary / secondary) are handled independently,
    matching the rule: cannot_rate in a set nulls that set's turn+purpose."""
    entry = dict(raw)

    # Enforce cannot_rate logic per set, independently.
    _clean_set(entry, "mode_medium", "mode_turn", "field_purpose")
    _clean_set(entry, "mode_medium_2", "mode_turn_2", "field_purpose_2")

    # Safety net: if the model emitted a value outside the allowed set, drop it.
    allowed = {
        "mode_medium": MEDIUM_VALUES,
        "mode_medium_2": MEDIUM_VALUES,
        "mode_turn": TURN_VALUES,
        "mode_turn_2": TURN_VALUES,
        "field_purpose": PURPOSE_VALUES,
        "field_purpose_2": PURPOSE_VALUES,
    }
    for k, vals in allowed.items():
        if entry.get(k) is not None and entry[k] not in vals:
            entry[k] = None

    # Drop null / missing keys (your "absent" choice for skipped fields).
    return {k: v for k, v in entry.items() if v is not None}


def empty_entry():
    """Entry for an empty (whitespace-only) segment: auto cannot_rate, nothing
    run through the model, others absent."""
    return {"mode_medium": "cannot_rate"}


# ===========================================================================
# IO helpers
# ===========================================================================
def split_segments(text):
    """Segments are newline-separated items in the 'text' field. Kept verbatim
    (including empty lines) so indices align 1:1 with llm_register."""
    return text.split("\n")


def count_lines(path):
    if not os.path.exists(path):
        return 0
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n


# ===========================================================================
# Main
# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="input JSONL path")
    ap.add_argument("--output", required=True, help="output JSONL path")
    ap.add_argument("--text-key", default="text")
    ap.add_argument("--url-key", default="u")
    ap.add_argument("--out-key", default="llm_register")
    ap.add_argument("--doc-batch", type=int, default=DOC_BATCH)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    args = ap.parse_args()

    # --- resume: count documents already completed in the output file --------
    done = count_lines(args.output)
    if done:
        print(
            f"[resume] {done} documents already in {args.output}; skipping them.",
            flush=True,
        )

    llm = LLM(
        model=MODEL,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=MAX_MODEL_LEN,
        download_dir=os.environ["HF_HUB_CACHE"],
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=128,  # JSON object with 6 short fields; generous headroom
        structured_outputs=StructuredOutputsParams(json=RESPONSE_SCHEMA),
    )

    # Append mode; resume relies on line count matching input order.
    out_f = open(args.output, "a", encoding="utf-8")

    buffer = []  # list of (raw_line_dict, segments) for the current doc batch
    processed = 0

    def flush_batch():
        """Annotate every non-empty segment across the buffered documents in a
        single generate() call, regroup per document, write completed docs."""
        nonlocal processed
        if not buffer:
            return

        # Build the flat list of prompts for all non-empty segments.
        prompts = []
        # index map: prompt position -> (doc_index_in_buffer, segment_index)
        index_map = []
        # pre-fill each doc's register with empty-entry placeholders
        registers = []
        for di, (line, segments) in enumerate(buffer):
            reg = [None] * len(segments)
            url = line.get(args.url_key, "")
            for si, seg in enumerate(segments):
                if seg.strip() == "":
                    reg[si] = empty_entry()  # no model call
                else:
                    index_map.append((di, si))
                    prompts.append(make_prompt(llm, url, segments, si))
            registers.append(reg)

        if prompts:
            outputs = llm.generate(prompts, sampling_params)
            for (di, si), out in zip(index_map, outputs):
                text = out.outputs[0].text.strip()
                try:
                    raw = json.loads(text)
                except json.JSONDecodeError:
                    # Grammar should prevent this; if it ever happens, mark
                    # cannot_rate rather than crash the whole batch.
                    raw = {"mode_medium": "cannot_rate"}
                registers[di][si] = post_process(raw)

        # Write completed documents (every segment now has an entry).
        for (line, segments), reg in zip(buffer, registers):
            assert all(e is not None for e in reg), "incomplete register"
            assert len(reg) == len(segments), "register/segment length mismatch"
            line[args.out_key] = reg
            out_f.write(json.dumps(line, ensure_ascii=False) + "\n")
        out_f.flush()
        os.fsync(out_f.fileno())  # durable: a crash won't lose flushed docs

        processed += len(buffer)
        print(
            f"[progress] wrote {processed} docs this run ({done + processed} total)",
            flush=True,
        )
        buffer.clear()

    with open(args.input, "r", encoding="utf-8") as in_f:
        for li, raw_line in enumerate(in_f):
            if li < done:
                continue  # already written in a previous run
            line = json.loads(raw_line)
            segments = split_segments(line.get(args.text_key, ""))
            buffer.append((line, segments))
            if len(buffer) >= args.doc_batch:
                flush_batch()

    flush_batch()  # write the final partial batch
    out_f.close()
    print(f"[done] processed {processed} documents this run.", flush=True)


if __name__ == "__main__":
    main()
