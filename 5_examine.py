"""
Terminal viewer for inspecting register annotations (no curses required).

Shows each segment the way the LLM saw it — the target plus +-5 context
segments, each truncated to 1000 chars — alongside the model's annotation.
Navigate with single-key commands followed by Enter.

Usage:
    python3 view_annotations.py labeled.jsonl

Commands (type, then Enter):
    <Enter> or j   next segment
    k              previous segment
    n / p          next / previous document
    g N            jump to global segment index N
    f VALUE        show only segments whose annotation contains VALUE
    f              clear filter
    q              quit
"""

import json
import sys
import textwrap

CONTEXT_WINDOW = 5
CHAR_CAP = 1000

PRIMARY = ["mode_medium", "mode_turn", "field_purpose"]
SECONDARY = ["mode_medium_2", "mode_turn_2", "field_purpose_2"]

# ANSI colors (fall back to nothing if not a tty)
if sys.stdout.isatty():
    CYAN, YELLOW, GREEN, MAGENTA, DIM, BOLD, RESET = (
        "\033[36m",
        "\033[33m",
        "\033[32m",
        "\033[35m",
        "\033[2m",
        "\033[1m",
        "\033[0m",
    )
else:
    CYAN = YELLOW = GREEN = MAGENTA = DIM = BOLD = RESET = ""


def load(path):
    """Flatten the file into a list of per-segment view items."""
    items = []
    with open(path, encoding="utf-8") as f:
        for doc_i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            segments = d.get("text", "").split("\n")
            url = d.get("u", "")
            register = d.get("llm_register", [])
            for seg_i in range(len(segments)):
                ann = register[seg_i] if seg_i < len(register) else {}
                items.append(
                    {
                        "doc_i": doc_i,
                        "url": url,
                        "segments": segments,
                        "seg_i": seg_i,
                        "ann": ann,
                    }
                )
    return items


def build_context(item):
    segments = item["segments"]
    t = item["seg_i"]
    lo = max(0, t - CONTEXT_WINDOW)
    hi = min(len(segments), t + CONTEXT_WINDOW + 1)
    rows = []
    for i in range(lo, hi):
        offset = i - t
        text = segments[i][:CHAR_CAP]
        tag = "[TARGET]" if offset == 0 else f"[{offset:+d}]"
        rows.append((tag, text, offset == 0))
    return rows


def render(item, idx, total, vis_pos, vis_total, flt):
    width = 100
    out = []
    head = (
        f"{GREEN}{BOLD} doc {item['doc_i']}  seg {item['seg_i']}  "
        f"| segment {vis_pos}/{vis_total} (global {idx + 1}/{total})"
    )
    if flt:
        head += f"  | filter: {flt}"
    out.append(head + RESET)
    out.append(f" URL: {item['url']}")
    out.append("")
    out.append(f"{BOLD} CONTEXT (as seen by the model):{RESET}")
    for tag, text, is_target in build_context(item):
        color = CYAN + BOLD if is_target else MAGENTA
        body_attr = BOLD if is_target else DIM
        wrapped = textwrap.wrap(text, width) or [""]
        out.append(f"  {color}{tag:9s}{RESET} {body_attr}{wrapped[0]}{RESET}")
        for wl in wrapped[1:]:
            out.append(f"           {body_attr}{wl}{RESET}")
    out.append("")
    out.append(f"{YELLOW}{BOLD} ANNOTATION:{RESET}")
    shown = False
    for k in PRIMARY:
        if k in item["ann"]:
            out.append(f"   {YELLOW}{k:16s} {item['ann'][k]}{RESET}")
            shown = True
    sec = [k for k in SECONDARY if k in item["ann"]]
    if sec:
        out.append("")
        for k in sec:
            out.append(f"   {YELLOW}{k:16s} {item['ann'][k]}{RESET}")
        shown = True
    if not shown:
        out.append("   (no annotation)")
    return "\n".join(out)


def matches(item, flt):
    return (not flt) or (flt in item["ann"].values())


def main():
    if len(sys.argv) != 2:
        print("usage: python3 view_annotations.py labeled.jsonl")
        sys.exit(1)
    items = load(sys.argv[1])
    if not items:
        print("no segments found in file")
        sys.exit(1)

    idx = 0
    flt = None

    while True:
        vis = [i for i, it in enumerate(items) if matches(it, flt)] or list(
            range(len(items))
        )
        if idx not in vis:
            idx = min(vis, key=lambda v: abs(v - idx))
        item = items[idx]

        print("\n" * 2)
        print(render(item, idx, len(items), vis.index(idx) + 1, len(vis), flt))
        print(
            f"\n{DIM} [Enter]/j next  k prev  n/p doc  g N goto  "
            f"f VALUE filter  f clear  q quit{RESET}"
        )

        try:
            cmd = input(" > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if cmd in ("q", "quit"):
            break
        elif cmd in ("", "j"):
            cur = vis.index(idx)
            idx = vis[min(cur + 1, len(vis) - 1)]
        elif cmd == "k":
            cur = vis.index(idx)
            idx = vis[max(cur - 1, 0)]
        elif cmd == "n":
            d = item["doc_i"]
            nxt = [v for v in vis if items[v]["doc_i"] > d]
            if nxt:
                idx = nxt[0]
        elif cmd == "p":
            d = item["doc_i"]
            prev = [v for v in vis if items[v]["doc_i"] < d]
            if prev:
                tdoc = items[prev[-1]]["doc_i"]
                idx = next(v for v in prev if items[v]["doc_i"] == tdoc)
        elif cmd.startswith("g"):
            arg = cmd[1:].strip()
            try:
                n = int(arg) - 1
                if 0 <= n < len(items):
                    idx = n
                else:
                    print(" out of range")
            except ValueError:
                print(" usage: g N")
        elif cmd == "f":
            flt = None
        elif cmd.startswith("f "):
            flt = cmd[2:].strip() or None
        else:
            print(" unknown command")


if __name__ == "__main__":
    main()
