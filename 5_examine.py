"""
Terminal viewer for inspecting register annotations.

Shows each segment the way the LLM saw it — the target plus +-5 context
segments, each truncated to 1000 chars — alongside the annotation the model
produced for that segment.

Usage:
    python3 view_annotations.py labeled.jsonl

Keys:
    j / down / space   next segment
    k / up             previous segment
    n                  next document
    p                  previous document
    g                  jump to a segment index (then type number, Enter)
    f                  filter: only show segments matching a label (e.g. "transcribed")
    F                  clear filter
    q                  quit
"""

import sys
import json
import curses
import textwrap

CONTEXT_WINDOW = 5
CHAR_CAP = 1000

PRIMARY = ["mode_medium", "mode_turn", "field_purpose"]
SECONDARY = ["mode_medium_2", "mode_turn_2", "field_purpose_2"]


def load(path):
    """Flatten the file into a list of (doc_index, url, segments, seg_index, ann)."""
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
            for seg_i, seg in enumerate(segments):
                ann = register[seg_i] if seg_i < len(register) else {}
                items.append({
                    "doc_i": doc_i,
                    "url": url,
                    "segments": segments,
                    "seg_i": seg_i,
                    "ann": ann,
                })
    return items


def fmt_ann(ann):
    """Render the annotation dict as readable lines, primary then secondary."""
    lines = []
    for k in PRIMARY:
        if k in ann:
            lines.append(f"  {k:16s} {ann[k]}")
    sec = [k for k in SECONDARY if k in ann]
    if sec:
        lines.append("")
        for k in sec:
            lines.append(f"  {k:16s} {ann[k]}")
    if not lines:
        lines.append("  (no annotation)")
    return lines


def build_context(item):
    """Reproduce the +-CONTEXT_WINDOW window exactly as the prompt built it."""
    segments = item["segments"]
    t = item["seg_i"]
    lo = max(0, t - CONTEXT_WINDOW)
    hi = min(len(segments), t + CONTEXT_WINDOW + 1)
    rows = []
    for i in range(lo, hi):
        offset = i - t
        text = segments[i][:CHAR_CAP]
        is_target = (offset == 0)
        tag = "[TARGET]" if is_target else f"[{offset:+d}]"
        rows.append((tag, text, is_target))
    return rows


def matches_filter(item, flt):
    if not flt:
        return True
    return flt in item["ann"].values()


def main(stdscr, items):
    curses.curs_set(0)
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_CYAN, -1)     # target marker
    curses.init_pair(2, curses.COLOR_YELLOW, -1)   # annotation
    curses.init_pair(3, curses.COLOR_GREEN, -1)    # header
    curses.init_pair(4, curses.COLOR_MAGENTA, -1)  # context tags

    idx = 0
    flt = None
    status = ""

    def visible_indices():
        return [i for i, it in enumerate(items) if matches_filter(it, flt)]

    while True:
        vis = visible_indices()
        if not vis:
            vis = list(range(len(items)))  # filter matched nothing; show all
        if idx not in vis:
            # snap to nearest visible
            idx = min(vis, key=lambda v: abs(v - idx))

        item = items[idx]
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        y = 0

        pos_in_vis = vis.index(idx) + 1
        header = (f" doc {item['doc_i']}  seg {item['seg_i']}  "
                  f"| segment {pos_in_vis}/{len(vis)} "
                  f"(global {idx + 1}/{len(items)})")
        if flt:
            header += f"  | filter: {flt}"
        stdscr.addnstr(y, 0, header.ljust(w), w, curses.color_pair(3) | curses.A_BOLD)
        y += 1
        stdscr.addnstr(y, 0, (" URL: " + item["url"])[:w], w)
        y += 2

        # context block
        stdscr.addnstr(y, 0, " CONTEXT (as seen by the model):", w,
                       curses.A_BOLD)
        y += 1
        for tag, text, is_target in build_context(item):
            if y >= h - 8:
                stdscr.addnstr(y, 0, "  … (context truncated to fit screen)", w)
                y += 1
                break
            tag_attr = (curses.color_pair(1) | curses.A_BOLD) if is_target \
                else curses.color_pair(4)
            stdscr.addnstr(y, 2, tag, 10, tag_attr)
            wrapped = textwrap.wrap(text, max(10, w - 14)) or [""]
            for j, wl in enumerate(wrapped):
                if y >= h - 8:
                    break
                prefix = 13
                attr = curses.A_BOLD if is_target else curses.A_DIM
                stdscr.addnstr(y, prefix, wl, w - prefix, attr)
                y += 1
        y += 1

        # annotation block
        if y < h - 2:
            stdscr.addnstr(y, 0, " ANNOTATION:", w,
                           curses.color_pair(2) | curses.A_BOLD)
            y += 1
            for ln in fmt_ann(item["ann"]):
                if y >= h - 1:
                    break
                stdscr.addnstr(y, 0, ln, w, curses.color_pair(2))
                y += 1

        # status line
        help_line = " j/k move  n/p doc  g goto  f filter  F clear  q quit"
        stdscr.addnstr(h - 1, 0, (status or help_line).ljust(w)[:w], w,
                       curses.A_REVERSE)

        c = stdscr.getch()
        status = ""

        if c in (ord("q"),):
            break
        elif c in (ord("j"), curses.KEY_DOWN, ord(" ")):
            cur = vis.index(idx)
            idx = vis[min(cur + 1, len(vis) - 1)]
        elif c in (ord("k"), curses.KEY_UP):
            cur = vis.index(idx)
            idx = vis[max(cur - 1, 0)]
        elif c == ord("n"):
            # next segment belonging to a later document
            d = item["doc_i"]
            for v in vis:
                if items[v]["doc_i"] > d:
                    idx = v
                    break
        elif c == ord("p"):
            d = item["doc_i"]
            prev = [v for v in vis if items[v]["doc_i"] < d]
            if prev:
                # first segment of the immediately previous document
                target_doc = items[prev[-1]]["doc_i"]
                for v in prev:
                    if items[v]["doc_i"] == target_doc:
                        idx = v
                        break
        elif c == ord("g"):
            curses.echo()
            curses.curs_set(1)
            stdscr.addnstr(h - 1, 0, " goto global index: ".ljust(w), w,
                           curses.A_REVERSE)
            stdscr.move(h - 1, 20)
            try:
                s = stdscr.getstr(h - 1, 20, 12).decode().strip()
                n = int(s) - 1
                if 0 <= n < len(items):
                    idx = n
                    status = f" jumped to {n + 1}"
                else:
                    status = " out of range"
            except (ValueError, KeyboardInterrupt):
                status = " invalid number"
            curses.noecho()
            curses.curs_set(0)
        elif c == ord("f"):
            curses.echo()
            curses.curs_set(1)
            stdscr.addnstr(h - 1, 0, " filter by label value: ".ljust(w), w,
                           curses.A_REVERSE)
            try:
                s = stdscr.getstr(h - 1, 25, 30).decode().strip()
                flt = s or None
                status = f" filter set: {flt}" if flt else " filter cleared"
            except KeyboardInterrupt:
                pass
            curses.noecho()
            curses.curs_set(0)
        elif c == ord("F"):
            flt = None
            status = " filter cleared"


def cli():
    if len(sys.argv) != 2:
        print("usage: python3 view_annotations.py labeled.jsonl")
        sys.exit(1)
    items = load(sys.argv[1])
    if not items:
        print("no segments found in file")
        sys.exit(1)
    curses.wrapper(main, items)


if __name__ == "__main__":
    cli()