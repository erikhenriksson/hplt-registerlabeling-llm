import io, os, glob, random, zstandard

langs = ["fin_Latn", "eng_Latn", "swe_Latn",
         "cmn_Hans", "cmn_Hant", "fas_Arab", "ell_Grek", "pol_Latn", "deu_Latn", "tur_Latn"]
base = "/appl/local/openeurollm/training/catalogue/hplt/4.0/clean"
OUTDIR = "."
POOL = 10_000_000
SEED = 0

CHUNK_SIZE = 1 << 20
MAX_LINE_BYTES = 256 << 20      # guard against a truly pathological no-newline stream

def bin_of(path):
    return int(os.path.basename(path).split("_", 1)[0])

def iter_shard(path):
    # stream lines (bytes, no trailing \n) from one .zst shard, constant memory.
    # uses an index pointer instead of re-slicing buf, so no quadratic copying.
    dctx = zstandard.ZstdDecompressor()
    with open(path, "rb") as fh:
        reader = dctx.stream_reader(fh)
        buf = bytearray()
        start = 0
        while True:
            chunk = reader.read(CHUNK_SIZE)
            if not chunk:
                break
            buf.extend(chunk)
            while True:
                nl = buf.find(b"\n", start)
                if nl == -1:
                    break
                yield bytes(buf[start:nl])
                start = nl + 1
            if start:                  # drop consumed prefix so buf stays small
                del buf[:start]
                start = 0
            if len(buf) > MAX_LINE_BYTES:
                print(f"  [warn] {os.path.basename(path)}: dropping over-long region", flush=True)
                buf.clear()
        if buf:                        # trailing line, no final newline
            yield bytes(buf)

def count_lines(path):
    n = 0
    for _ in iter_shard(path):
        n += 1
    return n

def line_count_text(path):
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n

os.makedirs(OUTDIR, exist_ok=True)
rng = random.Random(SEED)

for lang in langs:
    out_path = os.path.join(OUTDIR, f"{lang}_10M.jsonl")

    if os.path.exists(out_path):
        existing = line_count_text(out_path)
        if existing >= POOL:
            print(f"[{lang}] SKIP: {out_path} already has {existing} lines", flush=True)
            continue
        print(f"[{lang}] re-doing: {out_path} has only {existing} lines, regenerating", flush=True)

    lang_dir = os.path.join(base, lang)
    shards = glob.glob(os.path.join(lang_dir, "*.jsonl.zst"))
    if not shards:
        print(f"[{lang}] WARNING: no shards in {lang_dir}", flush=True)
        continue

    bins = {}
    for s in shards:
        bins.setdefault(bin_of(s), []).append(s)

    plan = []
    remaining = POOL
    for b in sorted(bins, reverse=True):
        if remaining <= 0:
            break
        bin_shards = sorted(bins[b])

        # PASS 1: count lines across this bin's shards
        bin_total = sum(count_lines(p) for p in bin_shards)
        print(f"[{lang}] bin {b}: {bin_total} docs across {len(bin_shards)} shards "
              f"(need {remaining})", flush=True)

        if bin_total <= remaining:
            keep = None        # keep everything in this bin
            take = bin_total
        else:
            keep = set(rng.sample(range(bin_total), remaining))
            take = remaining

        plan.append((b, bin_shards, keep))
        remaining -= take

    written_target = POOL - remaining

    # PASS 2: stream chosen lines straight to disk
    tmp_path = out_path + ".tmp"
    written = 0
    with open(tmp_path, "wb") as out:
        for b, bin_shards, keep in plan:
            gidx = 0
            for p in bin_shards:
                for line in iter_shard(p):
                    if keep is None or gidx in keep:
                        out.write(line)
                        out.write(b"\n")
                        written += 1
                    gidx += 1
            print(f"[{lang}] bin {b}: running total {written}", flush=True)
    os.replace(tmp_path, out_path)
    print(f"[{lang}] DONE: wrote {written} docs to {out_path} (target {written_target})", flush=True)

print("done.", flush=True)
