import random

langs = ["fin_Latn", "eng_Latn", "swe_Latn",
         "cmn_Hans", "cmn_Hant", "fas_Arab", "ell_Grek", "pol_Latn", "deu_Latn", "tur_Latn"]
N = 50_000
rng = random.Random(0)

for lang in langs:
    res = []
    with open(f"{lang}_10M.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < N:
                res.append(line)
            else:
                j = rng.randint(0, i)
                if j < N:
                    res[j] = line
    with open(f"{lang}_50k.jsonl", "w", encoding="utf-8") as out:
        out.writelines(res)
    print(f"{lang}: wrote {len(res)} of {i + 1}", flush=True)
