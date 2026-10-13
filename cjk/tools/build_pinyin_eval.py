#!/usr/bin/env python3
"""
Build the held-out evaluation set for the Pinyin decoder (WMKeyboard #580):
    pinyin <TAB> initials <TAB> hanzi <TAB> source

    pinyin    toneless readings run together, no apostrophes   (wobuxihuan)
    initials  first letter of each syllable; zh/ch/sh count as z/c/s   (wbxh)
    hanzi     the Simplified clause the reading should decode to     (我不喜欢)
    source    tatoeba | leipzig

Why it exists: pinyin_v2.tsv and pinyin_ngrams.tsv are both counted from
OpenSubtitles and Leipzig web/news text, so scoring the decoder on sentences
from those corpora measures memory, not ranking. These clauses come from text
neither pack ever saw:

  - Tatoeba Mandarin (cmn) sentences, CC BY 2.0 FR: short, conversational,
    the closest free stand-in for what people type on a phone.
  - Leipzig Corpora Collection zho_wikipedia_2018_10K, CC BY 4.0: written,
    encyclopaedic text, so the set is not only chat.

Filtering:
  - Simplified only. A sentence is dropped whole if any character is a
    Traditional-only form: a target of the app's OpenCC s2t.txt that differs
    from its key and is not itself a key (so 后 stays, 後 goes).
  - Each sentence is cut at every non-Han character into clauses; a clause of
    2..14 Han characters is kept when the pinyin pack's vocabulary splits it
    (the same Viterbi build_pinyin_ngrams.py uses) into known words with no
    unknown character left over.
  - Each word reads as its most frequent reading in pinyin_v2.tsv. Syllable
    boundaries come from the GB/T 16159 syllable inventory, one character per
    syllable; where a reading splits more than one way (fangan: fang|an or
    fan|gan) the split whose syllables are each a reading of their own
    character wins, then the one with fewer vowel-initial syllables (an
    unmarked reading implies no apostrophe).
  - Deduplicated on the hanzi, then sampled deterministically (seed 580).

Usage:
  python3 tools/build_pinyin_eval.py --tatoeba cmn_sentences.tsv[.bz2] \
      --leipzig zho_wikipedia_2018_10K/zho_wikipedia_2018_10K-sentences.txt
      [--app ../WMKeyboard] [--per-source 1500] [--out cjk/eval/pinyin_eval.tsv]

--app is the WM Keyboard checkout whose app/src/main/assets/dictionaries holds
s2t.txt and pinyin_syllables.txt (default: a sibling checkout, or $WMKB_APP).
"""

import argparse
import bz2
import gzip
import os
import random
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CJK_DIR = SCRIPT_DIR.parent
REPO_DIR = CJK_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import build_pinyin_ngrams as ngrams  # noqa: E402  (shared segmenter)

MIN_CLAUSE = 2
MAX_CLAUSE = 14
SEED = 580


def open_text(path):
    p = str(path)
    if p.endswith(".bz2"):
        return bz2.open(p, "rt", encoding="utf-8", errors="replace")
    if p.endswith(".gz"):
        return gzip.open(p, "rt", encoding="utf-8", errors="replace")
    return open(p, encoding="utf-8", errors="replace")


def load_traditional_only(s2t_path):
    keys = set()
    targets = set()
    with open(s2t_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or "\t" not in line:
                continue
            key, values = line.rstrip("\n").split("\t", 1)
            keys.add(key)
            for v in values.split():
                if v != key:
                    targets.add(v)
    return {t for t in targets if len(t) == 1} - keys


def load_syllables(path):
    with open(path, encoding="utf-8") as f:
        return {s.strip() for s in f if s.strip() and not s.startswith("#")}


def load_readings(pack):
    """word -> its most frequent reading; char -> every reading it has."""
    best = {}
    char_readings = {}
    with open(pack, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            reading, word = parts[0], parts[1]
            try:
                freq = int(parts[2])
            except ValueError:
                continue
            if word not in best or freq > best[word][1]:
                best[word] = (reading, freq)
            if len(word) == 1:
                char_readings.setdefault(word, set()).add(reading)
    return {w: r for w, (r, _) in best.items()}, char_readings


def splits(reading, n, syllables):
    """Every way to cut `reading` into exactly n inventory syllables."""
    out = []

    def go(i, acc):
        if len(acc) == n:
            if i == len(reading):
                out.append(list(acc))
            return
        for j in range(i + 1, min(len(reading), i + 6) + 1):
            s = reading[i:j]
            if s in syllables:
                acc.append(s)
                go(j, acc)
                acc.pop()

    go(0, [])
    return out


def syllabify(word, reading, syllables, char_readings):
    options = splits(reading, len(word), syllables)
    if not options:
        return None

    def rank(opt):
        own = sum(1 for c, s in zip(word, opt) if s in char_readings.get(c, ()))
        vowel = sum(1 for s in opt if s[0] in "aoe")
        return (-own, vowel)

    return min(options, key=rank)


def clauses(text, trad_only):
    if any(c in trad_only for c in text):
        return []
    return [r for r in ngrams.HAN_RUN.findall(text) if MIN_CLAUSE <= len(r) <= MAX_CLAUSE]


def to_row(clause, readings, syllables, char_readings):
    words = ngrams.segment(clause)
    sylls = []
    for w in words:
        r = readings.get(w)
        if w not in ngrams._vocab or r is None:
            return None
        s = syllabify(w, r, syllables, char_readings)
        if s is None:
            return None
        sylls.extend(s)
    return "".join(sylls), "".join(s[0] for s in sylls)


def tatoeba_sentences(path):
    with open_text(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3 and parts[1] == "cmn":
                yield parts[2]


def leipzig_sentences(path):
    with open_text(path) as f:
        for line in f:
            tab = line.find("\t")
            yield line[tab + 1:].rstrip("\n") if tab >= 0 else line.rstrip("\n")


def main():
    default_app = os.environ.get("WMKB_APP", str(REPO_DIR.parent / "WMKeyboard"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--tatoeba", required=True)
    ap.add_argument("--leipzig", required=True)
    ap.add_argument("--app", default=default_app)
    ap.add_argument("--pack", default=str(CJK_DIR / "pinyin_v2.tsv"))
    ap.add_argument("--freq", default=str(REPO_DIR / "data" / "zh_cn" / "zh_cn_full.txt.gz"))
    ap.add_argument("--per-source", type=int, default=1500)
    ap.add_argument("--out", default=str(CJK_DIR / "eval" / "pinyin_eval.tsv"))
    args = ap.parse_args()

    dicts = Path(args.app) / "app" / "src" / "main" / "assets" / "dictionaries"
    trad_only = load_traditional_only(dicts / "s2t.txt")
    syllables = load_syllables(dicts / "pinyin_syllables.txt")
    vocab, unknown = ngrams.load_vocab(args.pack, args.freq)
    ngrams._init(vocab, unknown)
    readings, char_readings = load_readings(args.pack)
    print(f"{len(trad_only)} Traditional-only chars, {len(syllables)} syllables, {len(vocab)} words")

    seen = set()
    picked = []
    for source, stream in (("tatoeba", tatoeba_sentences(args.tatoeba)),
                           ("leipzig", leipzig_sentences(args.leipzig))):
        sentences = dropped = 0
        pool = {}
        for text in stream:
            sentences += 1
            cs = clauses(text, trad_only)
            if not cs and any(c in trad_only for c in text):
                dropped += 1
            for c in cs:
                if c in seen or c in pool:
                    continue
                row = to_row(c, readings, syllables, char_readings)
                if row is not None:
                    pool[c] = row
        seen.update(pool)
        keys = sorted(pool)
        sample = random.Random(SEED).sample(keys, min(args.per_source, len(keys)))
        print(f"{source}: {sentences} sentences, {dropped} Traditional, {len(pool)} usable clauses, {len(sample)} kept")
        picked.extend((pool[c][0], pool[c][1], c, source) for c in sorted(sample))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write("# wmkeyboard cjk pinyin eval v1\n")
        f.write("# pinyin<TAB>initials<TAB>hanzi<TAB>source. Held-out clauses from Tatoeba (cmn, CC BY 2.0 FR)\n")
        f.write("# and Leipzig zho_wikipedia_2018_10K (CC BY 4.0). Built by tools/build_pinyin_eval.py.\n")
        for row in picked:
            f.write("\t".join(row) + "\n")
    print(f"wrote {args.out}: {len(picked)} rows")


if __name__ == "__main__":
    main()
