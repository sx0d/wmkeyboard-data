#!/usr/bin/env python3
"""
Build the word-pair model the Pinyin decoder ranks with (WMKeyboard #580):
    # wmkeyboard cjk ngram v1
    word <TAB> count                 (unigram)
    word <TAB> word <TAB> count      (bigram: the second straight after the first)

Why it exists: the conversion pack (pinyin_v2.tsv) says how common a word is on
its own and nothing about what follows what. Ranked on that alone, `wbxh` came
out as 亡八喜欢 — two real words, so it beat 我不喜欢, which is four. Counts of
我→不 and 不→喜欢 are what make the everyday reading win.

Text: OpenSubtitles 2018 (Simplified Chinese, via OPUS; Lison & Tiedemann
2016), the same subtitles the zh_cn frequency list behind pinyin_v2.tsv was
counted from, plus the Leipzig Corpora Collection's Chinese web and news
sentences (CC BY 4.0), the source of the app's other n-gram packs. Subtitles
are most of it on purpose: they are people talking, and 我不喜欢 is what a
phone keyboard is asked for far more often than 外部性. News alone made
不→喜欢 a rare pair (#580).

Chinese is not written with spaces, so each run of Han characters is split into words first:
a unigram Viterbi over the conversion pack's own vocabulary, scored by the
zh_cn frequency list. Splitting with the pack's vocabulary is deliberate — the
decoder only ever proposes pack words, so a count for any other token could
never be looked up.

Usage:
  python3 tools/build_pinyin_ngrams.py --corpus A-sentences.txt [--corpus B-sentences.txt]
      [--pack cjk/pinyin_v2.tsv] [--freq data/zh_cn/zh_cn_full.txt.gz] [--out cjk/pinyin_ngrams.tsv]

A corpus file is Leipzig's `<name>-sentences.txt` (id <TAB> sentence) or plain
text, one sentence per line (OPUS `mono/zh_cn.txt.gz`); .gz is read directly.
"""

import argparse
import gzip
import math
import re
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CJK_DIR = SCRIPT_DIR.parent
REPO_DIR = CJK_DIR.parent

# Longest word the decoder builds (Lattice.Opts.maxWordUnits).
MAX_WORD = 6
# What survives into the pack. Below these a count is mostly segmentation
# noise, and the app holds every row in memory.
MIN_UNIGRAM = 2
MIN_BIGRAM = 3
MAX_UNIGRAMS = 200_000
MAX_BIGRAMS = 700_000

HAN_RUN = re.compile(r"[㐀-䶿一-鿿豈-﫿]+")

_vocab = {}
_unknown = 0.0


def _init(vocab, unknown):
    global _vocab, _unknown
    _vocab = vocab
    _unknown = unknown


def segment(run):
    """Most probable split of one Han run into vocabulary words."""
    n = len(run)
    best = [0.0] + [-math.inf] * n
    back = [0] * (n + 1)
    for i in range(n):
        if best[i] == -math.inf:
            continue
        base = best[i]
        for j in range(i + 1, min(n, i + MAX_WORD) + 1):
            w = run[i:j]
            score = _vocab.get(w)
            if score is None:
                # A single character the vocabulary lacks still has to be
                # stepped over; anything longer simply is not a word.
                if j != i + 1:
                    continue
                score = _unknown
            if base + score > best[j]:
                best[j] = base + score
                back[j] = i
    out = []
    j = n
    while j > 0:
        i = back[j]
        out.append(run[i:j])
        j = i
    out.reverse()
    return out


def count_lines(lines):
    uni = Counter()
    bi = Counter()
    for line in lines:
        for run in HAN_RUN.findall(line):
            words = segment(run)
            prev = None
            for w in words:
                known = w in _vocab
                if known:
                    uni[w] += 1
                if prev is not None and known:
                    bi[(prev, w)] += 1
                prev = w if known else None
    return uni, bi


def load_vocab(pack, freq_path):
    words = set()
    with open(pack, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and 0 < len(parts[1]) <= MAX_WORD and HAN_RUN.fullmatch(parts[1]):
                words.add(parts[1])
    counts = {}
    opener = gzip.open if str(freq_path).endswith(".gz") else open
    with opener(freq_path, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2 and parts[0] in words:
                try:
                    counts[parts[0]] = counts.get(parts[0], 0) + int(parts[1])
                except ValueError:
                    pass
    total = sum(counts.values()) + len(words)
    # Add-half smoothing, so a pack word the frequency list never saw is still
    # a word, just a rare one.
    vocab = {w: math.log((counts.get(w, 0) + 0.5) / total) for w in words}
    return vocab, math.log(0.05 / total)


def sentences(paths):
    for p in paths:
        opener = gzip.open if str(p).endswith(".gz") else open
        with opener(p, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                tab = line.find("\t")
                yield line[tab + 1:] if tab >= 0 else line


def chunks(it, size):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", action="append", required=True)
    ap.add_argument("--pack", default=str(CJK_DIR / "pinyin_v2.tsv"))
    ap.add_argument("--freq", default=str(REPO_DIR / "data" / "zh_cn" / "zh_cn_full.txt.gz"))
    ap.add_argument("--out", default=str(CJK_DIR / "pinyin_ngrams.tsv"))
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()

    vocab, unknown = load_vocab(args.pack, args.freq)
    print(f"vocabulary: {len(vocab)} words")
    uni = Counter()
    bi = Counter()
    with Pool(args.jobs, initializer=_init, initargs=(vocab, unknown)) as pool:
        for i, (u, b) in enumerate(pool.imap_unordered(count_lines, chunks(sentences(args.corpus), 20_000))):
            uni.update(u)
            bi.update(b)
            if i % 10 == 0:
                print(f"  chunk {i}: {len(uni)} words, {len(bi)} pairs")

    unigrams = [(w, c) for w, c in uni.most_common(MAX_UNIGRAMS) if c >= MIN_UNIGRAM]
    kept = {w for w, _ in unigrams}
    bigrams = [
        (p, c) for p, c in bi.most_common()
        if c >= MIN_BIGRAM and p[0] in kept and p[1] in kept
    ][:MAX_BIGRAMS]
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write("# wmkeyboard cjk ngram v1\n")
        f.write("# Word counts from OpenSubtitles 2018 (via OPUS) and the Leipzig Corpora Collection (CC BY 4.0),\n")
        f.write("# split with the pinyin pack's vocabulary. Built by tools/build_pinyin_ngrams.py.\n")
        # Sorted, so a rebuild from the same text is byte-identical.
        for w, c in sorted(unigrams):
            f.write(f"{w}\t{c}\n")
        for (a, b), c in sorted(bigrams):
            f.write(f"{a}\t{b}\t{c}\n")
    print(f"wrote {args.out}: {len(unigrams)} unigrams, {len(bigrams)} bigrams")


if __name__ == "__main__":
    main()
