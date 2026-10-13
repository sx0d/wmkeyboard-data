#!/usr/bin/env python3
"""
Build pinyin_ngrams_v2.tsv: the v1 word-pair pack (build_pinyin_ngrams.py) with
the 万象 grammar model (RIME-LMDG, amzxyz, CC BY 4.0) folded in (WMKeyboard #580).
Same format and header as v1, so the app reads it unchanged:
    # wmkeyboard cjk ngram v1
    word <TAB> count                 (unigram)
    word <TAB> word <TAB> count      (bigram)

Why: v1 is counted from ~2 GB of subtitles and Leipzig text, and 700k pairs is
where its counts run into noise. 万象 is a RIME octagram model built from a far
larger corpus; its 61M collocations know pairs v1 never saw.

How:
  1. tools/dump_rime_gram.c lists every (key, value) of the .gram file (a
     darts-clone double array; see the C file for the format). Keys are runs of
     2..8 characters, `$` marking a sentence end; values are
     max(0, int(ln(x) * 10000)).
  2. Each all-Han key is split with pinyin_v2.tsv's vocabulary by the same
     Viterbi as v1 (build_pinyin_ngrams.segment). A key that splits into exactly
     two words is evidence for that pair, one that is a single word evidence for
     that word. Keys with `$`, other scripts, or any other split are skipped.
  3. Calibration onto v1's scale, separately for each class of word lengths
     (1+1, 1+2, 2+1, 2+2 characters ...; for words, by length). Over the rows
     both have, a class's scale is the median of ln(v1 count) - value/10000;
     a 万象-only row becomes round(e^(value/10000 + offset)). 万象's values are
     not counts: they climb ~4.2 per key character, keys of five or more
     characters sit mostly at the 223327 ceiling, and even inside a class they
     agree with v1 only loosely (Spearman ~0.3 for pairs). So a class is used
     only if it has MIN_CLASS shared rows and Spearman >= MIN_RHO; in practice
     pairs of one- and two-character words, and two- and three-character
     words. The run prints every class's offset, spread and correlation.
  4. Output: every v1 unigram and bigram as is, plus 万象-only pairs (>=
     MIN_BIGRAM) and unigrams for pack words v1 lacked. A new pair is kept only
     when both words have a unigram row (the app's P(w2 | w1) divides by the
     first word's count) and never exceeds its first word's count. Capped at
     MAX_BIGRAMS rows and MAX_BYTES, dropping the weakest new pairs first.
     Layout as v1: all unigram rows, then all bigram rows, each sorted.

Usage:
  clang -O2 -o /tmp/dump_rime_gram tools/dump_rime_gram.c -lpthread
  /tmp/dump_rime_gram wanxiang-lts-zh-hans.gram /tmp/gram/part 8 && cat /tmp/gram/part.* > gram.tsv
  python3 tools/build_pinyin_ngrams_v2.py --gram-tsv gram.tsv --work /tmp/work
      [--base cjk/pinyin_ngrams.tsv] [--pack cjk/pinyin_v2.tsv]
      [--freq data/zh_cn/zh_cn_full.txt.gz] [--out cjk/pinyin_ngrams_v2.tsv]
"""

import argparse
import heapq
import math
import os
import statistics
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CJK_DIR = SCRIPT_DIR.parent
REPO_DIR = CJK_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import build_pinyin_ngrams as v1  # noqa: E402  (shared segmenter and vocabulary)

SCALE = 10000.0
MIN_BIGRAM = 3
MIN_UNIGRAM = v1.MIN_UNIGRAM
MAX_BIGRAMS = 1_200_000
MAX_BYTES = 25_000_000
# Fewest shared rows a word-length class needs for an offset of its own.
MIN_CLASS = 200
# Weakest rank agreement with v1 a class may have and still be used.
MIN_RHO = 0.25


def segment_lines(lines):
    pairs = []
    unis = []
    vocab = v1._vocab
    for line in lines:
        tab = line.find("\t")
        key = line[:tab]
        if not v1.HAN_RUN.fullmatch(key):
            continue
        words = v1.segment(key)
        if len(words) == 2:
            if words[0] in vocab and words[1] in vocab:
                pairs.append(f"{words[0]}\t{words[1]}\t{line[tab + 1:].strip()}\n")
        elif len(words) == 1 and key in vocab:
            unis.append(f"{key}\t{line[tab + 1:].strip()}\n")
    return pairs, unis


def chunks(f, size):
    buf = []
    for line in f:
        buf.append(line)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


def extract(gram_tsv, work, pack, freq, jobs):
    """Pass 1: gram keys -> pair and word evidence files in `work`."""
    pairs_path = work / "pairs.tsv"
    unis_path = work / "unis.tsv"
    if pairs_path.exists() and unis_path.exists():
        print(f"reusing {pairs_path} and {unis_path}")
        return pairs_path, unis_path
    vocab, unknown = v1.load_vocab(pack, freq)
    print(f"vocabulary: {len(vocab)} words")
    n_pairs = n_unis = 0
    with open(gram_tsv, encoding="utf-8") as f, \
            open(str(pairs_path) + ".part", "w", encoding="utf-8") as po, \
            open(str(unis_path) + ".part", "w", encoding="utf-8") as uo, \
            Pool(jobs, initializer=v1._init, initargs=(vocab, unknown)) as pool:
        for i, (p, u) in enumerate(pool.imap(segment_lines, chunks(f, 200_000), chunksize=1)):
            po.writelines(p)
            uo.writelines(u)
            n_pairs += len(p)
            n_unis += len(u)
            if i % 50 == 0:
                print(f"  {(i + 1) * 200_000:,} keys: {n_pairs:,} pairs, {n_unis:,} words")
    os.replace(str(pairs_path) + ".part", pairs_path)
    os.replace(str(unis_path) + ".part", unis_path)
    print(f"evidence: {n_pairs:,} pairs, {n_unis:,} words")
    return pairs_path, unis_path


def load_base(path):
    header = []
    uni = {}
    bi = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                header.append(line)
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2:
                uni[parts[0]] = int(parts[1])
            elif len(parts) == 3:
                bi[(parts[0], parts[1])] = int(parts[2])
    return header, uni, bi


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=v.__getitem__)
        r = [0] * len(v)
        for i, j in enumerate(order):
            r[j] = i
        return r
    rx, ry = ranks(xs), ranks(ys)
    m = (len(xs) - 1) / 2
    var = sum((a - m) ** 2 for a in rx)
    return sum((a - m) * (b - m) for a, b in zip(rx, ry)) / var if var else 0.0


def spread(xs):
    xs = sorted(xs)
    n = len(xs)
    q = lambda p: xs[min(n - 1, int(p * n))]  # noqa: E731
    return statistics.median(xs), q(0.25), q(0.75), q(0.75) - q(0.25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gram-tsv", required=True, help="dump_rime_gram output, key<TAB>value")
    ap.add_argument("--work", required=True, help="scratch directory for the evidence files")
    ap.add_argument("--base", default=str(CJK_DIR / "pinyin_ngrams.tsv"))
    ap.add_argument("--pack", default=str(CJK_DIR / "pinyin_v2.tsv"))
    ap.add_argument("--freq", default=str(REPO_DIR / "data" / "zh_cn" / "zh_cn_full.txt.gz"))
    ap.add_argument("--out", default=str(CJK_DIR / "pinyin_ngrams_v2.tsv"))
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    pairs_path, unis_path = extract(args.gram_tsv, work, args.pack, args.freq, args.jobs)
    _, uni, bi = load_base(args.base)
    print(f"v1: {len(uni):,} words, {len(bi):,} pairs")

    # Calibration samples: pairs and words both sides have, grouped by the
    # character lengths of the words. One global offset is useless: 万象's
    # values climb ~4.2 (x65) per character of key, which leaves them and v1
    # uncorrelated overall (Spearman ~ -0.05). Within a class they agree, if
    # loosely, but only for some classes: keys of five or more characters are
    # mostly pinned at the model's ceiling (223327), and pairs with a
    # three-character word do not track v1 at all. A class is used only when it
    # has MIN_CLASS shared rows and a Spearman correlation of at least MIN_RHO.
    pair_samples = defaultdict(list)
    with open(pairs_path, encoding="utf-8") as f:
        for line in f:
            a, b, v = line.rstrip("\n").split("\t")
            c = bi.get((a, b))
            if c is not None:
                pair_samples[(len(a), len(b))].append((int(v) / SCALE, math.log(c)))
    wanx_uni = {}
    with open(unis_path, encoding="utf-8") as f:
        for line in f:
            w, v = line.rstrip("\n").split("\t")
            wanx_uni[w] = max(wanx_uni.get(w, 0), int(v))
    uni_samples = defaultdict(list)
    for w, v in wanx_uni.items():
        if w in uni:
            uni_samples[len(w)].append((v / SCALE, math.log(uni[w])))

    def fit(groups, label, name):
        """Offset (median of ln(v1) - value/10000) for each class that is usable."""
        everything = [y - x for g in groups.values() for x, y in g]
        med, _, _, iqr = spread(everything)
        print(f"{label}: {len(everything):,} shared; one offset {med:+.3f} IQR {iqr:.3f}, Spearman "
              f"{spearman([x for g in groups.values() for x, _ in g], [y for g in groups.values() for _, y in g]):.2f}")
        offsets = {}
        for k, g in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            if len(g) < 50:
                continue
            rho = spearman([x for x, _ in g], [y for _, y in g])
            med, _, _, iqr = spread([y - x for x, y in g])
            ok = len(g) >= MIN_CLASS and rho >= MIN_RHO
            if ok:
                offsets[k] = med
            print(f"    {name(k)}: {len(g):>7,} shared, offset {med:+.3f} (x{math.exp(med):.3g}), "
                  f"IQR {iqr:.3f}, Spearman {rho:.2f}{'' if ok else '  -> unused'}")
        return offsets

    pair_off = fit(pair_samples, "pairs", lambda k: f"{k[0]}+{k[1]} chars")
    uni_off = fit(uni_samples, "words", lambda k: f"{k} chars")

    # New words first, so new pairs can be anchored on them.
    new_uni = {}
    for w, v in wanx_uni.items():
        if w not in uni and len(w) in uni_off:
            c = round(math.exp(v / SCALE + uni_off[len(w)]))
            if c >= MIN_UNIGRAM:
                new_uni[w] = c
    all_uni = dict(uni)
    all_uni.update(new_uni)

    def scaled(a, b, x):
        off = pair_off.get((len(a), len(b)))
        if off is None:
            return 0
        # Never above the first word's own count: P(b | a) <= 1 in the app.
        return min(all_uni[a], round(math.exp(x + off)))

    budget = MAX_BIGRAMS - len(bi)
    cand = []  # min-heap of (count, a, b), the strongest `budget` new pairs
    seen_new = 0
    with open(pairs_path, encoding="utf-8") as f:
        for line in f:
            a, b, v = line.rstrip("\n").split("\t")
            if (a, b) in bi or a not in all_uni or b not in all_uni:
                continue
            c = scaled(a, b, int(v) / SCALE)
            if c < MIN_BIGRAM:
                continue
            seen_new += 1
            item = (c, a, b)
            if len(cand) < budget:
                heapq.heappush(cand, item)
            elif item > cand[0]:
                heapq.heapreplace(cand, item)
    print(f"万象-only pairs >= {MIN_BIGRAM}: {seen_new:,}; room for {budget:,}")

    def render(new_pairs):
        # v1 layout: every unigram row, then every bigram row (the app indexes
        # successors assuming that order). Every word in a pair gets a unigram
        # row; a missing one is added at no less than its largest pair count.
        merged = dict(bi)
        merged.update(((a, b), c) for c, a, b in new_pairs)
        words = dict(all_uni)
        for (a, b), c in merged.items():
            for w in (a, b):
                if w not in all_uni:
                    words[w] = max(words.get(w, MIN_UNIGRAM), c)
        rows = [f"{w}\t{c}\n" for w, c in sorted(words.items())]
        rows += [f"{a}\t{b}\t{c}\n" for (a, b), c in sorted(merged.items())]
        return rows, len(merged), len(words)

    header = [
        "# wmkeyboard cjk ngram v1\n",
        "# Word counts from OpenSubtitles 2018 (via OPUS) and the Leipzig Corpora Collection (CC BY 4.0),\n",
        "# split with the pinyin pack's vocabulary, plus pairs from the RIME-LMDG 万象 grammar model\n",
        "# (amzxyz, CC BY 4.0). Built by tools/build_pinyin_ngrams_v2.py.\n",
    ]
    new_pairs = sorted(cand, reverse=True)
    while True:
        rows, n_bi, n_words = render(new_pairs)
        size = sum(len(r.encode("utf-8")) for r in header + rows)
        if size <= MAX_BYTES:
            break
        # Over the byte cap: drop the weakest new pairs (~14 bytes a row) and retry.
        excess_rows = max(1000, int((size - MAX_BYTES) / 14) + 1000)
        new_pairs = new_pairs[:-excess_rows]
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(header)
        f.writelines(rows)
    print(f"wrote {args.out}: {n_words:,} words ({n_words - len(uni):,} new), "
          f"{n_bi:,} pairs ({len(new_pairs):,} new), {size:,} bytes")


if __name__ == "__main__":
    main()
