#!/usr/bin/env python3
"""
Convert CC-CEDICT into a TSV file for WMKeyboard's ConversionDictionary format:
    reading <TAB> word <TAB> freq

Where:
  - reading: toneless Pinyin of the whole entry with syllables concatenated
    (lowercase ASCII, no spaces, no tone marks/digits, no apostrophes).
  - word: Simplified Hanzi character or word/phrase.
  - freq: integer frequency (higher = commoner), 1-1000.

CC-CEDICT carries no frequencies, so they come from the zh_cn frequency list in
data/ (hermitdave/FrequencyWords, OpenSubtitles 2018). An earlier version of
this script seeded each run from its own previous output and otherwise scored
rows by their line position in CC-CEDICT, which is sorted by character — so
the column was close to noise: 齐 sat 113th of 119 under `qi`, below 稽 and 鳍,
and a third of the 3,000 commonest characters were outside the top ten for
their own reading (issue #633).

Usage:
  python3 tools/build_cedict_pinyin.py [--cedict PATH] [--corpus PATH] [--out PATH]
"""

import argparse
import gzip
import hashlib
import math
import re
import urllib.request
from collections import defaultdict
from pathlib import Path

CEDICT_URL = "https://www.mdbg.net/chinese/export/cedict/cedict_1_0_ts_utf-8_mdbg.txt.gz"

SCRIPT_DIR = Path(__file__).resolve().parent
CJK_DIR = SCRIPT_DIR.parent
REPO_DIR = CJK_DIR.parent
# pinyin_v2.tsv, not pinyin.tsv: the app pins each pack by SHA-256, so a
# rebuilt pack goes out under a new name and the old file stays byte-for-byte
# as it was for the app versions that pinned it. Bump the name again for the
# next rebuild that changes the bytes, and repin it in CjkDictCatalog.
DEFAULT_OUT = CJK_DIR / "pinyin_v2.tsv"
# Simplified-script corpus counts (hermitdave/FrequencyWords, OpenSubtitles
# 2018), the same family of list build_jyutping_dict.py reads for Traditional.
DEFAULT_CORPUS = REPO_DIR / "data" / "zh_cn" / "zh_cn_full.txt.gz"

# Frequency bands, as in build_jyutping_dict.py. Kept disjoint so a row with
# corpus evidence always outranks one without, whatever its length.
CORPUS_MIN, CORPUS_MAX = 101, 1000  # attested in the corpus, log-scaled by count
TAIL_MIN, TAIL_MAX = 1, 100         # CC-CEDICT only: no frequency evidence

# How much in-word evidence a single character needs before its own standalone
# count is believed. The corpus is dirty at the rare end — 琌 is listed 22168
# times standalone and 硂 11983, yet neither appears in a single real word —
# so a lone character's count only counts once real words vouch for it.
MIN_CHAR_EVIDENCE = 50

# Standalone-only characters that are nonetheless real: interjections and
# sentence particles (嗯 69k, 哦 107k, 呃, 喔, 咦, 诶) never sit inside a word,
# so the in-word test above would throw them away with the debris. Their
# CC-CEDICT glosses say what they are, and debris glosses (door latch, cliff,
# bamboo mold) never match.
FUNCTION_GLOSS = re.compile(
    r"interjection|exclamation|particle|\bonom\.|\binterj|"
    r"expression of surprise|indicate realization|\bhey\b|\boh\b|\bwow\b|\bsigh\b|\bgrunt",
    re.IGNORECASE,
)

# Hanzi: Unified Ideographs, Extension A, compatibility, and the astral
# extensions (Python strings hold astral code points directly, not surrogates).
_HAN_RANGES = (
    (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF),
    (0x20000, 0x2A6DF), (0x2A700, 0x2EBEF), (0x2F800, 0x2FA1F),
)

# Tone marks to plain ascii map
TONE_MAP = str.maketrans(
    "āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜĀÁǍÀĒÉĚÈĪÍǏÌŌÓǑÒŪÚǓÙǕǗǙǛ",
    "aaaaeeeeiiiioooouuuuvvvvaaaaeeeeiiiioooouuuuvvvv"
)


def is_han(word: str) -> bool:
    return bool(word) and all(any(lo <= ord(c) <= hi for lo, hi in _HAN_RANGES) for c in word)


def clean_syllable(raw: str) -> str:
    """One CC-CEDICT pinyin syllable to toneless lowercase ASCII."""
    s = raw.lower().replace("u:", "v").replace("ü", "v")
    s = s.translate(TONE_MAP)
    s = re.sub(r"[0-9]", "", s)
    return re.sub(r"[^a-z]", "", s)


def fetch_or_load_cedict(cedict_path: Path = None) -> str:
    """Loads CC-CEDICT text from local file or downloads from official MDBG URL."""
    if cedict_path and cedict_path.exists():
        print(f"Reading CC-CEDICT from local file: {cedict_path}")
        if str(cedict_path).endswith(".gz"):
            with gzip.open(cedict_path, "rt", encoding="utf-8") as f:
                return f.read()
        return cedict_path.read_text(encoding="utf-8")

    print(f"Downloading CC-CEDICT from {CEDICT_URL}...")
    req = urllib.request.Request(CEDICT_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        return gzip.decompress(resp.read()).decode("utf-8")


def parse_cedict(content: str):
    """
    CC-CEDICT entries as (word, syllables, gloss), Simplified headword, toneless
    syllables. Syllables stay split so a character can be paired with the one
    it is read as; `syllables` is None when they don't line up one per
    character (a handful of entries spell a character with two syllables).
    """
    out = []
    for line in content.splitlines():
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^(\S+)\s+(\S+)\s+\[(.*?)\]\s+/(.*)/$", line.strip())
        if not m:
            continue
        _trad, simp, py, gloss = m.groups()
        if not is_han(simp):
            continue
        syls = [s for s in (clean_syllable(p) for p in py.split()) if s]
        if not syls:
            continue
        out.append((simp, syls if len(syls) == len(simp) else None, "".join(syls), gloss))
    return out


def load_corpus_counts(corpus_path: Path) -> dict:
    """Loads `word count` pairs from a gzipped frequency list, Han headwords only."""
    if not corpus_path.exists():
        raise SystemExit(
            f"corpus not found at {corpus_path}; without it every row falls to the "
            "no-evidence band and candidate order is arbitrary. Pass --corpus."
        )
    print(f"Loading corpus frequencies from {corpus_path}...")
    counts = {}
    with gzip.open(corpus_path, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 2 or not is_han(parts[0]):
                continue
            try:
                n = int(parts[1])
            except ValueError:
                continue
            if n > 0:
                counts[parts[0]] = max(counts.get(parts[0], 0), n)
    print(f"  {len(counts)} Han words with counts")
    return counts


class Scorer:
    """
    Turns a (word, reading) row into a frequency.

    A multi-character word is self-validating — corpus debris almost never
    forms one — so its own corpus count is used as is.

    A single character is scored per *reading*, as a count on the same scale as
    the words so the two stay comparable in the decoder:

        count(c, r) = inword(c) * share(c, r) + alone(c) * alone_share(c, r)

    - `inword(c)` sums the counts of every corpus word containing c. It cannot
      be faked by debris (琌 scores 13, 齐 5068) and it is what lifts 议, 际
      and 集 — characters that almost never stand alone but are everywhere
      inside words — from the bottom of `yi`/`ji` to the top.
    - `share(c, r)` splits that across c's readings by how the CC-CEDICT words
      in the corpus actually read it: 行 is hang in 银行 and xing in 不行, and
      each word's count goes to the reading it uses.
    - `alone(c)` is c's own standalone count, believed only once words vouch
      for the character (MIN_CHAR_EVIDENCE) or CC-CEDICT calls it an
      interjection or particle. Standalone use of a character with a function
      reading is that reading — 的 alone is de, not the di of 目的 — so it goes
      to those readings; otherwise it follows `share`.
    """

    def __init__(self, corpus: dict, entries: list):
        self.corpus = corpus

        self.inword = defaultdict(int)
        for word, count in corpus.items():
            if len(word) >= 2:
                for ch in set(word):
                    self.inword[ch] += count

        # Per-character reading weights from the CC-CEDICT words the corpus has.
        aligned = defaultdict(lambda: defaultdict(int))
        seen = set()
        self.readings = defaultdict(set)
        self.function = defaultdict(set)
        for word, syls, reading, gloss in entries:
            if len(word) == 1:
                self.readings[word].add(reading)
                if FUNCTION_GLOSS.search(gloss):
                    self.function[word].add(reading)
                continue
            count = corpus.get(word, 0)
            if not count or syls is None or (word, reading) in seen:
                continue
            seen.add((word, reading))
            for ch, syl in zip(word, syls):
                aligned[ch][syl] += count
        self.aligned = aligned

        self.char_count = {}
        for ch, readings in self.readings.items():
            for r in readings:
                self.char_count[(ch, r)] = self._char_count(ch, r)

        word_counts = [c for w, c in corpus.items() if len(w) >= 2]
        all_counts = word_counts + [c for c in self.char_count.values() if c >= 1]
        self.lo = max(1, min(all_counts))
        self.hi = max(all_counts)

    def alone(self, ch: str) -> int:
        own = self.corpus.get(ch, 0)
        if self.inword.get(ch, 0) >= MIN_CHAR_EVIDENCE or self.function.get(ch):
            return own
        return 0

    def share(self, ch: str, reading: str) -> float:
        """Fraction of [ch]'s in-word use read as [reading], add-one smoothed."""
        readings = self.readings[ch]
        weights = self.aligned.get(ch, {})
        total = sum(weights.get(r, 0) for r in readings)
        return (weights.get(reading, 0) + 1) / (total + len(readings))

    def alone_share(self, ch: str, reading: str) -> float:
        function = self.function.get(ch)
        if not function:
            return self.share(ch, reading)
        if reading not in function:
            return 0.0
        total = sum(self.share(ch, r) for r in function)
        return self.share(ch, reading) / total

    def _char_count(self, ch: str, reading: str) -> float:
        return (self.inword.get(ch, 0) * self.share(ch, reading)
                + self.alone(ch) * self.alone_share(ch, reading))

    def count(self, word: str, reading: str) -> float:
        if len(word) == 1:
            return self.char_count.get((word, reading), 0.0)
        return self.corpus.get(word, 0)

    def char_strength(self, ch: str) -> float:
        """0-1: how common [ch] is, from its in-word evidence alone."""
        n = self.inword.get(ch, 0)
        if n < 1:
            return 0.0
        return min(1.0, math.log(n) / math.log(self.hi))

    def score(self, word: str, reading: str) -> int:
        count = self.count(word, reading)
        if count >= 1:
            # Log rather than linear because word frequency is Zipfian: on a
            # linear scale 的 alone would flatten everything below the top
            # hundred into one value.
            t = (math.log(count) - math.log(self.lo)) / (math.log(self.hi) - math.log(self.lo))
            return CORPUS_MIN + int(round((CORPUS_MAX - CORPUS_MIN) * max(0.0, min(1.0, t))))
        # No evidence for the row itself. Shorter words first, since they
        # compete for shorter buffers where being passed over is most visible;
        # then words built of commoner characters, so 一石二鸟 outranks an
        # idiom of characters nobody has typed this year.
        length_part = {1: 1.0, 2: 0.8, 3: 0.6, 4: 0.45}.get(len(word), 0.3)
        char_part = min(self.char_strength(ch) for ch in word)
        scale = length_part * (0.3 + 0.7 * char_part)
        return max(TAIL_MIN, min(TAIL_MAX, TAIL_MIN + int(round((TAIL_MAX - TAIL_MIN) * scale))))


def convert_cedict(entries: list, scorer: Scorer) -> list:
    """(reading, word, freq) rows, sorted by reading then freq descending then word."""
    entry_map = {}
    for word, _syls, reading, _gloss in entries:
        key = (reading, word)
        if key not in entry_map:
            entry_map[key] = scorer.score(word, reading)
    return [
        (r, w, f)
        for (r, w), f in sorted(entry_map.items(), key=lambda x: (x[0][0], -x[1], x[0][1]))
    ]


def main():
    parser = argparse.ArgumentParser(description="Convert CC-CEDICT to ConversionDictionary TSV format.")
    parser.add_argument("--cedict", type=Path, help="Path to local cedict_1_0_ts_utf-8_mdbg.txt.gz or .txt file")
    parser.add_argument(
        "--corpus", type=Path, default=DEFAULT_CORPUS,
        help="Gzipped Simplified-script `word count` frequency list",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output TSV file path")
    args = parser.parse_args()

    corpus = load_corpus_counts(args.corpus)
    entries = parse_cedict(fetch_or_load_cedict(args.cedict))
    print(f"  {len(entries)} CC-CEDICT entries with Han headwords")
    scorer = Scorer(corpus, entries)
    rows = convert_cedict(entries, scorer)

    out_lines = [
        "# CC-CEDICT Chinese-English Dictionary (CC BY-SA 4.0)",
        "# Frequencies derived from the zh_cn frequency list",
        "# (hermitdave/FrequencyWords, OpenSubtitles 2018, CC BY-SA 4.0)",
        "# Format: reading\tword\tfreq",
    ]
    out_lines.extend(f"{r}\t{w}\t{f}" for r, w, f in rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    data = args.out.read_bytes()
    evidenced = sum(1 for _r, _w, f in rows if f >= CORPUS_MIN)
    print(f"Wrote {len(rows)} rows to {args.out}")
    print(f"  {CORPUS_MIN}-{CORPUS_MAX} corpus-attested: {evidenced}")
    print(f"  {TAIL_MIN}-{TAIL_MAX} no evidence:      {len(rows) - evidenced}")
    print(f"Size: {len(data)} bytes")
    print(f"SHA-256: {hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    main()
