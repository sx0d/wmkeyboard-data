#!/usr/bin/env python3

"""Build a frequency word list from Tatoeba sentences, for languages that have
no other corpus in this repository:

    data/<folder>/<stem>.txt.gz      one `word count` line each, count descending

Tatoeba publishes one file per language:

    https://downloads.tatoeba.org/exports/per_language/<iso3>/<iso3>_sentences.tsv.bz2

Each line is `id<TAB>iso3<TAB>sentence`. Only sentences written in the
requested script are counted, so a language Tatoeba mixes scripts for
(Ottoman Turkish in Arabic letters and in Latin transliteration, Interslavic in
Latin and Cyrillic) gives one list per script.

Words are NFKC+NFC normalised and lower-cased, like the generic n-gram
importer. A word is a run of letters and combining marks; an apostrophe
(' U+2019 U+02BC) inside a word is kept, hyphens and everything else split.
Arabic- and Hebrew-script lists drop the vowel points (and Arabic's tatweel),
which a keyboard user rarely types.

Attribution: Tatoeba sentences are licensed CC BY 2.0 FR (https://tatoeba.org).

Usage:
    python3 scripts/import_tatoeba_words.py --sentences /tmp/swc_sentences.tsv.bz2 \\
        --folder swc --script latin
    python3 scripts/import_tatoeba_words.py --sentences /tmp/ota_sentences.tsv.bz2 \\
        --folder ota --stem ota_latn --script latin
"""

import argparse
import bz2
import gzip
import sys
import unicodedata
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

APOSTROPHES = str.maketrans({"\u2019": "'", "\u02bc": "'"})
# Latin-script lists only. Kabyle and Tarifit writers borrow Greek epsilon and
# gamma for ɛ and ɣ, and Nuer writers reach for any "line below" mark for the
# macron below (U+0331) of their breathy vowels.
LATIN_LOOKALIKES = str.maketrans({"\u03b5": "\u025b", "\u03b3": "\u0263", "\u0332": "\u0331", "\u0320": "\u0331"})
ARABIC_MARKS = frozenset(chr(c) for c in list(range(0x064B, 0x0653)) + [0x0670, 0x0640])
# Niqqud and cantillation: Tatoeba's Biblical Hebrew is fully pointed, a keyboard user types it bare.
HEBREW_MARKS = frozenset(chr(c) for c in list(range(0x0591, 0x05BE)) + [0x05BF, 0x05C1, 0x05C2, 0x05C4, 0x05C5, 0x05C7])
MAX_LEN = 32


def script_of(ch: str) -> str | None:
    """The script of a letter, coarsely: only the scripts these lists are built in."""
    o = ord(ch)
    if 0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F or 0xFB50 <= o <= 0xFDFF or 0xFE70 <= o <= 0xFEFF:
        return "arabic"
    if 0x0400 <= o <= 0x052F:
        return "cyrillic"
    if 0x0370 <= o <= 0x03FF or 0x1F00 <= o <= 0x1FFF:
        return "greek"
    if 0x0590 <= o <= 0x05FF or 0xFB1D <= o <= 0xFB4F:
        return "hebrew"
    if o <= 0x02AF or 0x1E00 <= o <= 0x1EFF or 0x2C60 <= o <= 0x2C7F or 0xA720 <= o <= 0xA7FF:
        return "latin"
    return None


def sentence_script(text: str) -> str | None:
    counts = Counter(s for s in (script_of(c) for c in text if c.isalpha()) if s)
    return counts.most_common(1)[0][0] if counts else None


def tokens(text: str, strip_arabic_marks: bool, latin: bool):
    text = unicodedata.normalize("NFC", unicodedata.normalize("NFKC", text)).lower()
    # lower() turns İ into i + combining dot above; Turkish-style text means plain i.
    text = text.translate(APOSTROPHES).replace("i\u0307", "i")
    if latin:
        text = unicodedata.normalize("NFC", text.translate(LATIN_LOOKALIKES))
    word = []
    for ch in text + " ":
        cat = unicodedata.category(ch)
        if strip_arabic_marks and (ch in ARABIC_MARKS or ch in HEBREW_MARKS):
            continue
        if cat[0] == "L" or (cat[0] == "M" and word) or (ch == "'" and word):
            word.append(ch)
            continue
        token = "".join(word).strip("'")
        word = []
        if token and len(token) <= MAX_LEN:
            yield token


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sentences", required=True, help="<iso3>_sentences.tsv[.bz2]")
    parser.add_argument("--folder", required=True, help="folder under data/")
    parser.add_argument("--stem", help="output file stem (default <folder>_full)")
    parser.add_argument("--script", required=True, choices=("latin", "arabic", "cyrillic", "greek", "hebrew"))
    parser.add_argument("--min-count", type=int, default=1)
    args = parser.parse_args()

    opener = bz2.open if args.sentences.endswith(".bz2") else open
    counts = Counter()
    used = skipped = 0
    with opener(args.sentences, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\r\n").split("\t", 2)
            if len(parts) < 3:
                continue
            if sentence_script(parts[2].translate(LATIN_LOOKALIKES)) != args.script:
                skipped += 1
                continue
            used += 1
            for token in tokens(parts[2], args.script in ("arabic", "hebrew"), args.script == "latin"):
                if all(script_of(c) in (args.script, None) for c in token if c.isalpha()):
                    counts[token] += 1

    stem = args.stem or f"{args.folder}_full"
    out = DATA_DIR / args.folder / f"{stem}.txt.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(((w, c) for w, c in counts.items() if c >= args.min_count), key=lambda wc: (-wc[1], wc[0]))
    # mtime=0 keeps the gzip byte-identical across reruns.
    with open(out, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        gz.write("".join(f"{w} {c}\n" for w, c in rows).encode("utf-8"))
    print(f"{out.relative_to(DATA_DIR.parent)}: {len(rows)} words from {used} sentences ({skipped} in another script)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
