#!/usr/bin/env python3

"""Build a frequency word list from eBible and Glot500 text, for languages no
word-list source in this repository covers:

    data/<folder>/<stem>.txt.gz      one `word count` line each, count descending

Inputs, any number of each, merged into one count:

  --ebible FILE     a BibleNLP eBible corpus file (https://github.com/BibleNLP/ebible,
                    corpus/<iso3>-<translation>.txt), one verse per line
  --glot500 FILE    a Glot500 shard (https://huggingface.co/datasets/cis-lmu/Glot500,
                    <iso3>_<Script>/train/data-*.arrow), an Arrow IPC stream with a
                    `text` column. A shard cut short by a ranged download is read
                    up to the last whole record batch.

Only words in the text's dominant script are counted (Glot500 lines and
Bible front matter carry English and Latin-script names), judged by the
first word of each letter's Unicode name. Words are NFKC+NFC normalised and
lower-cased, keep their combining marks and inner apostrophes, and split on
everything else, as in import_tatoeba_words.py.

    python3 scripts/import_corpus_words.py --folder mah \\
        --ebible mah-mah.txt --glot500 mah_Latn-00000.arrow --script-name LATIN
"""

import argparse
import gzip
import sys
import unicodedata
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
APOSTROPHES = str.maketrans({"’": "'", "ʼ": "'"})
MAX_LEN = 32


def script_name(ch: str) -> str:
    try:
        return unicodedata.name(ch).split()[0]
    except ValueError:
        return ""


def words(text: str):
    text = unicodedata.normalize("NFC", unicodedata.normalize("NFKC", text)).lower()
    text = text.translate(APOSTROPHES).replace("i̇", "i")
    word = []
    for ch in text + " ":
        cat = unicodedata.category(ch)
        if cat[0] == "L" or (cat[0] == "M" and word) or (ch == "'" and word):
            word.append(ch)
            continue
        token = "".join(word).strip("'")
        word = []
        if token and len(token) <= MAX_LEN:
            yield token


def ebible_lines(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line and line != "<range>":
                yield line


def glot500_lines(path):
    import pyarrow as pa

    with open(path, "rb") as f:
        try:
            reader = pa.ipc.open_stream(f)
            for batch in reader:
                for text in batch.column(batch.schema.get_field_index("text")).to_pylist():
                    if text:
                        yield text
        except (pa.ArrowInvalid, OSError):
            return  # truncated by the ranged download: stop at the last whole batch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folder", required=True)
    parser.add_argument("--stem", help="output stem (default <folder>_full)")
    parser.add_argument("--ebible", action="append", default=[])
    parser.add_argument("--glot500", action="append", default=[])
    parser.add_argument("--script-name", help="Unicode script to keep (LATIN, CYRILLIC...); default: the text's commonest")
    parser.add_argument("--min-words", type=int, default=1000, help="write nothing below this many distinct words")
    args = parser.parse_args()

    counts = Counter()
    for path in args.ebible:
        for line in ebible_lines(path):
            counts.update(words(line))
    for path in args.glot500:
        for line in glot500_lines(path):
            counts.update(words(line))

    letters = Counter()
    for w, n in counts.items():
        for ch in w:
            if ch.isalpha():
                letters[script_name(ch)] += n
    keep = args.script_name or (letters.most_common(1)[0][0] if letters else "")
    rows = [
        (w, n) for w, n in counts.items()
        if all(script_name(ch) == keep for ch in w if ch.isalpha())
    ]
    rows.sort(key=lambda wn: (-wn[1], wn[0]))
    stem = args.stem or f"{args.folder}_full"
    if len(rows) < args.min_words:
        print(f"{args.folder}/{stem}: only {len(rows)} {keep} words, not written")
        return 1
    out = DATA_DIR / args.folder / f"{stem}.txt.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        gz.write("".join(f"{w} {n}\n" for w, n in rows).encode("utf-8"))
    print(f"{out.relative_to(DATA_DIR.parent)}: {len(rows)} {keep} words")
    return 0


if __name__ == "__main__":
    sys.exit(main())
