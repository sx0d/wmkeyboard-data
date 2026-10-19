#!/usr/bin/env python3

"""Write a language's word list in its other script, from the list it has:

    sr   data/sr/sr_full      -> data/sr/sr_latn      Serbian Cyrillic -> Latin (Gaj)
    bs   data/bs/bs_full      -> data/bs/bs_cyrl      Bosnian Latin -> Cyrillic (Gaj, reversed)
    kk   data/kk/kk_full      -> data/kk/kk_latn      Kazakh Cyrillic -> Latin (2021 alphabet)
    be   data/be/be_full      -> data/be/be_latn      Belarusian Cyrillic -> Łacinka
    zgh  data/zgh/zgh_full    -> data/zgh/zgh_latn    Tamazight Tifinagh -> Latin (IRCAM)

Counts are carried over; two source words that land on one spelling add
up. Words the scheme cannot carry (a letter it has no mapping for) are
dropped rather than half-converted.

    python3 scripts/transliterate_wordlist.py sr bs kk be zgh
"""

import gzip
import sys
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SR = dict(zip(
    "абвгдђежзијклљмнњопрстћуфхцчџш",
    ["a", "b", "v", "g", "d", "đ", "e", "ž", "z", "i", "j", "k", "l", "lj", "m", "n", "nj",
     "o", "p", "r", "s", "t", "ć", "u", "f", "h", "c", "č", "dž", "š"],
))
BS_DIGRAPHS = {"lj": "љ", "nj": "њ", "dž": "џ"}
BS = {v: k for k, v in SR.items() if len(v) == 1}

KK = {
    "а": "a", "ә": "ä", "б": "b", "в": "v", "г": "g", "ғ": "ğ", "д": "d", "е": "e", "ё": "io",
    "ж": "j", "з": "z", "и": "i", "й": "i", "к": "k", "қ": "q", "л": "l", "м": "m", "н": "n",
    "ң": "ñ", "о": "o", "ө": "ö", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ұ": "ū",
    "ү": "ü", "ф": "f", "х": "h", "һ": "h", "ц": "ts", "ч": "ç", "ш": "ş", "щ": "şş", "ъ": "",
    "ы": "ı", "і": "i", "ь": "", "э": "e", "ю": "iu", "я": "ia",
}

ZGH = {
    "ⴰ": "a", "ⴱ": "b", "ⴳ": "g", "ⴷ": "d", "ⴹ": "ḍ", "ⴻ": "e", "ⴼ": "f", "ⴽ": "k", "ⵀ": "h",
    "ⵃ": "ḥ", "ⵄ": "ɛ", "ⵅ": "x", "ⵇ": "q", "ⵉ": "i", "ⵊ": "j", "ⵍ": "l", "ⵎ": "m", "ⵏ": "n",
    "ⵓ": "u", "ⵔ": "r", "ⵕ": "ṛ", "ⵖ": "ɣ", "ⵙ": "s", "ⵚ": "ṣ", "ⵛ": "c", "ⵜ": "t", "ⵟ": "ṭ",
    "ⵡ": "w", "ⵢ": "y", "ⵣ": "z", "ⵥ": "ẓ", "ⵯ": "ʷ", "ⵒ": "p", "ⵠ": "v",
}

# Belarusian classical Łacinka. Hard л is ł, soft (before a front vowel or ь) is l;
# е ё ю я are je jo ju ja at the start, after a vowel, ў or an apostrophe, and
# ie io iu ia after a consonant (e o u a after л, whose softness l already shows);
# і after a vowel is ji.
BE_SIMPLE = {
    "а": "a", "б": "b", "в": "v", "г": "h", "ґ": "g", "д": "d", "ж": "ž", "з": "z", "і": "i",
    "й": "j", "к": "k", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t",
    "у": "u", "ў": "ŭ", "ф": "f", "х": "ch", "ц": "c", "ч": "č", "ш": "š", "ы": "y", "э": "e",
}
BE_IOTATED = {"е": "e", "ё": "o", "ю": "u", "я": "a"}
BE_SOFT_SIGN = {"з": "ź", "с": "ś", "н": "ń", "ц": "ć", "л": "l"}
BE_VOWELS = set("аеёіоуыэюяў")


def sr(word):
    return "".join(SR[c] for c in word) if all(c in SR for c in word) else None


def bs(word):
    out, i = [], 0
    while i < len(word):
        two = word[i:i + 2]
        if two in BS_DIGRAPHS:
            out.append(BS_DIGRAPHS[two])
            i += 2
        elif word[i] in BS:
            out.append(BS[word[i]])
            i += 1
        else:
            return None
    return "".join(out)


def kk(word):
    return "".join(KK[c] for c in word) if all(c in KK for c in word) else None


def zgh(word):
    return "".join(ZGH[c] for c in word) if all(c in ZGH for c in word) else None


def be(word):
    word = word.replace("’", "'").replace("ʼ", "'")
    out = []
    for i, c in enumerate(word):
        prev = word[i - 1] if i else ""
        nxt = word[i + 1] if i + 1 < len(word) else ""
        if c == "'":
            continue
        if c == "л":
            out.append("l" if nxt and nxt in "еёюяіь" else "ł")
        elif c in BE_IOTATED:
            base = BE_IOTATED[c]
            if not prev or prev in BE_VOWELS or prev == "'":
                out.append("j" + base)
            elif prev == "л":
                out.append(base)
            else:
                out.append("i" + base)
        elif c == "і" and prev and (prev in BE_VOWELS or prev == "'"):
            out.append("ji")
        elif c == "ь":
            if prev in BE_SOFT_SIGN and prev != "л":
                out[-1] = BE_SOFT_SIGN[prev]
        elif c in BE_SIMPLE:
            out.append(BE_SIMPLE[c])
        else:
            return None
    return "".join(out)


JOBS = {
    "sr": ("sr", "sr_full", "sr_latn", sr),
    "bs": ("bs", "bs_full", "bs_cyrl", bs),
    "kk": ("kk", "kk_full", "kk_latn", kk),
    "be": ("be", "be_full", "be_latn", be),
    "zgh": ("zgh", "zgh_full", "zgh_latn", zgh),
}


def run(name):
    folder, src, dst, fn = JOBS[name]
    counts = Counter()
    dropped = 0
    with gzip.open(DATA_DIR / folder / f"{src}.txt.gz", "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").rsplit(" ", 1)
            if len(parts) != 2 or not parts[1].isdigit():
                continue
            out = fn(parts[0].lower())
            if out:
                counts[out] += int(parts[1])
            else:
                dropped += 1
    rows = sorted(counts.items(), key=lambda wn: (-wn[1], wn[0]))
    path = DATA_DIR / folder / f"{dst}.txt.gz"
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        gz.write("".join(f"{w} {n}\n" for w, n in rows).encode("utf-8"))
    print(f"{path.relative_to(DATA_DIR.parent)}: {len(rows)} words ({dropped} not carried)")


if __name__ == "__main__":
    for name in sys.argv[1:] or JOBS:
        run(name)
