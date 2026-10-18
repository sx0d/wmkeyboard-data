#!/usr/bin/env python3

"""Build or improve word bigram/trigram frequency lists from the Tatoeba sentence corpus:

    <word1> <word2> <count>
    <word1> <word2> <word3> <count>

sorted by count, descending, then key ascending. Output is written to:

    data/<lang>/<prefix>_bigrams.txt.gz
    data/<lang>/<prefix>_trigrams.txt.gz

Why Tatoeba:
Tatoeba is a massive open database of short, natural, conversational sentences
written and translated by native speakers. While news and encyclopedic corpora
(like Leipzig or Wikipedia) provide vocabulary breadth, phone keyboards must
predict what someone types to friends and family in daily conversation. Tatoeba
provides the ideal register for next-word prediction.

This script supports:
  1. Adding word pairs for languages that currently have none (e.g. Esperanto,
     Toki Pona, Latin, Lithuanian, Latvian, Macedonian, Armenian, Belarusian,
     Klingon, Lojban, Tatar, etc.).
  2. Improving existing word pairs (--blend): folds conversational Tatoeba sentences
     into existing bigram/trigram lists with a configurable weight (default x3),
     boosting everyday conversational pairs and filling gaps in lower-resource languages.

Attribution:
Sentences from Tatoeba are licensed under CC BY 2.0 FR / CC0 (https://tatoeba.org).

Usage:
    # Add word pairs for Esperanto:
    python3 scripts/import_tatoeba_ngrams.py --lang eo --tatoeba /tmp/sentences.csv

    # Add word pairs for Toki Pona and Klingon:
    python3 scripts/import_tatoeba_ngrams.py --lang tok tlh --tatoeba /tmp/sentences.csv

    # Improve existing Assamese word pairs by blending Tatoeba sentences:
    python3 scripts/import_tatoeba_ngrams.py --lang as --blend --weight 3 --tatoeba /tmp/sentences.csv
"""

import argparse
import bz2
import gzip
import re
import sys
import unicodedata
from array import array
from collections import defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

# Standard ISO 639-1 / BCP-47 to Tatoeba ISO 639-3 mapping
ISO1_TO_ISO3 = {
    "aa": "aar", "ab": "abk", "af": "afr", "ak": "aka", "am": "amh", "an": "arg",
    "ar": "ara", "as": "asm", "av": "ava", "ay": "aym", "az": "aze", "ba": "bak",
    "be": "bel", "bg": "bul", "bi": "bis", "bm": "bam", "bn": "ben", "bo": "bod",
    "br": "bre", "bs": "bos", "ca": "cat", "ce": "che", "ch": "cha", "co": "cos",
    "cr": "cre", "cs": "ces", "cu": "chu", "cv": "chv", "cy": "cym", "da": "dan",
    "de": "deu", "dv": "div", "dz": "dzo", "ee": "ewe", "el": "ell", "en": "eng",
    "eo": "epo", "es": "spa", "et": "est", "eu": "eus", "fa": "pes", "ff": "ful",
    "fi": "fin", "fj": "fij", "fo": "fao", "fr": "fra", "fy": "fry", "ga": "gle",
    "gd": "gla", "gl": "glg", "gn": "grn", "gu": "guj", "gv": "glv", "ha": "hau",
    "he": "heb", "hi": "hin", "ho": "hmo", "hr": "hrv", "ht": "hat", "hu": "hun",
    "hy": "hye", "hz": "her", "ia": "ina", "id": "ind", "ie": "ile", "ig": "ibo",
    "ii": "iii", "ik": "ipk", "io": "ido", "is": "isl", "it": "ita", "iu": "iku",
    "ja": "jpn", "jv": "jav", "ka": "kat", "kg": "kon", "ki": "kik", "kj": "kua",
    "kk": "kaz", "kl": "kal", "km": "khm", "kn": "kan", "ko": "kor", "kr": "kau",
    "ks": "kas", "ku": "kmr", "kv": "kom", "kw": "cor", "ky": "kir", "la": "lat",
    "lb": "ltz", "lg": "lug", "li": "lim", "ln": "lin", "lo": "lao", "lt": "lit",
    "lu": "lub", "lv": "lvs", "mg": "mlg", "mh": "mah", "mi": "mri", "mk": "mkd",
    "ml": "mal", "mn": "mon", "mr": "mar", "ms": "zsm", "mt": "mlt", "my": "mya",
    "na": "nau", "nb": "nob", "nd": "nde", "ne": "npi", "ng": "ndo", "nl": "nld",
    "nn": "nno", "no": "nob", "nr": "nbl", "nv": "nav", "ny": "nya", "oc": "oci",
    "oj": "oji", "om": "orm", "or": "ori", "os": "oss", "pa": "pan", "pi": "pli",
    "pl": "pol", "ps": "pus", "pt": "por", "qu": "que", "rm": "roh", "rn": "run",
    "ro": "ron", "ru": "rus", "rw": "kin", "sa": "san", "sc": "srd", "sd": "snd",
    "se": "sme", "sg": "sag", "sh": "hbs", "si": "sin", "sk": "slk", "sl": "slv",
    "sm": "smo", "sn": "sna", "so": "som", "sq": "sqi", "sr": "srp", "ss": "ssw",
    "st": "sot", "su": "sun", "sv": "swe", "sw": "swh", "ta": "tam", "te": "tel",
    "tg": "tgk", "th": "tha", "ti": "tir", "tk": "tuk", "tl": "tgl", "tn": "tsn",
    "to": "ton", "tr": "tur", "ts": "tso", "tt": "tat", "tw": "twi", "ty": "tah",
    "ug": "uig", "uk": "ukr", "ur": "urd", "uz": "uzb", "ve": "ven", "vi": "vie",
    "vo": "vol", "wa": "wln", "wo": "wol", "xh": "xho", "yi": "yid", "yo": "yor",
    "za": "zha", "zh": "cmn", "zu": "zul",
    # Regional and orthographic aliases in wmkeyboard-data
    "fil": "tgl", "no": "nob", "nds_nl": "nds", "be_x_old": "bel",
    "bxr": "bua", "sun": "sun", "cr": "crk", "eml": "egl",
    "fiu_vro": "vro", "div": "div"
}


def get_tatoeba_code(lang: str) -> str:
    """Map a WM Keyboard folder / ISO 639-1 code to Tatoeba ISO 639-3 code."""
    return ISO1_TO_ISO3.get(lang, lang)

LATIN_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)*")
LATIN_MAX_TOKEN_LEN = 24

KLINGON_TOKEN_RE = re.compile(r"[A-Za-z']+")
KLINGON_MAX_TOKEN_LEN = 28

ZWNJ = "\u200c"
ZWJ = "\u200d"
ZERO_WIDTH = str.maketrans("", "", "\u200b\u200c\u200d\ufeff\u00ad")
GENERIC_MAX_TOKEN_LEN = 32

BENGALI_BASE_RANGES = (
    (0x0985, 0x098C),
    (0x098F, 0x0990),
    (0x0993, 0x09A8),
    (0x09AA, 0x09B0),
    (0x09B2, 0x09B2),
    (0x09B6, 0x09B9),
    (0x09CE, 0x09CE),
)
BENGALI_MARK_RANGES = (
    (0x0981, 0x0983),
    (0x09BC, 0x09BC),
    (0x09BE, 0x09C4),
    (0x09C7, 0x09C8),
    (0x09CB, 0x09CD),
)
BENGALI_BASE = frozenset(
    chr(c) for lo, hi in BENGALI_BASE_RANGES for c in range(lo, hi + 1)
)
BENGALI_VOWEL_SIGN = frozenset(
    chr(c)
    for lo, hi in ((0x09BE, 0x09C4), (0x09C7, 0x09C8), (0x09CB, 0x09CC))
    for c in range(lo, hi + 1)
)
BENGALI_NUKTA = "\u09BC"
BENGALI_VIRAMA = "\u09CD"
BENGALI_NUKTA_BASES = frozenset("\u09A1\u09A2\u09AF")
BENGALI_TOKEN_RE = re.compile(
    "["
    + "".join(
        f"\\u{lo:04X}-\\u{hi:04X}"
        for lo, hi in BENGALI_BASE_RANGES + BENGALI_MARK_RANGES
    )
    + "]+"
)
BENGALI_MAX_TOKEN_LEN = 32


def is_valid_bengali(token: str) -> bool:
    if not token or len(token) > BENGALI_MAX_TOKEN_LEN:
        return False
    if token[0] not in BENGALI_BASE:
        return False
    previous = ""
    for char in token:
        if char == BENGALI_NUKTA and previous not in BENGALI_NUKTA_BASES:
            return False
        if char == BENGALI_VIRAMA and previous == BENGALI_VIRAMA:
            return False
        if char in BENGALI_VOWEL_SIGN and previous == BENGALI_VIRAMA:
            return False
        if char in BENGALI_VOWEL_SIGN and previous in BENGALI_VOWEL_SIGN:
            return False
        previous = char
    return True


def normalize_bengali(sentence: str) -> str:
    sentence = sentence.translate(ZERO_WIDTH)
    if not unicodedata.is_normalized("NFKC", sentence):
        sentence = unicodedata.normalize("NFKC", sentence)
    if not unicodedata.is_normalized("NFC", sentence):
        sentence = unicodedata.normalize("NFC", sentence)
    return sentence


def normalize_generic(text: str) -> str:
    text = unicodedata.normalize("NFC", unicodedata.normalize("NFKC", text)).lower()
    return text.replace("\u200b", " ").replace(ZWJ, "").replace("\u2019", "'").replace("\u02BC", "'")


def normalize_klingon(text: str) -> str:
    # Klingon case is phonemic (D, H, I, S, Q vs q); do not lower-case.
    text = unicodedata.normalize("NFC", unicodedata.normalize("NFKC", text))
    return text.replace(ZWJ, "").replace("\u2019", "'").replace("\u02BC", "'")


def prepare_latin(sentence: str) -> str:
    return sentence.lower().replace("\u2019", "'").replace("\u02BC", "'")


def generic_alphabet(words):
    chars = set()
    apostrophe = False
    zwnj = False
    for w in words:
        for ch in w:
            if unicodedata.category(ch)[0] in "LM":
                chars.add(ord(ch))
            elif ch == "'":
                apostrophe = True
            elif ch == ZWNJ:
                zwnj = True
    if apostrophe:
        chars.add(ord("'"))
    if zwnj:
        chars.add(ord(ZWNJ))
    if not chars:
        return re.compile(r"[\w]+")
    ranges = []
    for c in sorted(chars):
        if ranges and c == ranges[-1][1] + 1:
            ranges[-1][1] = c
        else:
            ranges.append([c, c])
    body = "".join(
        f"\\U{lo:08X}" if lo == hi else f"\\U{lo:08X}-\\U{hi:08X}" for lo, hi in ranges
    )
    return re.compile("[" + body + "]+")


class Tokenizer:
    def __init__(self, pattern, max_length, prepare=None, accept=None, vocab=None):
        self.pattern = pattern
        self.max_length = max_length
        self.prepare = prepare
        self.accept = accept
        self.vocab = vocab

    def __call__(self, sentence: str):
        if self.prepare is not None:
            sentence = self.prepare(sentence)
        runs = []
        run = []
        pos = 0
        for match in self.pattern.finditer(sentence):
            if sentence[pos:match.start()].strip() and run:
                runs.append(run)
                run = []
            token = match.group(0).strip("'")
            ok = bool(token) and len(token) <= self.max_length
            if ok and self.accept is not None:
                ok = self.accept(token)
            if ok and self.vocab is not None:
                ok = token in self.vocab
            if ok:
                run.append(token)
            elif run:
                runs.append(run)
                run = []
            pos = match.end()
        if run:
            runs.append(run)
        return runs


class KlingonTokenizer(Tokenizer):
    """Klingon tokenizer: preserves phonemic capitals, keeps word-internal and
    leading glottal stop apostrophes (like 'e' and 'ej), but strips outer quotes."""

    def __call__(self, sentence: str):
        sentence = normalize_klingon(sentence)
        runs = []
        run = []
        pos = 0
        for match in self.pattern.finditer(sentence):
            if sentence[pos:match.start()].strip() and run:
                runs.append(run)
                run = []
            token = match.group(0)
            # A Klingon word may start or end with a glottal stop ', but outer quotes
            # around a word (e.g. "..." or '...') should be stripped unless the token in vocab.
            if token not in self.vocab and token.startswith("'") and token.endswith("'"):
                token = token[1:-1]
            ok = bool(token) and len(token) <= self.max_length and token in self.vocab
            if ok:
                run.append(token)
            elif run:
                runs.append(run)
                run = []
            pos = match.end()
        if run:
            runs.append(run)
        return runs


class GenericTokenizer(Tokenizer):
    def __call__(self, sentence: str):
        sentence = normalize_generic(sentence)
        runs = []
        run = []
        pos = 0
        for match in self.pattern.finditer(sentence):
            if sentence[pos:match.start()].strip() and run:
                runs.append(run)
                run = []
            token = match.group(0).strip("'")
            if token and len(token) <= self.max_length and token in self.vocab:
                run.append(token)
            elif run:
                runs.append(run)
                run = []
            pos = match.end()
        if run:
            runs.append(run)
        return runs


def make_tokenizer(script: str, vocab):
    if script == "latin":
        return Tokenizer(LATIN_TOKEN_RE, LATIN_MAX_TOKEN_LEN, prepare_latin, None, vocab)
    if script == "bengali":
        return Tokenizer(
            BENGALI_TOKEN_RE,
            BENGALI_MAX_TOKEN_LEN,
            normalize_bengali,
            is_valid_bengali,
            vocab,
        )
    if script == "klingon":
        return KlingonTokenizer(KLINGON_TOKEN_RE, KLINGON_MAX_TOKEN_LEN, None, None, vocab)
    if script == "generic":
        if vocab is None:
            raise ValueError("--script generic needs --vocab")
        return GenericTokenizer(generic_alphabet(vocab), GENERIC_MAX_TOKEN_LEN, vocab=vocab)
    raise ValueError(f"unknown --script {script}")


def auto_script_for_lang(lang: str) -> str:
    if lang == "tlh":
        return "klingon"
    if lang in ("bn", "as"):
        return "bengali"
    if lang == "en":
        return "latin"
    return "generic"


MEMORY_VOCAB_LIMIT = (1 << 21) - 1


def load_vocab(path: Path, normalize=None, limit=None):
    opener = gzip.open if path.suffix == ".gz" else open
    words = set()
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if limit is not None and len(words) >= limit:
                break
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[-1].isdigit():
                # Line like "word freq" or "word word freq"
                word = parts[0]
            else:
                word = line.rsplit(" ", 1)[0]
            if normalize is not None:
                word = normalize(word)
                if not word or " " in word or "_" in word:
                    continue
            words.add(word)
    return sorted(words)


class MemorySink:
    """Counts n-grams as int64 word-id keys, sorted in place by numpy."""

    def __init__(self, words):
        self.words = words
        self.ids = {word: index for index, word in enumerate(words)}
        self.radix = len(words)
        self.bigrams = defaultdict(lambda: array("q"))
        self.trigrams = defaultdict(lambda: array("q"))
        self._weight = 1
        self.existing_bi_keys = []
        self.existing_bi_counts = []
        self.existing_tri_keys = []
        self.existing_tri_counts = []

    def set_weight(self, weight: int) -> None:
        self._weight = weight

    def add(self, run) -> None:
        ids = [self.ids[word] for word in run]
        radix = self.radix
        w = self._weight
        if len(ids) >= 2:
            self.bigrams[w].extend(
                a * radix + b for a, b in zip(ids, ids[1:])
            )
        if len(ids) >= 3:
            self.trigrams[w].extend(
                (a * radix + b) * radix + c for a, b, c in zip(ids, ids[1:], ids[2:])
            )

    def add_existing_bigrams(self, path: Path, base_weight: int = 1) -> int:
        if not path.is_file():
            return 0
        radix = self.radix
        ids = self.ids
        keys = array("q")
        counts = array("q")
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.rstrip("\n").split(" ")
                if len(parts) >= 3:
                    w1, w2 = parts[0], parts[1]
                    try:
                        c = int(parts[2]) * base_weight
                    except ValueError:
                        continue
                    if w1 in ids and w2 in ids:
                        keys.append(ids[w1] * radix + ids[w2])
                        counts.append(c)
        if keys:
            self.existing_bi_keys.append(keys)
            self.existing_bi_counts.append(counts)
        return len(keys)

    def add_existing_trigrams(self, path: Path, base_weight: int = 1) -> int:
        if not path.is_file():
            return 0
        radix = self.radix
        ids = self.ids
        keys = array("q")
        counts = array("q")
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.rstrip("\n").split(" ")
                if len(parts) >= 4:
                    w1, w2, w3 = parts[0], parts[1], parts[2]
                    try:
                        c = int(parts[3]) * base_weight
                    except ValueError:
                        continue
                    if w1 in ids and w2 in ids and w3 in ids:
                        keys.append((ids[w1] * radix + ids[w2]) * radix + ids[w3])
                        counts.append(c)
        if keys:
            self.existing_tri_keys.append(keys)
            self.existing_tri_counts.append(counts)
        return len(keys)

    @staticmethod
    def _run_starts(keys):
        import numpy as np

        if keys.size == 0:
            return np.zeros(0, dtype=np.intp)
        change = np.empty(keys.size, dtype=bool)
        change[0] = True
        np.not_equal(keys[1:], keys[:-1], out=change[1:])
        return np.flatnonzero(change)

    def _collapse(self, groups, existing_keys_list, existing_counts_list, min_count, max_items):
        import numpy as np

        key_parts = []
        count_parts = []

        for weight in sorted(groups):
            raw = groups[weight]
            if len(raw) == 0:
                continue
            keys = np.frombuffer(raw, dtype=np.int64).copy()
            keys.sort()
            starts = self._run_starts(keys)
            lengths = np.diff(np.append(starts, keys.size)).astype(np.int64)
            key_parts.append(keys[starts])
            count_parts.append(lengths * weight)
            del keys, starts, lengths

        for k_arr, c_arr in zip(existing_keys_list, existing_counts_list):
            if len(k_arr) > 0:
                key_parts.append(np.frombuffer(k_arr, dtype=np.int64))
                count_parts.append(np.frombuffer(c_arr, dtype=np.int64))

        if not key_parts:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

        keys = np.concatenate(key_parts)
        counts = np.concatenate(count_parts)
        del key_parts, count_parts

        order = np.argsort(keys, kind="stable")
        keys = keys[order]
        counts = counts[order]
        del order

        starts = self._run_starts(keys)
        unique = keys[starts]
        summed = (
            np.add.reduceat(counts, starts)
            if starts.size
            else np.zeros(0, dtype=np.int64)
        )
        del keys, counts, starts

        keep = summed >= min_count
        unique = unique[keep]
        summed = summed[keep]

        # Stable sort: count descending, key ascending
        order = np.lexsort((unique, -summed))
        if max_items:
            order = order[:max_items]
        return unique[order], summed[order]

    def _write(self, groups, existing_keys, existing_counts, out_gz: Path, arity: int, min_count: int, max_items: int) -> int:
        keys, counts = self._collapse(groups, existing_keys, existing_counts, min_count, max_items)
        radix = self.radix
        words = self.words
        written = 0
        out_gz.parent.mkdir(parents=True, exist_ok=True)
        tmp_target = out_gz.with_suffix(".tmp.gz")
        with gzip.open(tmp_target, "wt", encoding="utf-8") as outf:
            for key, count in zip(keys.tolist(), counts.tolist()):
                parts = []
                for _ in range(arity):
                    key, index = divmod(key, radix)
                    parts.append(words[index])
                parts.reverse()
                outf.write(f"{' '.join(parts)} {count}\n")
                written += 1
        tmp_target.replace(out_gz)
        return written

    def write_bigrams(self, out_gz: Path, min_count: int, max_items: int) -> int:
        return self._write(self.bigrams, self.existing_bi_keys, self.existing_bi_counts, out_gz, 2, min_count, max_items)

    def write_trigrams(self, out_gz: Path, min_count: int, max_items: int) -> int:
        return self._write(self.trigrams, self.existing_tri_keys, self.existing_tri_counts, out_gz, 3, min_count, max_items)


def open_sentences_file(path: Path):
    p = str(path)
    if p.endswith(".bz2"):
        return bz2.open(p, "rt", encoding="utf-8", errors="replace")
    if p.endswith(".gz"):
        return gzip.open(p, "rt", encoding="utf-8", errors="replace")
    return open(p, "rt", encoding="utf-8", errors="replace")


def stream_tatoeba_sentences(tatoeba_path: Path, target_langs: set[str]):
    """Stream sentence strings matching any language code in target_langs.
    Yields (tatoeba_lang_code, sentence_string)."""
    with open_sentences_file(tatoeba_path) as f:
        for line in f:
            parts = line.split("\t", 2)
            if len(parts) >= 3:
                code = parts[1]
                if code in target_langs:
                    yield code, parts[2].rstrip("\r\n")


def process_language(
    lang: str,
    tatoeba_code: str,
    sentences: list[str],
    tatoeba_weight: int = 1,
    blend: bool = False,
    base_weight: int = 1,
    min_bigram: int = 2,
    min_trigram: int = 2,
    max_bigrams: int = 150_000,
    max_trigrams: int = 75_000,
    script: str = "auto",
    custom_vocab: Path | None = None,
    include_aosp: bool = True,
    prefix: str | None = None,
) -> tuple[int, int]:
    """Process sentences for one language, build/improve bigrams and trigrams."""
    out_dir = DATA_DIR / lang
    prefix = prefix or lang
    bi_gz = out_dir / f"{prefix}_bigrams.txt.gz"
    tri_gz = out_dir / f"{prefix}_trigrams.txt.gz"

    if script == "auto":
        script = auto_script_for_lang(lang)

    # 1. Load vocabulary
    vocab_path = custom_vocab or (out_dir / f"{lang}_full.txt.gz")
    if not vocab_path.is_file():
        raise FileNotFoundError(f"vocabulary file not found: {vocab_path}")

    normalize = None
    if script == "generic":
        normalize = normalize_generic
    elif script == "klingon":
        normalize = normalize_klingon

    words = load_vocab(vocab_path, normalize, limit=MEMORY_VOCAB_LIMIT)

    # Optionally merge AOSP vocabulary words if available
    if include_aosp:
        aosp_path = out_dir / f"{lang}_aosp.txt.gz"
        if aosp_path.is_file():
            aosp_words = load_vocab(aosp_path, normalize, limit=MEMORY_VOCAB_LIMIT)
            words = sorted(set(words) | set(aosp_words))

    if len(words) > MEMORY_VOCAB_LIMIT:
        words = words[:MEMORY_VOCAB_LIMIT]

    vocab_set = frozenset(words)
    print(f"[{lang}] Vocabulary: {len(words)} words (script={script})")

    tokenizer = make_tokenizer(script, vocab_set)
    sink = MemorySink(words)
    sink.set_weight(tatoeba_weight)

    # 2. If blending, load existing bigrams and trigrams
    if blend:
        n_exist_bi = sink.add_existing_bigrams(bi_gz, base_weight)
        n_exist_tri = sink.add_existing_trigrams(tri_gz, base_weight)
        print(f"[{lang}] Blending with existing: {n_exist_bi} bigrams, {n_exist_tri} trigrams (base_weight={base_weight})")

    # 3. Tokenize and count Tatoeba sentences
    n_tokens = 0
    n_runs = 0
    for s in sentences:
        for run in tokenizer(s):
            n_tokens += len(run)
            n_runs += 1
            sink.add(run)

    print(f"[{lang}] Tatoeba: {len(sentences)} sentences -> {n_runs} runs, {n_tokens} tokens (weight={tatoeba_weight})")

    # 4. Write output files
    n_bi = sink.write_bigrams(bi_gz, min_bigram, max_bigrams)
    n_tri = sink.write_trigrams(tri_gz, min_trigram, max_trigrams)
    print(f"[{lang}] Wrote {n_bi} bigrams -> {bi_gz.name}, {n_tri} trigrams -> {tri_gz.name}")
    return n_bi, n_tri


def main() -> int:
    parser = argparse.ArgumentParser(description="Import/improve word pairs from Tatoeba sentences")
    parser.add_argument("--tatoeba", default="/tmp/sentences.csv", help="path to Tatoeba sentences.csv[.bz2/.gz]")
    parser.add_argument("--lang", nargs="+", required=True, help="language directory in data/ (e.g. eo tok la)")
    parser.add_argument("--tatoeba-lang", help="override Tatoeba ISO 639-3 code (for single --lang)")
    parser.add_argument("--script", default="auto", choices=("auto", "generic", "klingon", "bengali", "latin"))
    parser.add_argument("--vocab", help="custom vocabulary word list (default: data/<lang>/<lang>_full.txt.gz)")
    parser.add_argument("--weight", type=int, default=0, help="Tatoeba sentence weight (default 3 if --blend, else 1)")
    parser.add_argument("--blend", action="store_true", help="blend with existing _bigrams.txt.gz and _trigrams.txt.gz")
    parser.add_argument("--base-weight", type=int, default=1, help="weight for existing n-grams when blending (default: 1)")
    parser.add_argument("--min-bigram", type=int, default=0, help="minimum count for bigram (default: 1 if blend/small, 2 if standalone)")
    parser.add_argument("--min-trigram", type=int, default=0, help="minimum count for trigram (default: 1 if blend/small, 2 if standalone)")
    parser.add_argument("--max-bigrams", type=int, default=150_000, help="maximum bigrams to keep (default: 150000)")
    parser.add_argument("--max-trigrams", type=int, default=75_000, help="maximum trigrams to keep (default: 75000)")
    parser.add_argument("--prefix", help="output file prefix (default: same as --lang)")
    args = parser.parse_args()

    tatoeba_path = Path(args.tatoeba)
    if not tatoeba_path.is_file():
        parser.error(f"Tatoeba file not found: {tatoeba_path}")

    langs = args.lang
    if args.tatoeba_lang and len(langs) > 1:
        parser.error("--tatoeba-lang can only be specified when a single --lang is given")

    # Map each target lang to its Tatoeba code
    lang_map = {}  # tatoeba_code -> list of wm_langs
    wm_to_tat = {}
    for l in langs:
        tat_code = args.tatoeba_lang if args.tatoeba_lang else ISO1_TO_ISO3.get(l, l)
        wm_to_tat[l] = tat_code
        lang_map.setdefault(tat_code, []).append(l)

    tatoeba_weight = args.weight if args.weight > 0 else (3 if args.blend else 1)
    min_bi = args.min_bigram if args.min_bigram > 0 else (1 if args.blend else 2)
    min_tri = args.min_trigram if args.min_trigram > 0 else (1 if args.blend else 2)

    print(f"Target languages: {', '.join(f'{l}->{wm_to_tat[l]}' for l in langs)}")
    print(f"Reading Tatoeba sentences from {tatoeba_path} ...")

    # Collect sentences for targeted languages in a single pass
    sentences_by_code = defaultdict(list)
    total_found = 0
    needed_codes = set(lang_map.keys())

    for code, sent in stream_tatoeba_sentences(tatoeba_path, needed_codes):
        sentences_by_code[code].append(sent)
        total_found += 1

    print(f"Loaded {total_found} sentences across {len(needed_codes)} Tatoeba language codes.")

    for l in langs:
        tat_code = wm_to_tat[l]
        sents = sentences_by_code.get(tat_code, [])
        if not sents:
            print(f"[{l}] Warning: no sentences found for code '{tat_code}' in Tatoeba file")
            continue
        process_language(
            lang=l,
            tatoeba_code=tat_code,
            sentences=sents,
            tatoeba_weight=tatoeba_weight,
            blend=args.blend,
            base_weight=args.base_weight,
            min_bigram=min_bi,
            min_trigram=min_tri,
            max_bigrams=args.max_bigrams,
            max_trigrams=args.max_trigrams,
            script=args.script,
            custom_vocab=Path(args.vocab) if args.vocab else None,
            prefix=args.prefix,
        )

    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
