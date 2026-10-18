import gzip
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from import_tatoeba_ngrams import (
    ISO1_TO_ISO3,
    GenericTokenizer,
    KlingonTokenizer,
    MemorySink,
    Tokenizer,
    auto_script_for_lang,
    get_tatoeba_code,
    make_tokenizer,
    normalize_bengali,
    normalize_generic,
    normalize_klingon,
    process_language,
)


def test_iso_mapping():
    assert get_tatoeba_code("eo") == "epo"
    assert get_tatoeba_code("tok") == "tok"
    assert get_tatoeba_code("la") == "lat"
    assert get_tatoeba_code("lt") == "lit"
    assert get_tatoeba_code("lv") == "lvs"
    assert get_tatoeba_code("mk") == "mkd"
    assert get_tatoeba_code("hy") == "hye"
    assert get_tatoeba_code("be") == "bel"
    assert get_tatoeba_code("tlh") == "tlh"
    assert get_tatoeba_code("jbo") == "jbo"
    assert get_tatoeba_code("as") == "asm"


def test_auto_script_selection():
    assert auto_script_for_lang("tlh") == "klingon"
    assert auto_script_for_lang("bn") == "bengali"
    assert auto_script_for_lang("as") == "bengali"
    assert auto_script_for_lang("en") == "latin"
    assert auto_script_for_lang("eo") == "generic"
    assert auto_script_for_lang("tok") == "generic"


def test_generic_tokenizer_vocab_and_boundary():
    vocab = frozenset(["bonan", "tagon", "amiko", "la", "mondo"])
    tok = make_tokenizer("generic", vocab)

    # Clean sentence
    runs = tok("Bonan tagon, amiko!")
    # "Bonan tagon" is a run; comma breaks the window; "amiko" is another run
    assert runs == [["bonan", "tagon"], ["amiko"]]

    # Unknown word breaks the run
    runs = tok("bonan tagon nekonata amiko")
    assert runs == [["bonan", "tagon"], ["amiko"]]


def test_klingon_tokenizer_case_preservation():
    vocab = frozenset(["tlhIngan", "Hol", "Dajatlh'a'", "qaStaHvIS", "'e'", "'ej", "QaQ"])
    tok = make_tokenizer("klingon", vocab)

    runs = tok("tlhIngan Hol Dajatlh'a'?")
    assert runs == [["tlhIngan", "Hol", "Dajatlh'a'"]]

    # Whole sentence in quotes
    runs = tok("\"tlhIngan Hol\"")
    assert runs == [["tlhIngan", "Hol"]]

    # Glottal stop apostrophes preserved
    runs = tok("'e' 'ej QaQ")
    assert runs == [["'e'", "'ej", "QaQ"]]

    # Case distinction: "qaq" not in vocab, "QaQ" is in vocab
    runs = tok("QaQ qaq QaQ")
    assert runs == [["QaQ"], ["QaQ"]]


def test_bengali_tokenizer():
    vocab = frozenset(["মই", "তোমাক", "ভাল", "পাওঁ"])
    tok = make_tokenizer("bengali", vocab)

    runs = tok("মই তোমাক ভাল পাওঁ।")
    assert runs == [["মই", "তোমাক", "ভাল", "পাওঁ"]]


def test_memory_sink_counting():
    words = ["amiko", "bonan", "la", "mondo", "tagon"]
    sink = MemorySink(words)
    sink.set_weight(1)

    sink.add(["bonan", "tagon", "amiko"])
    sink.add(["bonan", "tagon"])

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        bi_file = tmp_path / "bi.txt.gz"
        tri_file = tmp_path / "tri.txt.gz"

        n_bi = sink.write_bigrams(bi_file, min_count=1, max_items=10)
        n_tri = sink.write_trigrams(tri_file, min_count=1, max_items=10)

        assert n_bi == 2
        assert n_tri == 1

        with gzip.open(bi_file, "rt", encoding="utf-8") as f:
            lines = [l.strip() for l in f]
            assert lines == ["bonan tagon 2", "tagon amiko 1"]

        with gzip.open(tri_file, "rt", encoding="utf-8") as f:
            lines = [l.strip() for l in f]
            assert lines == ["bonan tagon amiko 1"]


def test_memory_sink_blending():
    words = ["amiko", "bonan", "la", "mondo", "tagon"]
    sink = MemorySink(words)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        old_bi = tmp_path / "old_bi.txt.gz"
        with gzip.open(old_bi, "wt", encoding="utf-8") as f:
            f.write("bonan tagon 10\n")
            f.write("la mondo 5\n")

        sink.add_existing_bigrams(old_bi, base_weight=1)

        # Add Tatoeba occurrences with weight 3
        sink.set_weight(3)
        sink.add(["bonan", "tagon"])  # +3 -> 13
        sink.add(["tagon", "amiko"])  # +3 -> 3

        new_bi = tmp_path / "new_bi.txt.gz"
        sink.write_bigrams(new_bi, min_count=1, max_items=10)

        with gzip.open(new_bi, "rt", encoding="utf-8") as f:
            lines = [l.strip() for l in f]
            assert lines == ["bonan tagon 13", "la mondo 5", "tagon amiko 3"]


def test_process_language_end_to_end(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        # Mock DATA_DIR
        import import_tatoeba_ngrams
        monkeypatch.setattr(import_tatoeba_ngrams, "DATA_DIR", tmp_dir)

        lang_dir = tmp_dir / "eo"
        lang_dir.mkdir(parents=True)
        vocab_file = lang_dir / "eo_full.txt.gz"
        with gzip.open(vocab_file, "wt", encoding="utf-8") as f:
            f.write("bonan 100\ntagon 90\namiko 80\n")

        sentences = [
            "Bonan tagon!",
            "Bonan tagon amiko!",
            "Bonan tagon!",
        ]

        n_bi, n_tri = process_language(
            lang="eo",
            tatoeba_code="epo",
            sentences=sentences,
            tatoeba_weight=1,
            blend=False,
            min_bigram=1,
            min_trigram=1,
            max_bigrams=100,
            max_trigrams=100,
        )

        assert n_bi == 2
        assert n_tri == 1

        bi_file = lang_dir / "eo_bigrams.txt.gz"
        assert bi_file.is_file()
        with gzip.open(bi_file, "rt", encoding="utf-8") as f:
            lines = [l.strip() for l in f]
            assert lines == ["bonan tagon 3", "tagon amiko 1"]

