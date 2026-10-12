# CJK conversion tables

Reading→character tables that make Chinese and Japanese **typeable** in WM
Keyboard: type a reading (Pinyin, kana or a stroke sequence), pick a candidate.
The app downloads these on demand — they are host-only, not bundled in the APK —
and verifies each against a pinned SHA-256 before use.

All three are plain UTF-8 TSV, one entry per line, sorted most-frequent first, so
a lookup can offer the commonest candidate first. Lines beginning with `#` are
header comments.

## Files

| File | Format | Rows / size | Source | License |
|---|---|---|---|---|
| `pinyin_v2.tsv` | `reading⇥word⇥freq` — toneless concatenated Pinyin (lowercase ASCII) → Simplified Hanzi | ~121.4k · 2.6 MB | [CC-CEDICT](https://www.mdbg.net/chinese/dictionary?page=cc-cedict); frequencies from [`data/zh_cn`](../data/zh_cn) | CC BY-SA 4.0 |
| `pinyin.tsv` | The same, before the 2026-10 rescore. Frozen: app versions before it pin these exact bytes, so it is never rebuilt in place | ~120.6k · 2.4 MB | [CC-CEDICT](https://www.mdbg.net/chinese/dictionary?page=cc-cedict) | CC BY-SA 4.0 |
| `ja_kana.tsv` | `reading⇥word⇥freq` — hiragana reading → kanji/kana surface | ~1.08M · 40 MB | [Google Mozc](https://github.com/google/mozc) OSS dictionary (`dictionary_oss`) | BSD 3-Clause |
| `stroke.tsv` | `strokeCode⇥hanzi⇥freq` — 1–5 stroke-class sequence → Simplified Hanzi | ~20.9k · 445 KB | [yefeijiang/Chinese-characters-code-table](https://github.com/yefeijiang/Chinese-characters-code-table); frequencies from CC-CEDICT ordering | BSD 2-Clause |
| `cangjie.tsv` | `code⇥hanzi⇥freq` — 1–5 Cangjie radical code (lowercase ASCII a–y) → Hanzi | ~29.2k · 351 KB | [Unicode Unihan Database](https://www.unicode.org/Public/UCD/latest/ucd/Unihan.zip) (`kCangjie` field) | Unicode License V3 |
| `jyutping.tsv` | `reading⇥word⇥freq` — toneless concatenated Jyutping (lowercase ASCII) → Traditional Hanzi | ~162.6k · 3.4 MB | [rime-cantonese](https://github.com/rime/rime-cantonese) & [CC-Canto](https://cantonese.org/); frequencies and generated standard-vocabulary rows from [`data/zh_tw`](../data/zh_tw) | CC BY 4.0 / CC BY-SA 3.0 / CC BY-SA 4.0 |
| `s2t_phrases.txt` | `simplified⇥traditional [alternatives…]` — Simplified word → its Traditional spelling, OpenCC text format, sorted by key; the first value is the conversion. Used by Traditional output, not for typing | ~49.8k · 1.0 MB | [OpenCC](https://github.com/BYVoid/OpenCC) 1.4.2: `STPhrases` plus `STPhrases_GeneratedFromRegionalPhrases` | Apache-2.0 |
| `jyutping_syllables.txt` | `syllable` — valid toneless Cantonese Jyutping syllable inventory (lowercase ASCII) | 701 rows · ~3.1 KB | [lshk-org/jyutping-table](https://github.com/lshk-org/jyutping-table) | CC BY 4.0 |

Stroke classes: `1`=一 (héng), `2`=丨 (shù), `3`=丿 (piě), `4`=丶 (diǎn), `5`=乙 (zhé).

### Checksums

The app pins these (see `CjkDictCatalog.kt`); recompute with `shasum -a 256 <file>` after a rebuild.

A pinned file is never overwritten: every app version already installed checks
it against the checksum it shipped with, so changed bytes under the same name
break the download for all of them. A rebuild that changes a pack goes out
under a new name (`pinyin.tsv` → `pinyin_v2.tsv`), the old file stays, and the
app's catalog moves to the new name in the same release.

```
pinyin_v2.tsv 2586937  0c572b5f5699378beb0eab5ec81cf79bb277e0f100d3c944d8969b7445b1a0c9
pinyin.tsv    2489699  8baab4c758499272e36dba4bda4253317a4f93bb88bcf386ea86196c50d73715
ja_kana.tsv  41531397  189214b81968c857d7cb020c52fc087ee44918ab28534194f79ea66f45c17a70
stroke.tsv     445573  87d790c976b28e54107aef1af4ea3cea6e608500de0bcb3c01af7d8f1f8c52e8
cangjie.tsv    351337  77a5a4c054019e3a3ea875e86a37cb08b71ce29392b6680208bb0c5749feb25d
jyutping.tsv  3416241  cc0666179df615846328ed1e86b621919553d8ff30905f30d0b6ff45315bcfba
s2t_phrases.txt 1021797  a8c1ff12191edfc8dc435d23736e2bf097357650ea195f51767d397c5b70908c
```

## Rebuilding

The `tools/` scripts fetch each upstream source and regenerate its table (no
arguments needed; each downloads what it needs):

```bash
python3 tools/build_cedict_pinyin.py   # → pinyin_v2.tsv (from CC-CEDICT; reads data/zh_cn for frequencies)
python3 tools/build_mozc_ja_kana.py    # → ja_kana.tsv  (from Mozc)
python3 tools/build_stroke_dict.py     # → stroke.tsv   (needs pinyin.tsv for freq)
python3 tools/build_cangjie_dict.py    # → cangjie.tsv  (from Unihan)
python3 tools/build_jyutping_dict.py   # → jyutping.tsv (from rime-cantonese & CC-Canto; reads data/zh_tw for frequencies)
```

`s2t_phrases.txt` is built from the app repository instead, by
`scripts/update-opencc.py`, because the same run refreshes the small OpenCC
tables the APK bundles next to it. It writes here when this checkout sits
beside the app's (or at `$WMKB_DATA`) and prints the size and checksum to pin.

`build_cedict_pinyin.py` reads `data/zh_cn/zh_cn_full.txt.gz` the same way,
since CC-CEDICT has no frequencies either. Single characters are scored per
reading from the corpus words that contain them, so 议 and 际 — rare alone,
everywhere inside words — rank where they belong. Before 2026-10-12 the script
reseeded from its own previous output and otherwise ranked by line position in
CC-CEDICT, which left 齐 113th under `qi` (issue #633); that output is the
frozen `pinyin.tsv`. `stroke.tsv` still carries frequencies copied from it
until it is rebuilt.

`build_jyutping_dict.py` also needs a local file as well as
its downloads: neither Cantonese source ships frequencies, so it reads
`data/zh_tw/zh_tw_full.txt.gz` for them. Pass `--corpus` to point it elsewhere,
or `--no-coverage` to build from the Cantonese sources alone — the result is
usable but candidate order is close to arbitrary, which is what the pack looked
like before 2026-07-27.

## Attribution

- **`pinyin_v2.tsv`, `pinyin.tsv`**: CC-CEDICT data is redistributed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) with attribution to the CC-CEDICT project. Frequencies derive from the `zh_cn` frequency list in [`data/`](../data) ([hermitdave/FrequencyWords](https://github.com/hermitdave/FrequencyWords), OpenSubtitles 2018, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)).
- **`ja_kana.tsv`**: Google Mozc dictionary data (`dictionary_oss`) is redistributed under the [BSD 3-Clause License](LICENSE_mozc.txt).
- **`stroke.tsv`**: Derived from [yefeijiang/Chinese-characters-code-table](https://github.com/yefeijiang/Chinese-characters-code-table) by FeiJiang Ye, licensed under the [BSD 2-Clause License](LICENSE_stroke.txt).
- **`cangjie.tsv`**: Derived from the [Unicode Unihan Database](https://www.unicode.org/Public/UCD/latest/ucd/Unihan.zip), licensed under the [Unicode License V3](LICENSE_cangjie.txt).
- **`jyutping.tsv`**: Derived from [rime-cantonese](https://github.com/rime/rime-cantonese) (by CanCLID, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)) and [CC-Canto](https://cantonese.org/) (by Pleco Inc., licensed under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)). See [LICENSE_jyutping.txt](LICENSE_jyutping.txt). Frequencies, and the generated rows covering standard vocabulary neither Cantonese source lists, additionally derive from the `zh_tw` frequency list in [`data/`](../data) ([hermitdave/FrequencyWords](https://github.com/hermitdave/FrequencyWords), OpenSubtitles 2018, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)).
- **`s2t_phrases.txt`**: Copied from the dictionaries of [OpenCC](https://github.com/BYVoid/OpenCC) (Open Chinese Convert) 1.4.2, Copyright Carbo Kuo and OpenCC contributors, licensed under the [Apache License 2.0](LICENSE_opencc.txt). The rows from `STPhrases_GeneratedFromRegionalPhrases` are regenerated the way OpenCC's build does it, from the keys of its `TWPhrases` and `HKPhrases`.
- **`jyutping_syllables.txt`**: Derived from [lshk-org/jyutping-table](https://github.com/lshk-org/jyutping-table) by the Linguistic Society of Hong Kong (LSHK), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).



