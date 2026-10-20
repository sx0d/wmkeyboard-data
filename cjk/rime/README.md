# RIME schema packs

Schema and dictionary YAML for WM Keyboard's optional RIME engine (librime,
`native/librime-jni` in the app repository). The app downloads a zip when
someone picks RIME for Chinese or Cantonese and fetches a pack, checks it
against the SHA-256 pinned in `RimeSchemaCatalog.kt`, unpacks it into RIME's
shared data directory and compiles it on the phone.

Built by `scripts/build_rime_packs.py` from upstream at pinned commits; the
script names each commit. Every zip carries its upstream licence and a
`SOURCE` file under `licenses/<repo>/`, which librime never reads.
`packs.tsv` lists each file's size and SHA-256.

| File | Holds | Upstream | Licence |
|---|---|---|---|
| `rime_base.zip` | `default.yaml`, key bindings, punctuation, symbols, `essay.txt`, OpenCC's tables and configurations | rime-prelude, rime-essay, OpenCC | LGPL-3.0, LGPL-3.0, Apache-2.0 |
| `rime_luna_pinyin.zip` | luna_pinyin and its variants, luna_quanpin, stroke | rime-luna-pinyin, rime-stroke | LGPL-3.0 |
| `rime_ice.zip` | rime_ice, Double Pinyin schemes, t9, melt_eng, radical_pinyin, their dictionaries | iDvel/rime-ice | GPL-3.0 |
| `rime_bopomofo.zip` | bopomofo and its variants, terra_pinyin, stroke | rime-bopomofo, rime-terra-pinyin, rime-stroke | LGPL-3.0 |
| `rime_cangjie5.zip` | cangjie5, cangjie5_express | rime-cangjie | LGPL-3.0 |
| `rime_jyutping.zip` | jyut6ping3, its dictionaries, `essay-cantonese.txt`, its OpenCC table | rime-cantonese | CC BY 4.0, ODbL for the data |

Two changes from upstream, both recorded in the packs' `SOURCE` files:

- OpenCC's configurations name compiled `.ocd2` dictionaries, which only
  OpenCC's command-line tools build. The base pack rewrites them to read the
  plain-text tables, which OpenCC loads directly, and generates the merged
  and reversed tables the way OpenCC's own build does.
- Rime Ice leaves out `cn_dicts/tencent.dict.yaml` and its import line. It is
  the largest table, and compiling it on a phone takes minutes and most of a
  gigabyte of memory.

A published pack is never rebuilt in place: installed app versions check it
against the checksum they shipped with. New content goes out under a new
name (`rime_<id>_v2.zip`).
