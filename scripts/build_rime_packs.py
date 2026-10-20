#!/usr/bin/env python3
"""Builds the RIME schema packs in cjk/rime/ from upstream at pinned commits.

WM Keyboard's optional RIME engine (librime, native/librime-jni in the app
repository) reads a shared data directory of schema and dictionary YAML. The
app downloads these zips through its CJK pack path, checks each against the
SHA-256 pinned in RimeSchemaCatalog, unpacks them into that directory and
compiles them on the phone ("deploy").

Every pack carries its upstream licence and a SOURCE file naming the commit
it was cut from, under licenses/<repo>/, where librime never looks.

  python3 scripts/build_rime_packs.py            # fetch upstream, write cjk/rime/
  python3 scripts/build_rime_packs.py --src DIR  # use clones already in DIR

The zips are deterministic for a given Python (sorted entries, fixed
timestamps and modes), so a rebuild of an unchanged pin gives the same bytes.
A published pack is never rebuilt in place: new content goes out under a new
file name (rime_<id>_v2.zip), as the app's catalog expects.

Prints each pack's size and SHA-256 for RimeSchemaCatalog.kt.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "cjk" / "rime"

# repo -> (commit, licence files)
UPSTREAM = {
    "rime/rime-prelude": ("082425ea0684bca36474415d4a0e8db9b016487e", ["LICENSE", "AUTHORS"]),
    "rime/rime-essay": ("054920de4f54c9e5994276a96a4fc2a35cb51aa3", ["LICENSE", "AUTHORS"]),
    "rime/rime-luna-pinyin": ("56b934b099dfbeab842320f13aa8b461a6ab3e42", ["LICENSE", "AUTHORS"]),
    "rime/rime-stroke": ("1e8fff9b9494ddec23b0cbc526bcfd8171a6fd48", ["LICENSE", "AUTHORS"]),
    "rime/rime-bopomofo": ("6085c9a38a4a728047862b33d67eee18aa86f3b9", ["LICENSE", "AUTHORS"]),
    "rime/rime-terra-pinyin": ("723e51bc266cf9464530c1ddedb856aa18e3da34", ["LICENSE", "AUTHORS"]),
    "rime/rime-cangjie": ("52d90a1b1312e74042b38c1cbc8142defbc53171", ["LICENSE", "AUTHORS"]),
    "rime/rime-cantonese": ("259f0e48bba840c3a2e0d117539e96937f3d89bc", ["LICENSE-CC-BY", "LICENSE-ODbL"]),
    "iDvel/rime-ice": ("da1fbe602e38f26db846fa10120ee64c2b0324c0", ["LICENSE"]),
    # OpenCC's conversion tables, at the commit librime 1.17.0 pins as its
    # deps/opencc submodule, so the data matches the library that reads it.
    "BYVoid/OpenCC": ("556ed22496d650bd0b13b6c163be9814637970ae", ["LICENSE", "AUTHORS"]),
}

# What a pack changes in upstream's files, said in its SOURCE file.
MODIFIED = {
    "iDvel/rime-ice": "cn_dicts/tencent.dict.yaml left out, and its import line in rime_ice.dict.yaml removed.",
    "BYVoid/OpenCC": "configurations rewritten to read the text tables; the merged and reversed tables generated as OpenCC's build does.",
}

# OpenCC configurations the packed schemas name (opencc_config: ...), plus
# the regional ones a user's own schema is likely to.
OPENCC_CONFIGS = [
    "s2t", "t2s", "s2tw", "s2twp", "tw2s", "tw2sp", "s2hk", "hk2s", "t2tw", "t2hk",
]

STAMP = (1980, 1, 1, 0, 0, 0)


def fetch(src: Path, repo: str, commit: str) -> Path:
    """A checkout of repo at commit under src, cloned if it is not there."""
    dest = src / repo.split("/")[1]
    if not dest.exists():
        subprocess.run(["git", "init", "-q", str(dest)], check=True)
        subprocess.run(["git", "-C", str(dest), "remote", "add", "origin", f"https://github.com/{repo}"], check=True)
    have = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if have != commit:
        subprocess.run(["git", "-C", str(dest), "fetch", "-q", "--depth", "1", "origin", commit], check=True)
        subprocess.run(["git", "-C", str(dest), "checkout", "-q", "--detach", commit], check=True)
    return dest


def opencc_files(opencc: Path) -> dict:
    """OpenCC's configurations rewritten to read the plain-text tables.

    The library's own configurations name compiled .ocd2 dictionaries, which
    only OpenCC's command-line tools can build; the text tables they are made
    from load just as well (OpenCC's "text" dictionary type), so the packs ship
    those instead and nothing has to be compiled for OpenCC at all.
    """
    data = opencc / "data"
    tables = {}
    files = {}

    def parse(text: str) -> list:
        pairs = []
        for line in text.splitlines():
            if "\t" in line:
                key, value = line.split("\t", 1)
                pairs.append((key, value.rstrip("\r\n")))
        return pairs

    def render(pairs) -> bytes:
        return "".join(f"{k}\t{v}\n" for k, v in pairs).encode("utf-8")

    # The tables OpenCC's build generates (data/CMakeLists.txt), made the way
    # its scripts make them: merge.py sorts the union, reverse.py swaps every
    # key for each of its values.
    def generated(stem: str) -> list:
        source = data / "dictionary" / f"{stem}.txt"
        if source.exists():
            return parse(source.read_text(encoding="utf-8"))
        if stem == "TWPhrases":
            merged = {}
            for part in ("IT", "Name", "Other"):
                merged.update(parse((data / "dictionary" / f"TWPhrases{part}.txt").read_text(encoding="utf-8")))
            return sorted(merged.items())
        if stem.endswith("Rev"):
            reverse = {}
            for key, value in generated(stem[: -len("Rev")]):
                for v in value.split(" "):
                    reverse.setdefault(v, []).append(key)
            return [(k, " ".join(reverse[k])) for k in sorted(reverse)]
        raise SystemExit(f"no text table for {stem}")

    def table(name: str) -> str:
        if name in tables:
            return tables[name]
        stem = name[: -len(".ocd2")]
        files[f"opencc/{stem}.txt"] = render(generated(stem))
        tables[name] = f"{stem}.txt"
        return tables[name]

    def rewrite(node):
        if isinstance(node, dict):
            if node.get("type") == "ocd2":
                node["type"] = "text"
                node["file"] = table(node["file"])
            for value in node.values():
                rewrite(value)
        elif isinstance(node, list):
            for value in node:
                rewrite(value)

    for name in OPENCC_CONFIGS:
        config = json.loads((data / "config" / f"{name}.json").read_text(encoding="utf-8"))
        rewrite(config)
        files[f"opencc/{name}.json"] = (json.dumps(config, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    return files


def pick(repo_dir: Path, names) -> dict:
    return {name: (repo_dir / name).read_bytes() for name in names}


def tree(repo_dir: Path, sub: str) -> dict:
    base = repo_dir / sub
    return {
        f"{sub}/{p.relative_to(base).as_posix()}": p.read_bytes()
        for p in sorted(base.rglob("*"))
        if p.is_file()
    }


def licences(src: Path, repo: str) -> dict:
    commit, names = UPSTREAM[repo]
    repo_dir = fetch(src, repo, commit)
    short = repo.split("/")[1]
    out = {f"licenses/{short}/{n}": (repo_dir / n).read_bytes() for n in names}
    note = MODIFIED.get(repo)
    source = f"https://github.com/{repo}\ncommit {commit}\n" + (f"\nChanged here: {note}\n" if note else "")
    out[f"licenses/{short}/SOURCE"] = source.encode()
    return out


def packs(src: Path) -> dict:
    get = lambda repo: fetch(src, repo, UPSTREAM[repo][0])  # noqa: E731
    prelude, essay = get("rime/rime-prelude"), get("rime/rime-essay")
    luna, stroke = get("rime/rime-luna-pinyin"), get("rime/rime-stroke")
    bopomofo, terra = get("rime/rime-bopomofo"), get("rime/rime-terra-pinyin")
    cangjie, cantonese = get("rime/rime-cangjie"), get("rime/rime-cantonese")
    ice, opencc = get("iDvel/rime-ice"), get("BYVoid/OpenCC")

    stroke_files = pick(stroke, ["stroke.schema.yaml", "stroke.dict.yaml"])
    out = {}
    # What every schema stands on: the default configuration, punctuation and
    # symbols, the essay word-frequency table, and OpenCC for the switches
    # between Simplified and Traditional.
    out["base"] = {
        **pick(prelude, ["default.yaml", "key_bindings.yaml", "punctuation.yaml", "symbols.yaml"]),
        **pick(essay, ["essay.txt"]),
        **opencc_files(opencc),
        **licences(src, "rime/rime-prelude"),
        **licences(src, "rime/rime-essay"),
        **licences(src, "BYVoid/OpenCC"),
    }
    out["luna_pinyin"] = {
        **pick(luna, sorted(p.name for p in luna.glob("*.yaml"))),
        **stroke_files,
        **licences(src, "rime/rime-luna-pinyin"),
        **licences(src, "rime/rime-stroke"),
    }
    out["bopomofo"] = {
        **pick(bopomofo, sorted(p.name for p in bopomofo.glob("*.yaml"))),
        **pick(terra, sorted(p.name for p in terra.glob("*.yaml"))),
        **stroke_files,
        **licences(src, "rime/rime-bopomofo"),
        **licences(src, "rime/rime-terra-pinyin"),
        **licences(src, "rime/rime-stroke"),
    }
    out["cangjie5"] = {
        **pick(cangjie, sorted(p.name for p in cangjie.glob("*.yaml"))),
        **licences(src, "rime/rime-cangjie"),
    }
    out["jyutping"] = {
        **pick(cantonese, sorted(p.name for p in cantonese.glob("*.yaml")) + ["essay-cantonese.txt"]),
        **tree(cantonese, "opencc"),
        **licences(src, "rime/rime-cantonese"),
    }
    # rime-ice without its own default.yaml (the base pack's serves its
    # includes), its Lua scripts (this build has no Lua plugin; the schema
    # skips the components it cannot load) and the desktop front ends' files.
    ice_yaml = sorted(
        p.name for p in ice.glob("*.yaml") if p.name not in {"default.yaml", "squirrel.yaml", "weasel.yaml", "recipe.yaml"}
    )
    ice_files = pick(ice, ice_yaml + ["custom_phrase.txt"])
    # Without the Tencent word-vector list: it is the largest table, and
    # compiling it on a phone takes minutes and most of a gigabyte of memory.
    # The import line goes with it, the one change made to upstream's files.
    dict_yaml = ice_files["rime_ice.dict.yaml"].decode("utf-8")
    tencent = "  - cn_dicts/tencent"
    if tencent not in dict_yaml:
        raise SystemExit("rime_ice.dict.yaml no longer imports cn_dicts/tencent; check the pack")
    ice_files["rime_ice.dict.yaml"] = "\n".join(
        line for line in dict_yaml.split("\n") if not line.startswith(tencent)
    ).encode("utf-8")
    out["ice"] = {
        **ice_files,
        **{k: v for k, v in tree(ice, "cn_dicts").items() if k != "cn_dicts/tencent.dict.yaml"},
        **tree(ice, "en_dicts"),
        **tree(ice, "opencc"),
        **licences(src, "iDvel/rime-ice"),
    }
    return out


def write_zip(path: Path, files: dict) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name in sorted(files):
            info = zipfile.ZipInfo(name, STAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            info.create_system = 3
            z.writestr(info, files[name])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, help="directory of upstream clones (fetched into it when missing)")
    args = parser.parse_args()
    src = args.src or Path(tempfile.mkdtemp(prefix="rime-upstream-"))
    src.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for pack_id, files in packs(src).items():
        path = OUT / f"rime_{pack_id}.zip"
        write_zip(path, files)
        data = path.read_bytes()
        rows.append((pack_id, path.name, len(data), hashlib.sha256(data).hexdigest()))
    with open(OUT / "packs.tsv", "w", encoding="utf-8") as f:
        f.write("# id\tfile\tbytes\tsha256\n")
        for row in rows:
            f.write("\t".join(str(v) for v in row) + "\n")
    for pack_id, name, size, sha in rows:
        print(f"{pack_id:12} {name:24} {size:>10} {sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
