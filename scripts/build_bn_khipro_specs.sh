#!/usr/bin/env bash
# Publishes the Khipro team's current layout definitions as the set WM Keyboard
# installs without an app release (wasi-master/wmkeyboard#909):
# data/bn/khipro/, four .mim specs, the conformance table each spec's own
# repository keeps beside it, and bn_khipro_specs.json naming the set's
# revision and every file's size and SHA-256.
#
#   scripts/build_bn_khipro_specs.sh [path/to/WMKeyboard]
#
# Every spec is run against its table with the app's own interpreter
# (native/wm-engine, `cargo run --example khipro_check`) before anything is
# written, the check the app repeats before it installs a set. A failure is a
# spec feature the interpreter does not handle yet: fix the interpreter and
# ship it, never the table. Cases upstream left stale after a spec change
# overturned them go in STALE below, as the app's own tests list them.
#
# The revision is the time of the run, RFC 3339 UTC. Phones take a set only if
# it is newer than the specs they type with, so rerun after any upstream
# change and commit what it writes. Needs curl, cargo and python3.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="$(cd "${1:-$ROOT/../WMKeyboard}" && pwd)"
OUT="$ROOT/data/bn/khipro"
RAW="https://raw.githubusercontent.com"

# (input, stale expectation), tab-separated: see STALE_UPSTREAM in the app's
# native/wm-engine/src/khipro.rs for why each is stale.
STALE=$'kso/\tকসঁ\n........\t.......'

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# variant key, upstream repository, spec file there, local spec name, local table name
SETS=(
    "desktop rank-coder/khipro-m17n bn-khipro-classic.mim bn-khipro.mim khipro-testcases.tsv"
    "touchscreen KhiproTeam/khipro-mim-touchscreen bn-khipro-classic.mim bn-khipro-touchscreen.mim khipro-testcases-touch.tsv"
    "dark_desktop KhiproTeam/khipro-dark-m17n bn-khipro-dark.mim bn-khipro-dark.mim khipro-dark-testcases.tsv"
    "dark_touchscreen KhiproTeam/khipro-dark-mim-touchscreen bn-khipro-dark.mim bn-khipro-dark-touchscreen.mim khipro-dark-testcases-touch.tsv"
)

printf 'input\texpected\n%s\n' "$STALE" > "$WORK/stale.tsv"
(cd "$APP/native/wm-engine" && cargo build -q --example khipro_check)
CHECK="$APP/native/wm-engine/target/debug/examples/khipro_check"

: > "$WORK/upstream.txt"
for set in "${SETS[@]}"; do
    read -r key repo spec name table <<< "$set"
    commit="$(curl -fsSL "https://api.github.com/repos/$repo/commits?per_page=1" | python3 -c 'import json,sys; print(json.load(sys.stdin)[0]["sha"])')"
    curl -fsSL "$RAW/$repo/$commit/$spec" -o "$WORK/$name"
    curl -fsSL "$RAW/$repo/$commit/khipro-testcases.tsv" -o "$WORK/$table"
    "$CHECK" "$WORK/$name" "$WORK/$table" "$WORK/stale.tsv"
    printf '%s %s %s %s %s\n' "$key" "$repo" "${commit:0:12}" "$name" "$table" >> "$WORK/upstream.txt"
done

mkdir -p "$OUT"
for set in "${SETS[@]}"; do
    read -r _ _ _ name table <<< "$set"
    cp "$WORK/$name" "$WORK/$table" "$OUT/"
done

python3 - "$OUT" "$WORK/upstream.txt" "$STALE" <<'EOF'
import datetime, hashlib, json, os, sys

out, upstream_file, stale = sys.argv[1], sys.argv[2], sys.argv[3]

def ref(name):
    data = open(os.path.join(out, name), "rb").read()
    return {"name": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}

files, upstream = {}, {}
for line in open(upstream_file, encoding="utf-8"):
    key, repo, commit, name, table = line.split()
    files[key] = {"spec": ref(name), "cases": ref(table)}
    upstream[repo] = commit
manifest = {
    "format": 1,
    "revision": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "upstream": upstream,
    "files": files,
    "stale": [line.split("\t") for line in stale.split("\n") if line],
}
with open(os.path.join(out, "bn_khipro_specs.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
    f.write("\n")
print(f"{out}/bn_khipro_specs.json: revision {manifest['revision']}")
EOF
