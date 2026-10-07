#!/usr/bin/env bash
# Rebuilds data/bn/bn_khipro_glide.txt.gz, the Khipro spellings of every word
# of data/bn/bn_full.txt.gz (wasi-master/wmkeyboard#593).
#
# Compiles WM Keyboard's own Khipro interpreter and KhiproRomanizer with
# kotlinc, so the table is exactly what the app would work out on a phone.
# Rerun whenever bn_full.txt.gz changes, or when the app bumps
# KhiproGlideSpellings.VERSION. Takes about five minutes on eight cores.
#
#   scripts/build_bn_khipro_glide.sh [path/to/WMKeyboard]
#
# Needs kotlinc and a JDK. KOTLINC and JAVA_HOME override where they are
# looked for; Android Studio's bundled copies are the macOS default.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="$(cd "${1:-$ROOT/../WMKeyboard}" && pwd)"
STUDIO="/Applications/Android Studio.app/Contents"
KOTLINC="${KOTLINC:-$STUDIO/plugins/Kotlin/kotlinc}"
export JAVA_HOME="${JAVA_HOME:-$STUDIO/jbr/Contents/Home}"
SRC="$APP/core/language/src/main/java/com/wasimaster/wmkeyboard/core/transliteration"
RES="$APP/core/language/src/main/resources"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

"$KOTLINC/bin/kotlinc" -nowarn -d "$WORK/classes" \
    "$SRC/Khipro.kt" "$SRC/KhiproRomanizer.kt" "$ROOT/scripts/khipro_glide/KhiproGlideTable.kt"
"$JAVA_HOME/bin/java" -Xmx4g -cp "$WORK/classes:$RES:$KOTLINC/lib/kotlin-stdlib.jar" KhiproGlideTableKt \
    "$ROOT/data/bn/bn_full.txt.gz" "$ROOT/data/bn/bn_khipro_glide.txt.gz"
