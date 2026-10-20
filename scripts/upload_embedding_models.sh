#!/usr/bin/env bash
# Uploads the files scripts/repack_embeddinggemma2.py wrote to this
# repository's `models` release, each as `<sha256>-<name>`, the address WM
# Keyboard's meaning-based search downloads from. Existing assets are kept:
# a pinned file is never replaced, and a new file has a new digest.
#
#   scripts/upload_embedding_models.sh models/embeddings
#   scripts/upload_embedding_models.sh models/indicxlit indicxlit-
#
# The second argument is the file name prefix to upload (eg2- by default):
# export_indicxlit.py's files go up the same way.
set -euo pipefail
dir=${1:?usage: $0 <directory written by repack_embeddinggemma2.py> [prefix]}
prefix=${2:-eg2-}
repo=wasi-master/wmkeyboard-data
tag=models
gh release view "$tag" --repo "$repo" >/dev/null 2>&1 ||
  gh release create "$tag" --repo "$repo" --title "Models" \
    --notes "Neural network files WM Keyboard downloads on demand. See models/README.md."
existing=$(gh release view "$tag" --repo "$repo" --json assets --jq '.assets[].name')
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
for file in "$dir"/"$prefix"*; do
  name=$(basename "$file")
  sha=$(shasum -a 256 "$file" | cut -d' ' -f1)
  asset="$sha-$name"
  if grep -qx "$asset" <<<"$existing"; then
    echo "have $asset"
    continue
  fi
  ln -s "$(cd "$(dirname "$file")" && pwd)/$name" "$stage/$asset"
  echo "upload $asset"
  gh release upload "$tag" "$stage/$asset" --repo "$repo"
done
