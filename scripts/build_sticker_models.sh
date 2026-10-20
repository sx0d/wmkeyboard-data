#!/usr/bin/env bash
# Builds the sticker generator's files for this repository's `models` release:
# SDXS-512-DreamShaper as an MNN bundle (WM Keyboard's tools/imagegen/convert_sd.py)
# and Falconsai's NSFW classifier as MNN, both from pinned Hugging Face commits.
# Upload with:
#   scripts/upload_embedding_models.sh <out> sdxs-ds-
#   scripts/upload_embedding_models.sh <out> nsfw-
#
#   scripts/build_sticker_models.sh <path to WMKeyboard checkout> <out dir>
#
# Needs a Python with: torch diffusers transformers safetensors onnx MNN
# (PYTHON=..., default python3). About 4 GB of downloads and 6 GB of RAM.
set -euo pipefail
APP=${1:?usage: $0 <WMKeyboard checkout> <out dir>}
OUT=${2:?usage: $0 <WMKeyboard checkout> <out dir>}
PY=${PYTHON:-python3}
mkdir -p "$OUT"

SDXS_REPO=IDKiro/sdxs-512-dreamshaper
SDXS_REV=76f720262bb051da75666b22c902a78c8e16c763
NSFW_REPO=Falconsai/nsfw_image_detection
NSFW_REV=96cb0d0342c7afb80cab76ecc58b265fa44da256

"$PY" "$APP/tools/imagegen/convert_sd.py" "$SDXS_REPO" "$OUT" --revision "$SDXS_REV" \
    --name "SDXS Dreamshaper" --steps 1 --timesteps 999 --guidance 0 --licence openrail++ --prefix sdxs-ds-
# The app builds CLIP's vocabulary from the merges; the vocab file is not published.
rm -f "$OUT/sdxs-ds-vocab.json"

"$PY" - "$NSFW_REPO" "$NSFW_REV" "$OUT/nsfw-vit-falconsai.mnn" <<'PY'
import os, sys, tempfile, torch, _tools
from transformers import AutoModelForImageClassification
repo, rev, out = sys.argv[1:]
m = AutoModelForImageClassification.from_pretrained(repo, revision=rev, torch_dtype=torch.float32).eval()
class W(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, x): return torch.softmax(s.m(pixel_values=x).logits, -1)
with tempfile.TemporaryDirectory() as tmp:
    torch.onnx.export(W(m), (torch.randn(1, 3, 224, 224),), f"{tmp}/n.onnx", input_names=["image"],
                      output_names=["probs"], opset_version=17, dynamo=False)
    # The converter library, not the mnnconvert script, which reports every run to Alibaba Cloud.
    _tools.mnnconvert(["mnnconvert", "-f", "ONNX", "--modelFile", f"{tmp}/n.onnx", "--MNNModel", out,
                       "--bizCode", "wmkeyboard", "--weightQuantBits", "8"])
print(out, os.path.getsize(out))
PY
for f in "$OUT"/sdxs-ds-* "$OUT"/nsfw-*; do
  printf '%12s  %s  %s\n' "$(wc -c < "$f" | tr -d ' ')" "$(shasum -a 256 "$f" | cut -d' ' -f1)" "$(basename "$f")"
done
