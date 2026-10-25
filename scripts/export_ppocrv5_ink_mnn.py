#!/usr/bin/env python3
"""Exports PaddleOCR's per-script PP-OCRv5 mobile recognisers to MNN for this
repository's `models` release: the keyboard's handwriting reader for the
languages ML Kit is not there for (F-Droid's build) and zinnia does not read.
The keyboard draws the ink as a line of text and reads it with one of these.

    python3 scripts/export_ppocrv5_ink_mnn.py <out dir> [script ...]

Upload with

    scripts/upload_embedding_models.sh <out dir> ppocrv5-ink-

Needs a Python with: paddlepaddle paddle2onnx onnx huggingface_hub pyyaml MNN
numpy (MNN 3.6.1, the version libwmimagegen is built from).

Per script, as export_ppocrv5_mnn.py does for the scanner's recogniser:
1. PaddlePaddle's own inference model, pinned to a Hugging Face commit and
   checked against the SHA-256 below, to ONNX with paddle2onnx (opset 14).
2. An NHWC input in front (`image`, [1, 48, 320, 3]) and the output renamed
   `probs`, the tensors the keyboard's Paddle pipeline feeds and reads.
3. MNN, weights stored as float32 (MNN's --fp16 storage misreads; see the
   scanner's script).

Unlike the scanner's recogniser there is no LiteRT twin to match, so the
weights are not rounded through float16 first.

Beside each model, its character list as `ppocrv5-ink-<script>-keys.txt`, one
character per line, out of the recogniser's own inference.yml.

The converter is called as a library: the `mnnconvert` script reports each
run to Alibaba Cloud. MNN writes a random id into every file, so a rebuild
has new bytes and needs new pins in the app (InkOcrModels.kt).
"""
import argparse
import hashlib
import os
import sys
import tempfile

import onnx
from onnx import TensorProto, helper

# Hugging Face repository commit per script, all Apache-2.0.
SOURCES = {
    "en": "267c36e24c331595590fe7bd72bde2436fd286f2",
    "latin": "ab2cd5cc5fa6309be2e5acdfe66eca2c2c127d57",
    "eslav": "7553801264d3379d8d2e854971989e5e26c22e03",
    "cyrillic": "712d2d65556ccc1ea7b5d2bb232b018838b6a3ab",
    "arabic": "33d91636a65dca87f5562cc48860332ae367ee1b",
    "devanagari": "99dcce6d196bd4aaf268c7a5c72c3cc9f3ea4932",
    "ta": "1bb164dad1d8eb23c7f7a382827e5305b37868d4",
    "te": "151ab3b1c2f2a058f07a944416b92e9eaec6bf36",
    "th": "9e25080455925d3943f5db885fddc79db6a07ca3",
    "el": "b15bcf2467a03a0c079baa4b0347da51e89aaaba",
    "korean": "c02ecaf1f22bfd1c618cce154fd19185b47e663a",
}
# SHA-256 of each downloaded file at those commits (first fetched 2026-10-25).
# A script missing here is fetched unchecked and its digests printed.
DIGESTS = {
    "en": {
        "inference.json": "fd1b6ec722ea841a72d3ba43e527df1d1066d5d7808e0503ee3eec7265188753",
        "inference.pdiparams": "3ec8a97ed6cefe8568d3e2ee90bb193299b566a7661aa4fd52d224b96b59f66b",
        "inference.yml": "27e91d0582f40168aa218303c76e184bc78fa7a5d105aad0cfbad8458b441067",
    },
    "latin": {
        "inference.json": "820b2ee8e5072de5156c3d55124c2b03c82ef278d59f453f857c0789bacb4da1",
        "inference.pdiparams": "53cdc8b481a7394bb108f96d0fb3432b0a8f392e22c7d18f06dbb2d42b8b25f9",
        "inference.yml": "0bbe984570f597af3638e50bdf2e8276f3ab26a61966096538b3b0d1849f5c84",
    },
    "eslav": {
        "inference.json": "3fb6e2e658f5139ff16e35260de8f0577f106a9505c902e1dfc1f4f1d03cc9cb",
        "inference.pdiparams": "f11057b05d8517868bca505271278973d706600d9dcc184cbcf5c4512091c32b",
        "inference.yml": "025039bac23eb4a308efcefa4d58eab3af440767815c6ba6938468bf6353ee5a",
    },
    "cyrillic": {
        "inference.json": "5d90f1bfca52d80c01de176c5238fae2459995a99ff1dbfe5319ab4ed1735df2",
        "inference.pdiparams": "434dc9fa2a99fa3653e08f8cf793ae56be7dd41c35c4980e6255147cc02bbc80",
        "inference.yml": "5c76cc91fa98410178a09f498db10050d0ec1634a660053d3005ab7be581f501",
    },
    "arabic": {
        "inference.json": "d43832804d89c0e18d59f8446a3babd0ef60c44bc335efddac3176caf71d50af",
        "inference.pdiparams": "4b4271fd1dd89a40b2056e1a42e58de7c0df2fdcf1a97ca9a8916d7ec45f9143",
        "inference.yml": "21368419e6c016c31db55d316d59e11c128e1913e6e6fe10287084710043d3a6",
    },
    "devanagari": {
        "inference.json": "20d3b121184ec6b7de1705015022f4b22117990f6e5728a5d21c314bac9c1444",
        "inference.pdiparams": "719be7d20bfe9530e2deae324c999e9911087496bce5e70846767c448d023a01",
        "inference.yml": "9bd172dd26440c8ce94d1cde5d5baea6aefdc7cf3c5c8492e0beedef656d4e54",
    },
    "ta": {
        "inference.json": "9f845b15eef3272a47201fa0e411f8e466392b7dcfd705ade2890bf434b70370",
        "inference.pdiparams": "7e7bb6c9cbcc401aeaceb0cdace485dbfcc15a410f299294e7c3123bbcbc68f0",
        "inference.yml": "88a28f5a1bb30cabe38a0985cb5e6619fa4f0c7c78e57a08274674228c5219a6",
    },
    "te": {
        "inference.json": "ad866e22872fbc0ee498b2c47d9537bd8f49f125353e43d82a4d173b595e6bfb",
        "inference.pdiparams": "45967d00d6b4af590221733bf0d93791babc1feb17b98da401dba53d3cf110c9",
        "inference.yml": "acebbe53f1831bf28ddfed75aedf58225d7aa5d09100c1d5a9140a2a53b137ce",
    },
    "th": {
        "inference.json": "b92a9442af5ac6da3c6ba30ce1672d00a2d7a0fbd8068ed543ac6dcb9d5c7764",
        "inference.pdiparams": "45ec91f2322b58b8d30ba27d18fcfdbb8bf388b918dd978162d2af91e0c66d4b",
        "inference.yml": "f6ba7fefc38ca1ff398ddafa75d67d16e0b3757c4e6c833adffee98a981766c9",
    },
    "el": {
        "inference.json": "7fb97d32c52562deb0a0654e0839acbb54cacc0441a857cc1d66e12892337e41",
        "inference.pdiparams": "4d69bbb8ed9f84373631d121ab459f8583cd978df813b5fb9b139b7783b05fbd",
        "inference.yml": "17d85b2fe2d2f24cd4ab07bcbc33e0c126859b956ced36e281dc65e2d0c1f0bf",
    },
    "korean": {
        "inference.json": "562404e3c590c50c93778d5f0a94df21b47b5ab8f3ea6d47c7f8a7930c3bc844",
        "inference.pdiparams": "cac3e5f12cf04aaa77f6a5bc704e4e736ef2908476551891d84b41b4e9090462",
        "inference.yml": "f757fa1c40e99edcf27e9cce879b93eb2a51fa46f5ef39095689b8c37dd75998",
    },
}
FILES = ("inference.json", "inference.pdiparams", "inference.yml")
HEIGHT, WIDTH = 48, 320


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(script, work):
    from huggingface_hub import hf_hub_download
    repo = f"PaddlePaddle/{script}_PP-OCRv5_mobile_rec"
    d = os.path.join(work, script)
    pinned = DIGESTS.get(script, {})
    for name in FILES:
        path = hf_hub_download(repo, name, revision=SOURCES[script], local_dir=d)
        got = sha256(path)
        if name in pinned and got != pinned[name]:
            sys.exit(f"{repo}@{SOURCES[script]} {name}: sha256 {got}, expected {pinned[name]}")
        if name not in pinned:
            print(f"  unpinned {script} {name} {got}")
    return d


def nhwc(model):
    """An NHWC `image` input in front of Paddle's NCHW one, output `probs`."""
    g = model.graph
    old = g.input[0]
    inner = old.name + "_nchw"
    for n in g.node:
        for i, x in enumerate(n.input):
            if x == old.name:
                n.input[i] = inner
    g.node.insert(0, helper.make_node("Transpose", ["image"], [inner], perm=[0, 3, 1, 2], name="nhwc_to_nchw"))
    g.input.remove(old)
    g.input.insert(0, helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, HEIGHT, WIDTH, 3]))
    out = g.output[0]
    for n in g.node:
        for i, x in enumerate(n.output):
            if x == out.name:
                n.output[i] = "probs"
        for i, x in enumerate(n.input):
            if x == out.name:
                n.input[i] = "probs"
    g.output.remove(out)
    g.output.append(helper.make_tensor_value_info("probs", TensorProto.FLOAT, [1, "steps", "classes"]))
    del g.value_info[:]
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    return model


def export(script, work, out):
    import paddle2onnx
    import yaml
    import _tools  # MNN's converter library
    d = fetch(script, work)
    with open(os.path.join(d, "inference.yml"), encoding="utf-8") as f:
        chars = yaml.safe_load(f)["PostProcess"]["character_dict"]
    keys = os.path.join(out, f"ppocrv5-ink-{script}-keys.txt")
    with open(keys, "w", encoding="utf-8", newline="\n") as f:
        f.write("".join(str(c) + "\n" for c in chars))
    raw = os.path.join(work, f"{script}.onnx")
    paddle2onnx.export(os.path.join(d, "inference.json"), os.path.join(d, "inference.pdiparams"), raw, opset_version=14)
    fixed = os.path.join(work, f"{script}-nhwc.onnx")
    onnx.save(nhwc(onnx.load(raw)), fixed)
    target = os.path.join(out, f"ppocrv5-ink-{script}.mnn")
    _tools.mnnconvert(["mnnconvert", "-f", "ONNX", "--modelFile", fixed, "--MNNModel", target, "--bizCode", "wmkeyboard"])
    return [target, keys]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("scripts", nargs="*", default=list(SOURCES))
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = []
    with tempfile.TemporaryDirectory() as work:
        for script in a.scripts:
            files += export(script, work, a.out)
    for f in files:
        print(f"{os.path.getsize(f):>12}  {sha256(f)}  {os.path.basename(f)}")


if __name__ == "__main__":
    main()
