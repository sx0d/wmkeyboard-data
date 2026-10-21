#!/usr/bin/env python3
"""Exports PaddleOCR's PP-OCRv5 mobile detector and recogniser to MNN for this
repository's `models` release: the scanner's Text scan reader on builds that
have no LiteRT (F-Droid's). Upload with

    scripts/upload_embedding_models.sh <out dir> ppocrv5-

    python3 scripts/export_ppocrv5_mnn.py <out dir> [--check <dir with the LiteRT pair>]

Needs a Python with: paddlepaddle paddle2onnx onnx polygraphy onnx_graphsurgeon
sympy huggingface_hub MNN numpy (MNN 3.6.1, the version libwmimagegen is built
from). --check also needs ai-edge-litert, and compares against the two
`ocr_*_fp16.tflite` files the app downloads on LiteRT builds.

What it does, per graph:
1. PaddlePaddle's own inference model, pinned to a Hugging Face commit and
   checked against the SHA-256 below, to ONNX with paddle2onnx (opset 14).
2. An NHWC input in front (`image`, [1, H, W, 3], a Transpose to Paddle's
   NCHW), fixed shapes, and the output renamed `probs`: the same tensors the
   LiteRT graphs take and give, so the app's pipeline feeds both alike.
3. Every float constant rounded through float16 and back. The LiteRT graphs
   the app runs elsewhere carry float16 weights; on the same inputs the
   rounded graph reads the same text as they do, where the unrounded one
   differs on 2 lines in 77 (measured on a 14-page set, 2026-10-21).
4. MNN, weights stored as float32. MNN's own --fp16 storage rounds again
   after its BatchNorm folding and misreads lines the LiteRT graphs read
   right, so it is not used even though it would halve the files.

The converter is called as a library: the `mnnconvert` script reports each
run to Alibaba Cloud. MNN writes a random id into every file, so a rebuild
has new bytes and needs new pins in the app (PaddleOcrModel.kt).
"""
import argparse
import hashlib
import os
import sys
import tempfile

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

SOURCES = {
    "det": ("PaddlePaddle/PP-OCRv5_mobile_det", "0d63e78e2b680928f6b1747d76a08db6e645efb7", (640, 640), {
        "inference.json": "05feef1acb00aa4cd7362b15f7f501fc4f99d7b1fa73c1c871e0c7b1504b0f5c",
        "inference.pdiparams": "afa1820cb16c1fd0dad589d0f8b389139061c1ef6d68019685fd07be997dda5b",
    }),
    "rec": ("PaddlePaddle/PP-OCRv5_mobile_rec", "682f20538d8c086cb2128e5cfac775e6c4904e85", (48, 320), {
        "inference.json": "24587345250c7332d0fc6f9a44e794d078cdaeb64c302fef906f325619de2569",
        "inference.pdiparams": "2460da90875937c94db97eba74ae3d9e5d4c4c57c42f1f41531c09a26bcc771a",
        "inference.yml": "5dfeb2777f6d0db8177d8128a8acfcf6e6276dc4ac73ea3bf0dc06d6a5e85d8e",
    }),
}
# The character list the app already downloads beside the LiteRT pair
# (iFleey/PPOCRv5-Android, keys_v5.txt): the same 18,383 lines as the
# recogniser's inference.yml, checked below.
KEYS_SHA256 = "d1979e9f794c464c0d2e0b70a7fe14dd978e9dc644c0e71f14158cdf8342af1b"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(kind, work):
    from huggingface_hub import hf_hub_download
    repo, rev, _, files = SOURCES[kind]
    d = os.path.join(work, kind)
    for name, want in files.items():
        path = hf_hub_download(repo, name, revision=rev, local_dir=d)
        got = sha256(path)
        if got != want:
            sys.exit(f"{repo}@{rev} {name}: sha256 {got}, expected {want}")
    return d


def nhwc(model, h, w):
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
    g.input.insert(0, helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, h, w, 3]))
    out = g.output[0]
    for n in g.node:
        for i, x in enumerate(n.output):
            if x == out.name:
                n.output[i] = "probs"
        for i, x in enumerate(n.input):
            if x == out.name:
                n.input[i] = "probs"
    g.output.remove(out)
    g.output.append(helper.make_tensor_value_info("probs", TensorProto.FLOAT, None))
    del g.value_info[:]
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    return model


def round_fp16(model):
    """Every float constant of more than one value through float16 and back."""
    def rounded(t):
        a = numpy_helper.to_array(t)
        if a.dtype == np.float32 and a.size > 1:
            return numpy_helper.from_array(a.astype(np.float16).astype(np.float32), t.name)
        return None
    for t in model.graph.initializer:
        r = rounded(t)
        if r is not None:
            t.CopyFrom(r)
    for node in model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.t and attr.t.data_type == TensorProto.FLOAT:
                    r = rounded(attr.t)
                    if r is not None:
                        attr.t.CopyFrom(r)
    return model


def export(kind, work, out):
    import paddle2onnx
    import _tools  # MNN's converter library
    d = fetch(kind, work)
    if kind == "rec":
        import yaml
        with open(os.path.join(d, "inference.yml"), encoding="utf-8") as f:
            chars = yaml.safe_load(f)["PostProcess"]["character_dict"]
        keys = "".join(c + "\n" for c in chars).encode("utf-8")
        if hashlib.sha256(keys).hexdigest() != KEYS_SHA256:
            sys.exit("the recogniser's character list is not keys_v5.txt")
    raw = os.path.join(work, f"{kind}.onnx")
    paddle2onnx.export(os.path.join(d, "inference.json"), os.path.join(d, "inference.pdiparams"), raw, opset_version=14)
    h, w = SOURCES[kind][2]
    fixed = os.path.join(work, f"{kind}-nhwc-fp16w.onnx")
    onnx.save(round_fp16(nhwc(onnx.load(raw), h, w)), fixed)
    target = os.path.join(out, f"ppocrv5-{kind}.mnn")
    _tools.mnnconvert(["mnnconvert", "-f", "ONNX", "--modelFile", fixed, "--MNNModel", target, "--bizCode", "wmkeyboard"])
    return target


def check(out, litert_dir):
    """The MNN graphs against the LiteRT pair on a few inputs."""
    import MNN
    from ai_edge_litert.interpreter import Interpreter
    F = MNN.expr
    rs = np.random.RandomState(0)
    for kind, shape in (("det", (1, 640, 640, 3)), ("rec", (1, 48, 320, 3))):
        rt = MNN.nn.create_runtime_manager(({"backend": "CPU", "precision": "normal", "numThread": 4},))
        mod = MNN.nn.load_module_from_file(os.path.join(out, f"ppocrv5-{kind}.mnn"), ["image"], ["probs"],
                                           runtime_manager=rt, shape_mutable=True)
        it = Interpreter(os.path.join(litert_dir, f"ocr_{kind}_fp16.tflite"))
        it.allocate_tensors()
        worst = 0.0
        for _ in range(4):
            x = ((rs.rand(*shape) - 0.5) * 3).astype(np.float32)
            it.set_tensor(it.get_input_details()[0]["index"], x)
            it.invoke()
            ref = it.get_tensor(it.get_output_details()[0]["index"]).reshape(-1)
            v = F.placeholder(list(shape), F.NCHW)
            v.write(x)
            y = mod.forward([v])[0]
            if y.data_format == F.NC4HW4:
                y = F.convert(y, F.NCHW)
            y = np.array(y.read(), dtype=np.float32).reshape(-1)
            worst = max(worst, float(np.abs(y - ref).max()))
        print(f"{kind}: largest difference from LiteRT on random inputs {worst:.3g}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--check", metavar="LITERT_DIR")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    with tempfile.TemporaryDirectory() as work:
        files = [export(k, work, a.out) for k in ("det", "rec")]
    for f in files:
        print(f"{os.path.getsize(f):>12}  {sha256(f)}  {os.path.basename(f)}")
    if a.check:
        check(a.out, a.check)


if __name__ == "__main__":
    main()
