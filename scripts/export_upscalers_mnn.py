#!/usr/bin/env python3
"""Exports waifu2x's UpConv7 networks to MNN for this repository's `models`
release: the keyboard's Upscale, in the sticker editor, the photo editor and
on small stickers sent to an app that wants a 512 square. Upload with

    scripts/upload_embedding_models.sh <out dir> sr-

    python3 scripts/export_upscalers_mnn.py <out dir>

Needs a Python with: torch onnx MNN numpy (MNN 3.6.1, the version
libwmimagegen is built from).

What it does, per network:
1. nagadomi/waifu2x's own weights for `upconv_7` (the torch7 model as JSON),
   pinned to one commit and checked against the SHA-256 below.
2. The same seven layers in PyTorch: six unpadded 3x3 convolutions with
   leaky ReLU (0.1), then a 4x4 transposed convolution, stride 2, padding 3.
   A tile of side n + 14 comes out as exactly 2n: the network eats 7 pixels
   on each side, which is the context the app reads around every tile.
3. ONNX (opset 17), input `image` [1, 3, H, W] RGB from 0 to 1, output
   `upscaled` [1, 3, 2(H - 14), 2(W - 14)], height and width free.
4. MNN, weights stored as float32 (2.2 MB each; float16 storage would save
   1.1 MB and is not worth a second rounding).

Checked here: MNN against the PyTorch network on random tiles (largest
difference 0.0015 on a 0 to 1 scale, under half of one 8-bit step).

The converter is called as a library: the `mnnconvert` script reports each
run to Alibaba Cloud. MNN writes a random id into every file, so a rebuild has
new bytes and needs new pins in the app (UpscaleModels.kt).
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile
import urllib.request

import numpy as np

COMMIT = "cc385f97a9debfe611316aabfd5d8bb30ba2dbeb"
# out name: (path in nagadomi/waifu2x, sha256 of that file)
SOURCES = {
    "sr-waifu2x-upconv7-art.mnn": (
        "models/upconv_7/art/noise1_scale2.0x_model.json",
        "6c0688159ab0023619d1b0e472fead370619df1d36251f14f299edc43b30d1e3",
    ),
    "sr-waifu2x-upconv7-photo.mnn": (
        "models/upconv_7/photo/noise1_scale2.0x_model.json",
        "adbce4a532d42bc96f17fea8a5aec9245c509ff44058263938ea232244dff0bc",
    ),
}
CONTEXT = 7


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(path, want, work):
    target = os.path.join(work, os.path.basename(os.path.dirname(path)) + "-" + os.path.basename(path))
    url = f"https://raw.githubusercontent.com/nagadomi/waifu2x/{COMMIT}/{path}"
    urllib.request.urlretrieve(url, target)
    got = sha256(target)
    if got != want:
        sys.exit(f"{path}: sha256 {got}, expected {want}")
    return target


def network(json_path):
    import torch
    import torch.nn as nn

    class UpConv7(nn.Module):
        def __init__(self):
            super().__init__()
            ch = [3, 16, 32, 64, 128, 128, 256]
            self.convs = nn.ModuleList([nn.Conv2d(ch[i], ch[i + 1], 3) for i in range(6)])
            self.act = nn.LeakyReLU(0.1)
            self.up = nn.ConvTranspose2d(256, 3, 4, 2, 3)

        def forward(self, x):
            for conv in self.convs:
                x = self.act(conv(x))
            return self.up(x)

    with open(json_path) as f:
        layers = json.load(f)
    model = UpConv7()
    modules = list(model.convs) + [model.up]
    if len(layers) != len(modules):
        sys.exit(f"{json_path}: {len(layers)} layers, expected {len(modules)}")
    for layer, module in zip(layers, modules):
        w = torch.tensor(np.array(layer["weight"], np.float32))
        b = torch.tensor(np.array(layer["bias"], np.float32))
        if w.numel() != module.weight.numel() or b.numel() != module.bias.numel():
            sys.exit(f"{json_path}: layer shapes differ from upconv_7")
        module.weight.data = w.reshape(module.weight.shape)
        module.bias.data = b.reshape(module.bias.shape)
    return model.eval()


def export(name, work, out):
    import torch
    import _tools  # MNN's converter library
    path, want = SOURCES[name]
    model = network(fetch(path, want, work))
    onnx_path = os.path.join(work, name.replace(".mnn", ".onnx"))
    torch.onnx.export(
        model, torch.rand(1, 3, 64, 64), onnx_path,
        input_names=["image"], output_names=["upscaled"], opset_version=17,
        dynamic_axes={"image": {2: "h", 3: "w"}, "upscaled": {2: "H", 3: "W"}}, dynamo=False,
    )
    target = os.path.join(out, name)
    _tools.mnnconvert(["mnnconvert", "-f", "ONNX", "--modelFile", onnx_path, "--MNNModel", target, "--bizCode", "wmkeyboard"])
    check(model, target)
    return target


def check(model, target):
    """MNN against the PyTorch network on random tiles of two sizes."""
    import torch
    import MNN
    F = MNN.expr
    rt = MNN.nn.create_runtime_manager(({"backend": "CPU", "precision": "normal", "numThread": 4},))
    mod = MNN.nn.load_module_from_file(target, ["image"], ["upscaled"], runtime_manager=rt, shape_mutable=True)
    rs = np.random.RandomState(0)
    worst = 0.0
    for side in (64 + 2 * CONTEXT, 256 + 2 * CONTEXT):
        x = rs.rand(1, 3, side, side).astype(np.float32)
        with torch.no_grad():
            ref = model(torch.from_numpy(x)).numpy()
        v = F.placeholder([1, 3, side, side], F.NCHW)
        v.write(x)
        y = mod.forward([v])[0]
        if y.data_format == F.NC4HW4:
            y = F.convert(y, F.NCHW)
        y = np.array(y.read(), dtype=np.float32).reshape(ref.shape)
        if ref.shape != (1, 3, 2 * (side - 2 * CONTEXT), 2 * (side - 2 * CONTEXT)):
            sys.exit(f"{target}: output {ref.shape} for a {side} tile")
        worst = max(worst, float(np.abs(y - ref).max()))
    print(f"{os.path.basename(target)}: largest difference from PyTorch on random tiles {worst:.3g}")
    # MNN's Winograd convolutions add in another order: well under half of
    # one 8-bit step is expected, a whole step is not.
    if worst > 1 / 255:
        sys.exit("MNN does not draw what PyTorch draws")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    with tempfile.TemporaryDirectory() as work:
        files = [export(name, work, a.out) for name in SOURCES]
    for f in files:
        print(f"{os.path.getsize(f):>12}  {sha256(f)}  {os.path.basename(f)}")


if __name__ == "__main__":
    main()
