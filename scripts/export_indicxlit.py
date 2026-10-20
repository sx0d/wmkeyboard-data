#!/usr/bin/env python3
"""Exports AI4Bharat's IndicXlit (English letters -> 21 Indic scripts) to the
ONNX files WM Keyboard's phonetic layouts run through ONNX Runtime.

IndicXlit v1.0 is a fairseq transformer (6 encoder + 6 decoder layers, width
256, 4 heads, pre-norm, 11.5M parameters), one multilingual model whose
source is `__<lang>__ l e t t e r s </s>`. It is published as a fairseq
checkpoint inside `indicxlit-en-indic-v1.0.zip` on the IndicXlit GitHub
release (MIT). fairseq itself is not needed: the checkpoint's state dict is
loaded into the small PyTorch modules below, which follow fairseq's
TransformerEncoder/Decoder (normalize_before, layernorm_embedding, sinusoidal
positions from padding_idx + 1, embeddings scaled by sqrt(256)).

Three files come out:

* `indicxlit-encoder.int8.onnx`: `src` [1, S] int64 -> `cross_k`, `cross_v`
  [6, 1, 4, S, 64], the decoder's cross-attention keys and values, computed
  once per word;
* `indicxlit-decoder.int8.onnx`: one decoding step with its own
  self-attention cache. `token` [B, 1] int64, `pos` [1] int64 (the step,
  from 0), `self_k`/`self_v` [6, B, 4, T, 64], `cross_k`/`cross_v` ->
  `logp` [B, 806] log-probabilities, `new_k`/`new_v` [6, B, 4, T+1, 64];
* `indicxlit-vocab.txt`: `s<TAB>symbol` lines for the 54 source ids, then
  `t<TAB>symbol` lines for the 806 target ids, in id order.

Weights are quantised to int8 (dynamic, MatMul only). On the Aksharantar
test words that changes top-1 and top-4 by less than half a point.

    python3 -m venv venv && venv/bin/pip install torch==2.14.1 onnx==1.23.2 \
        onnxruntime==1.31.0 numpy omegaconf
    venv/bin/python scripts/export_indicxlit.py --out models/indicxlit [--zip indicxlit-en-indic-v1.0.zip]

`--check` then reads a few words back through the int8 graphs with the same
beam search the keyboard runs. The files are assets of the `models` release
(see models/README.md); scripts/upload_embedding_models.sh uploads them.
"""
import argparse
import hashlib
import math
import os
import tempfile
import urllib.request
import zipfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RELEASE = "https://github.com/AI4Bharat/IndicXlit/releases/download/v1.0/indicxlit-en-indic-v1.0.zip"
RELEASE_SHA256 = "31dabc448c14dfac03015763d35cc4c895d8ee2b831fe23c64a5c5aea4f169ca"
CHECKPOINT_SHA256 = "2f03fe0dfb2edc40f3c11f9293c4feda4fa8ed5366fec2981391606f224a09f7"
# fairseq adds a token per language after the dictionary file's entries, in
# the order of the release's lang_list.txt.
LANGS = "en as bn brx gom gu hi kn ks mai ml mni mr ne or pa sa sd si ta te ur".split()
PAD, EOS = 1, 2
LAYERS, HEADS, HEAD_DIM, WIDTH = 6, 4, 64, 256


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_dict(lines):
    syms = ["<s>", "<pad>", "</s>", "<unk>"]
    syms += [line.rstrip("\n").rsplit(" ", 1)[0] for line in lines]
    return syms + [f"__{lang}__" for lang in LANGS]


def sinusoidal(num, dim):
    half = dim // 2
    step = math.log(10000) / (half - 1)
    freqs = torch.exp(torch.arange(half, dtype=torch.float) * -step)
    table = torch.arange(num, dtype=torch.float).unsqueeze(1) * freqs.unsqueeze(0)
    table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
    table[PAD] = 0
    return table


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.k_proj, self.v_proj, self.q_proj, self.out_proj = (nn.Linear(WIDTH, WIDTH) for _ in range(4))

    def heads(self, x, proj):
        return proj(x).view(x.shape[0], x.shape[1], HEADS, HEAD_DIM).transpose(1, 2)

    def attend(self, q, k, v):
        a = torch.softmax(q @ k.transpose(-1, -2), dim=-1) @ v
        return self.out_proj(a.transpose(1, 2).reshape(q.shape[0], q.shape[2], WIDTH))

    def query(self, x):
        return self.heads(x, self.q_proj) * HEAD_DIM ** -0.5


class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = Attention()
        self.self_attn_layer_norm = nn.LayerNorm(WIDTH)
        self.fc1, self.fc2 = nn.Linear(WIDTH, 1024), nn.Linear(1024, WIDTH)
        self.final_layer_norm = nn.LayerNorm(WIDTH)

    def forward(self, x):
        y = self.self_attn_layer_norm(x)
        a = self.self_attn
        x = x + a.attend(a.query(y), a.heads(y, a.k_proj), a.heads(y, a.v_proj))
        return x + self.fc2(F.gelu(self.fc1(self.final_layer_norm(x))))


class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = Attention()
        self.self_attn_layer_norm = nn.LayerNorm(WIDTH)
        self.encoder_attn = Attention()
        self.encoder_attn_layer_norm = nn.LayerNorm(WIDTH)
        self.fc1, self.fc2 = nn.Linear(WIDTH, 1024), nn.Linear(1024, WIDTH)
        self.final_layer_norm = nn.LayerNorm(WIDTH)


class Encoder(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, WIDTH, padding_idx=PAD)
        self.layernorm_embedding = nn.LayerNorm(WIDTH)
        self.layers = nn.ModuleList(EncoderLayer() for _ in range(LAYERS))
        self.layer_norm = nn.LayerNorm(WIDTH)
        self.register_buffer("pos", sinusoidal(256, WIDTH), persistent=False)


class Decoder(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, WIDTH, padding_idx=PAD)
        self.layernorm_embedding = nn.LayerNorm(WIDTH)
        self.layers = nn.ModuleList(DecoderLayer() for _ in range(LAYERS))
        self.layer_norm = nn.LayerNorm(WIDTH)
        self.output_projection = nn.Linear(WIDTH, vocab, bias=False)
        self.register_buffer("pos", sinusoidal(256, WIDTH), persistent=False)


class EncoderGraph(nn.Module):
    """The encoder, and every decoder layer's cross-attention keys and values over its output."""

    def __init__(self, enc, dec):
        super().__init__()
        self.enc, self.dec = enc, dec

    def forward(self, src):
        e = self.enc
        x = math.sqrt(WIDTH) * e.embed_tokens(src) + e.pos[2:2 + src.shape[1]].unsqueeze(0)
        x = e.layernorm_embedding(x)
        for layer in e.layers:
            x = layer(x)
        x = e.layer_norm(x)
        ks = [l.encoder_attn.heads(x, l.encoder_attn.k_proj) for l in self.dec.layers]
        vs = [l.encoder_attn.heads(x, l.encoder_attn.v_proj) for l in self.dec.layers]
        return torch.stack(ks), torch.stack(vs)


class DecoderStep(nn.Module):
    def __init__(self, dec):
        super().__init__()
        self.dec = dec

    def forward(self, token, pos, self_k, self_v, cross_k, cross_v):
        d = self.dec
        x = math.sqrt(WIDTH) * d.embed_tokens(token) + d.pos[pos + 2].unsqueeze(0)
        x = d.layernorm_embedding(x)
        new_k, new_v = [], []
        for i, layer in enumerate(d.layers):
            a = layer.self_attn
            y = layer.self_attn_layer_norm(x)
            k = torch.cat([self_k[i], a.heads(y, a.k_proj)], dim=2)
            v = torch.cat([self_v[i], a.heads(y, a.v_proj)], dim=2)
            new_k.append(k)
            new_v.append(v)
            x = x + a.attend(a.query(y), k, v)
            c = layer.encoder_attn
            x = x + c.attend(c.query(layer.encoder_attn_layer_norm(x)), cross_k[i], cross_v[i])
            x = x + layer.fc2(F.gelu(layer.fc1(layer.final_layer_norm(x))))
        x = d.layer_norm(x[:, 0, :])
        return torch.log_softmax(d.output_projection(x), dim=-1), torch.stack(new_k), torch.stack(new_v)


def fetch(zip_path):
    if zip_path is None:
        zip_path = os.path.join(tempfile.mkdtemp(), "indicxlit-en-indic-v1.0.zip")
        print("fetching", RELEASE)
        urllib.request.urlretrieve(RELEASE, zip_path)
    got = sha256(zip_path)
    if got != RELEASE_SHA256:
        raise SystemExit(f"{zip_path}: sha256 {got}, expected {RELEASE_SHA256}")
    return zip_path


def export(zip_path, out):
    from onnxruntime.quantization import QuantType, quantize_dynamic

    os.makedirs(out, exist_ok=True)
    work = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_path) as z:
        src = load_dict(z.read("corpus-bin/dict.en.txt").decode("utf-8").splitlines(True))
        tgt = load_dict(z.read("corpus-bin/dict.bn.txt").decode("utf-8").splitlines(True))
        ckpt = os.path.join(work, "indicxlit.pt")
        with open(ckpt, "wb") as f:
            f.write(z.read("transformer/indicxlit.pt"))
    if sha256(ckpt) != CHECKPOINT_SHA256:
        raise SystemExit("unexpected checkpoint inside the release zip")
    state = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
    skip = ("version", "_float_tensor")
    enc, dec = Encoder(len(src)), Decoder(len(tgt))
    for module, prefix in ((enc, "encoder."), (dec, "decoder.")):
        module.load_state_dict({k[len(prefix):]: v for k, v in state.items()
                                if k.startswith(prefix) and not any(s in k for s in skip)})
    enc.eval()
    dec.eval()

    example = torch.tensor([[src.index("__bn__"), 5, 6, 7, EOS]])
    enc_path = os.path.join(work, "encoder.onnx")
    torch.onnx.export(EncoderGraph(enc, dec).eval(), (example,), enc_path,
                      input_names=["src"], output_names=["cross_k", "cross_v"],
                      dynamic_axes={"src": {1: "S"}, "cross_k": {3: "S"}, "cross_v": {3: "S"}},
                      opset_version=17, dynamo=False)
    ck, cv = EncoderGraph(enc, dec).eval()(example)
    cache = torch.zeros(LAYERS, 2, HEADS, 3, HEAD_DIM)
    dec_path = os.path.join(work, "decoder.onnx")
    torch.onnx.export(DecoderStep(dec).eval(), (torch.tensor([[7], [8]]), torch.tensor([3]), cache, cache, ck, cv),
                      dec_path,
                      input_names=["token", "pos", "self_k", "self_v", "cross_k", "cross_v"],
                      output_names=["logp", "new_k", "new_v"],
                      dynamic_axes={"token": {0: "B"}, "self_k": {1: "B", 3: "T"}, "self_v": {1: "B", 3: "T"},
                                    "cross_k": {3: "S"}, "cross_v": {3: "S"}, "logp": {0: "B"},
                                    "new_k": {1: "B", 3: "T1"}, "new_v": {1: "B", 3: "T1"}},
                      opset_version=17, dynamo=False)
    for name, path in (("encoder", enc_path), ("decoder", dec_path)):
        quantize_dynamic(path, os.path.join(out, f"indicxlit-{name}.int8.onnx"),
                         weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"])
    with open(os.path.join(out, "indicxlit-vocab.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.writelines(f"s\t{s}\n" for s in src)
        f.writelines(f"t\t{s}\n" for s in tgt)
    for name in sorted(os.listdir(out)):
        if name.startswith("indicxlit-"):
            path = os.path.join(out, name)
            print(f"{name}\t{os.path.getsize(path)}\t{sha256(path)}")


def check(out):
    """Beam search over the int8 graphs, the keyboard's (4 wide, fairseq's length normalisation)."""
    import onnxruntime as ort

    vocab = [line.rstrip("\n").split("\t", 1) for line in open(os.path.join(out, "indicxlit-vocab.txt"), encoding="utf-8")]
    src = [s for kind, s in vocab if kind == "s"]
    tgt = [s for kind, s in vocab if kind == "t"]
    enc = ort.InferenceSession(os.path.join(out, "indicxlit-encoder.int8.onnx"))
    dec = ort.InferenceSession(os.path.join(out, "indicxlit-decoder.int8.onnx"))

    def run(word, lang, block, beam=4):
        allowed = np.array([i == EOS or (i > 3 and not s.startswith("__") and all(block[0] <= ord(c) <= block[1] for c in s))
                            for i, s in enumerate(tgt)])
        ids = [src.index(f"__{lang}__")] + [src.index(c) for c in word] + [EOS]
        ck, cv = enc.run(None, {"src": np.array([ids], dtype=np.int64)})
        live, done = [([], 0.0)], []
        sk = sv = np.zeros((LAYERS, 1, HEADS, 0, HEAD_DIM), dtype=np.float32)
        for step in range(2 * len(word) + 6):
            last = np.array([[h[-1] if h else EOS] for h, _ in live], dtype=np.int64)
            logp, sk, sv = dec.run(None, {"token": last, "pos": np.array([step]), "self_k": sk, "self_v": sv,
                                          "cross_k": ck, "cross_v": cv})
            exp = sorted(((s + logp[b, j], b, j) for b, (h, s) in enumerate(live) for j in range(len(tgt))
                          if allowed[j] and not (j == EOS and step == 0)), reverse=True)[:2 * beam]
            nxt, origin = [], []
            for s, b, j in exp:
                if j == EOS:
                    if len(done) < beam:
                        done.append(("".join(tgt[t] for t in live[b][0]), s / (len(live[b][0]) + 1)))
                else:
                    nxt.append((live[b][0] + [j], s))
                    origin.append(b)
                if len(nxt) == beam:
                    break
            if len(done) >= beam or not nxt:
                break
            live, sk, sv = nxt, sk[:, origin], sv[:, origin]
        return [w for w, _ in sorted(done, key=lambda d: -d[1])]

    for word, lang, block in (("ami", "bn", (0x980, 0x9FF)), ("kemon", "bn", (0x980, 0x9FF)),
                              ("namaste", "hi", (0x900, 0x97F)), ("kitab", "ur", (0x600, 0x6FF)),
                              ("vanakkam", "ta", (0xB80, 0xBFF))):
        print(word, lang, run(word, lang, block))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--zip", help="the release zip, already downloaded")
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    if not a.check:
        export(fetch(a.zip), a.out)
    check(a.out)


if __name__ == "__main__":
    main()
