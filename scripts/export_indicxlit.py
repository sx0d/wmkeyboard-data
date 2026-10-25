#!/usr/bin/env python3
"""Exports AI4Bharat's IndicXlit (English letters -> 21 Indic scripts) to the
ONNX files WM Keyboard's phonetic layouts run through ONNX Runtime, and with
`--direction indic-en` its mirror image (21 Indic scripts -> English letters),
which writes romanized dictation (Hinglish, Banglish...) for words the
keyboard's spelling maps do not know.

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

The other direction, `indicxlit-indic-en-v1.0.zip` on the same release, is
the same network with the vocabularies swapped: 806 source ids (the Indic
characters and the language tokens), 54 target ids (the 26 letters, two
unused `madeupword` fillers, the language tokens). Its tag is the source
language's (`encoder_langtok=src`), `__bn__ আ মি </s>`, from the same
lang_list.txt, so the token ids match. Same dims, so the keyboard's search
(native/indicxlit-jni) runs both. Its files are prefixed `indicxlit-latin-`:

    venv/bin/python scripts/export_indicxlit.py --direction indic-en --out models/indicxlit-latin
    venv/bin/python scripts/export_indicxlit.py --direction indic-en --out models/indicxlit-latin \\
        --check --eval aksharantar/ [--rules rules.tsv]

`--eval DIR` reads Aksharantar's `<lang>_test.json` files there (the
`ai4bharat/Aksharantar` dataset, one zip per language; the `<lang>` is the
keyboard's id) and prints top-1 and top-4 per language, a native word
counting as right when the model wrote any of the English spellings the test
set has for it. `--rules FILE` (`lang<TAB>native<TAB>romanized`, what the
keyboard's own romanizers write for each test word) adds their top-1 beside.
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

RELEASES = "https://github.com/AI4Bharat/IndicXlit/releases/download/v1.0/"
# Per direction: the release zip and the checkpoint inside it, both pinned;
# the dictionaries the source and target ids come from (every Indic language's
# dict.<lang>.txt is the same file); the prefix of the files written.
DIRECTIONS = {
    "en-indic": dict(
        zip="indicxlit-en-indic-v1.0.zip",
        zip_sha256="31dabc448c14dfac03015763d35cc4c895d8ee2b831fe23c64a5c5aea4f169ca",
        checkpoint_sha256="2f03fe0dfb2edc40f3c11f9293c4feda4fa8ed5366fec2981391606f224a09f7",
        source="corpus-bin/dict.en.txt", target="corpus-bin/dict.bn.txt", prefix="indicxlit-",
    ),
    "indic-en": dict(
        zip="indicxlit-indic-en-v1.0.zip",
        zip_sha256="420d23a13cfa953edfb47644bc805de06e97134f25e6d182303a29f55b5a3322",
        checkpoint_sha256="45ffecef88d990ae12f915939b91267a3fc8f69f26331baa59f7932005976f28",
        source="corpus-bin/dict.bn.txt", target="corpus-bin/dict.en.txt", prefix="indicxlit-latin-",
    ),
}
PARTS = ("encoder.int8.onnx", "decoder.int8.onnx", "vocab.txt")
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


def fetch(zip_path, direction):
    d = DIRECTIONS[direction]
    if zip_path is None:
        zip_path = os.path.join(tempfile.mkdtemp(), d["zip"])
        print("fetching", RELEASES + d["zip"])
        urllib.request.urlretrieve(RELEASES + d["zip"], zip_path)
    got = sha256(zip_path)
    if got != d["zip_sha256"]:
        raise SystemExit(f"{zip_path}: sha256 {got}, expected {d['zip_sha256']}")
    return zip_path


def export(zip_path, out, direction):
    from onnxruntime.quantization import QuantType, quantize_dynamic

    d = DIRECTIONS[direction]
    prefix = d["prefix"]
    os.makedirs(out, exist_ok=True)
    work = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_path) as z:
        src = load_dict(z.read(d["source"]).decode("utf-8").splitlines(True))
        tgt = load_dict(z.read(d["target"]).decode("utf-8").splitlines(True))
        ckpt = os.path.join(work, "indicxlit.pt")
        with open(ckpt, "wb") as f:
            f.write(z.read("transformer/indicxlit.pt"))
    if sha256(ckpt) != d["checkpoint_sha256"]:
        raise SystemExit("unexpected checkpoint inside the release zip")
    state = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
    skip = ("version", "_float_tensor")
    enc, dec = Encoder(len(src)), Decoder(len(tgt))
    for module, prefix_ in ((enc, "encoder."), (dec, "decoder.")):
        module.load_state_dict({k[len(prefix_):]: v for k, v in state.items()
                                if k.startswith(prefix_) and not any(s in k for s in skip)})
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
        quantize_dynamic(path, os.path.join(out, f"{prefix}{name}.int8.onnx"),
                         weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"])
    with open(os.path.join(out, f"{prefix}vocab.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.writelines(f"s\t{s}\n" for s in src)
        f.writelines(f"t\t{s}\n" for s in tgt)
    for part in PARTS:
        path = os.path.join(out, prefix + part)
        print(f"{prefix + part}\t{os.path.getsize(path)}\t{sha256(path)}")


# The Unicode block each language's readings are kept to, en-indic's way.
BLOCKS = {"bn": (0x980, 0x9FF), "hi": (0x900, 0x97F), "ur": (0x600, 0x6FF), "ta": (0xB80, 0xBFF)}


class Search:
    """Beam search over the int8 graphs, the keyboard's (4 wide, fairseq's length normalisation)."""

    def __init__(self, out, direction):
        import onnxruntime as ort

        prefix = DIRECTIONS[direction]["prefix"]
        vocab = [line.rstrip("\n").split("\t", 1)
                 for line in open(os.path.join(out, f"{prefix}vocab.txt"), encoding="utf-8")]
        self.src = [s for kind, s in vocab if kind == "s"]
        self.tgt = [s for kind, s in vocab if kind == "t"]
        self.src_ids = {s: i for i, s in enumerate(self.src)}
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.enc = ort.InferenceSession(os.path.join(out, f"{prefix}encoder.int8.onnx"), options)
        self.dec = ort.InferenceSession(os.path.join(out, f"{prefix}decoder.int8.onnx"), options)
        self.latin = direction == "indic-en"
        self.masks = {}

    def allowed(self, lang):
        if lang not in self.masks:
            if self.latin:
                # The 26 letters; not the madeupword fillers, which are letters too.
                ok = lambda s: len(s) == 1 and "a" <= s <= "z"
            else:
                lo, hi = BLOCKS[lang]
                ok = lambda s: not s.startswith("__") and all(lo <= ord(c) <= hi for c in s)
            self.masks[lang] = np.array([i == EOS or (i > 3 and ok(s)) for i, s in enumerate(self.tgt)])
        return self.masks[lang]

    def encode(self, word, lang):
        """The source ids, or None when a character is not one the model knows."""
        ids = [self.src_ids[f"__{lang}__"]]
        for c in word:
            if c not in self.src_ids or c.startswith("__"):
                return None
            ids.append(self.src_ids[c])
        return ids + [EOS]

    def run(self, word, lang, beam=4):
        ids = self.encode(word, lang)
        if ids is None:
            return None
        allowed = self.allowed(lang)
        tgt = self.tgt
        ck, cv = self.enc.run(None, {"src": np.array([ids], dtype=np.int64)})
        live, done = [([], 0.0)], []
        sk = sv = np.zeros((LAYERS, 1, HEADS, 0, HEAD_DIM), dtype=np.float32)
        for step in range(2 * len(word) + 6):
            last = np.array([[h[-1] if h else EOS] for h, _ in live], dtype=np.int64)
            logp, sk, sv = self.dec.run(None, {"token": last, "pos": np.array([step]), "self_k": sk, "self_v": sv,
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
        # Out of length: what is still alive counts, as if it ended here.
        for h, s in live:
            if len(done) >= beam:
                break
            done.append(("".join(tgt[t] for t in h), s / (len(h) + 1)))
        return [w for w, _ in sorted(done, key=lambda d: -d[1])]


def check(out, direction):
    search = Search(out, direction)
    if direction == "indic-en":
        words = (("আমি", "bn"), ("ভালোবাসি", "bn"), ("नमस्ते", "hi"), ("कल", "hi"), ("کتاب", "ur"),
                 ("வணக்கம்", "ta"), ("నమస్కారం", "te"))
    else:
        words = (("ami", "bn"), ("kemon", "bn"), ("namaste", "hi"), ("kitab", "ur"), ("vanakkam", "ta"))
    for word, lang in words:
        print(word, lang, search.run(word, lang))


def evaluate(out, direction, folder, rules_path, limit):
    """Top-1 and top-4 on Aksharantar's test words, native to English letters, beside the rules' top-1."""
    import json
    import time
    import unicodedata

    search = Search(out, direction)
    rules = {}
    if rules_path:
        for line in open(rules_path, encoding="utf-8"):
            lang, native, roman = line.rstrip("\n").split("\t")
            rules[(lang, native)] = roman
    print("lang\twords\tmodel top-1\tmodel top-4\trules top-1\tunreadable\tms/word")
    for name in sorted(os.listdir(folder)):
        if not name.endswith("_test.json"):
            continue
        lang = name[:-len("_test.json")]
        answers = {}
        for line in open(os.path.join(folder, name), encoding="utf-8"):
            if line.strip():
                item = json.loads(line)
                native = unicodedata.normalize("NFC", item["native word"].strip())
                answers.setdefault(native, set()).add(item["english word"].strip().lower())
        natives = sorted(answers)
        if limit and len(natives) > limit:
            natives = natives[::len(natives) // limit][:limit]
        top1 = top4 = rule1 = unreadable = 0
        started = time.perf_counter()
        for native in natives:
            right = answers[native]
            found = search.run(native, lang)
            if found is None:
                unreadable += 1
                found = [rules.get((lang, native), "")]
            top1 += bool(found) and found[0] in right
            top4 += any(w in right for w in found[:4])
            rule1 += rules.get((lang, native)) in right
        ms = 1000 * (time.perf_counter() - started) / max(1, len(natives))
        n = len(natives)
        pct = lambda k: f"{100 * k / n:.1f}"
        print(f"{lang}\t{n}\t{pct(top1)}\t{pct(top4)}\t{pct(rule1) if rules else '-'}\t{unreadable}\t{ms:.1f}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--direction", choices=sorted(DIRECTIONS), default="en-indic")
    p.add_argument("--zip", help="the release zip, already downloaded")
    p.add_argument("--check", action="store_true", help="only read the files in --out back")
    p.add_argument("--eval", metavar="DIR", help="Aksharantar <lang>_test.json files (indic-en)")
    p.add_argument("--rules", metavar="FILE", help="lang<TAB>native<TAB>romanized, the keyboard's rules")
    p.add_argument("--limit", type=int, default=0, help="at most this many test words a language")
    a = p.parse_args()
    if not a.check:
        export(fetch(a.zip, a.direction), a.out, a.direction)
    check(a.out, a.direction)
    if a.eval:
        evaluate(a.out, a.direction, a.eval, a.rules, a.limit)


if __name__ == "__main__":
    main()
