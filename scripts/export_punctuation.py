#!/usr/bin/env python3
"""Builds the punctuation model WM Keyboard's offline dictation downloads:
Horizon-Labs/punctuation-restoration-small (mmBERT-small token
classification, Apache-2.0, 88 languages), requantised for a phone and its
tokenizer written as a SentencePiece model.

    python3 -I scripts/export_punctuation.py models/punctuation

Writes three files, uploaded with
`scripts/upload_embedding_models.sh models/punctuation punct-`:

- punct-mmbert-small.onnx       the graph, weights outside it
- punct-mmbert-small.onnx.data  the weights: the token table in int8 (one
                                scale per row), every MatMul as 8-bit
                                weight-only MatMulNBits (blocks of 32,
                                symmetric, int8 compute); ONNX Runtime maps
                                this file instead of copying it
- punct-mmbert-small.model      the tokenizer: tokenizer.json's 256,000
                                pieces as a SentencePiece BPE ModelProto,
                                each piece scored by the rank of the first
                                merge that makes it

Then checks what it wrote: the tokenizer, run the way the app runs it (BPE by
piece score, one word at a time), against Hugging Face tokenizers on the
sample texts, and the quantised model's labels against the fp32 model's.

Needs onnx, onnx_ir, onnxruntime, sentencepiece, tokenizers and numpy. Built
with onnx 1.23.2, onnx_ir 1.1.0, onnxruntime 1.31.0 (checked on 1.30.0, the
app's), sentencepiece 0.2.2, tokenizers 0.23.3.
"""
import hashlib
import json
import os
import sys
import tempfile
import urllib.request

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.matmul_nbits_quantizer import DefaultWeightOnlyQuantConfig, MatMulNBitsQuantizer
from onnxruntime.quantization.shape_inference import quant_pre_process
from sentencepiece import sentencepiece_model_pb2 as spm_pb
from tokenizers import Tokenizer

REPO = "Horizon-Labs/punctuation-restoration-small"
REVISION = "ee31fdfc5304f876ea3b0dd72d30eda7e940b2b1"  # v1.1, 2026-10-09
SOURCES = {
    "onnx/model.onnx": (563_106_940, "751b4ba021f23d18d9cf7ac00938ce497a5ae58baa08b4fe5802f86e5d2e8940"),
    "tokenizer.json": (34_363_286, "f34b44a6dc8a0eef651bfa10e8325057530152606a69d9c3b6ae02efb1558f0a"),
    "config.json": (2_110, "f457c77954f0720ff2c0624b92714fb551bd19c42fba0625ee7af1740df49e05"),
}
NAME = "punct-mmbert-small"
BLOCK = 32

# Dictation as an engine writes it: lower case, no marks. A few languages and
# scripts, questions among them; the check is that nothing here differs.
SAMPLES = [
    "hello how are you today i hope everything is fine",
    "can you send me the file tomorrow morning i need it before the meeting",
    "hola cómo estás espero que todo vaya bien en el trabajo nos vemos mañana",
    "ich habe heute keine zeit aber morgen können wir uns treffen",
    "আমি আজ বাজারে যাব তুমি কি আমার সাথে আসবে",
    "मैं कल दिल्ली जा रहा हूँ क्या तुम भी चलोगे",
    "مرحبا كيف حالك هل انتهيت من العمل اليوم",
    "今天天气很好我们去公园散步吧你觉得怎么样",
    "привет как дела я сегодня работаю дома",
    "je pense que nous devrions partir tôt demain matin qu'en penses-tu",
    "お元気ですか明日は雨が降るそうです",
    "the meeting is at three pm on friday the twelfth so dont be late",
]


def fetch(path, dest):
    url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{path}"
    out = os.path.join(dest, os.path.basename(path))
    print("fetch", url)
    urllib.request.urlretrieve(url, out)
    size, sha = SOURCES[path]
    if size is not None and os.path.getsize(out) != size:
        sys.exit(f"{path}: {os.path.getsize(out)} bytes, expected {size}")
    if sha is not None and digest(out) != sha:
        sys.exit(f"{path}: sha256 {digest(out)}, expected {sha}")
    return out


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def quantise(fp32, work, out_dir):
    pre = os.path.join(work, "pre.onnx")
    quant_pre_process(fp32, pre, skip_symbolic_shape=True)
    # The token table: 256,000 x 384, int8 with a scale per row. Dequantised
    # after the Gather, so only the rows a phrase uses become float.
    table = os.path.join(work, "table8.onnx")
    quantize_dynamic(pre, table, weight_type=QuantType.QInt8, op_types_to_quantize=["Gather"])
    # Every MatMul weight-only in 8 bits: activations stay float, which is
    # what keeps the labels the fp32 model's (dynamic int8 MatMuls cost
    # 0.02-0.03 F1 on commas). accuracy_level 4 computes in int8 blocks.
    model = onnx.load(table)
    config = DefaultWeightOnlyQuantConfig(block_size=BLOCK, is_symmetric=True, bits=8, accuracy_level=4)
    q = MatMulNBitsQuantizer(model, block_size=BLOCK, is_symmetric=True, accuracy_level=4, algo_config=config)
    q.process()
    whole = os.path.join(work, "n8.onnx")
    q.model.save_model_to_file(whole, use_external_data_format=False)
    graph = os.path.join(out_dir, NAME + ".onnx")
    onnx.save_model(onnx.load(whole), graph, save_as_external_data=True, all_tensors_to_one_file=True,
                    location=NAME + ".onnx.data", size_threshold=4096)
    return graph


def write_tokenizer(tokenizer_json, out_dir):
    t = json.load(open(tokenizer_json, encoding="utf-8"))
    model = t["model"]
    assert model["type"] == "BPE" and model["byte_fallback"], "expected a byte-fallback BPE"
    assert t["pre_tokenizer"]["type"] == "Metaspace" and t["pre_tokenizer"]["split"], "expected Metaspace, split"
    vocab = model["vocab"]
    rank = {}
    for r, m in enumerate(model["merges"]):
        a, b = m if isinstance(m, list) else m.split(" ")
        rank.setdefault(a + b, r)
    added = {a["id"]: a for a in t["added_tokens"]}
    pieces = [None] * len(vocab)
    for piece, i in vocab.items():
        pieces[i] = piece
    proto = spm_pb.ModelProto()
    proto.trainer_spec.model_type = spm_pb.TrainerSpec.BPE
    proto.trainer_spec.vocab_size = len(pieces)
    proto.trainer_spec.byte_fallback = True
    proto.trainer_spec.unk_id, proto.trainer_spec.bos_id, proto.trainer_spec.eos_id, proto.trainer_spec.pad_id = 3, 2, 1, 0
    proto.normalizer_spec.name = "identity"
    proto.normalizer_spec.add_dummy_prefix = False
    proto.normalizer_spec.remove_extra_whitespaces = False
    T = spm_pb.ModelProto.SentencePiece
    for i, piece in enumerate(pieces):
        p = proto.pieces.add()
        p.piece = piece
        a = added.get(i)
        if a is not None:
            p.type = T.UNKNOWN if piece == model["unk_token"] else T.CONTROL if a["special"] else T.USER_DEFINED
            p.score = 0.0
        elif piece.startswith("<0x") and len(piece) == 6 and piece.endswith(">"):
            p.type = T.BYTE
            p.score = 0.0
        elif piece in rank:
            p.type = T.NORMAL
            p.score = float(-rank[piece])
        elif len(piece) == 1:
            p.type = T.NORMAL  # a character: never made by a merge, only matched
            p.score = 0.0
        else:
            # A piece no merge makes: Hugging Face's BPE never produces it,
            # so a score-based BPE must not either.
            p.type = T.CONTROL
            p.score = 0.0
    path = os.path.join(out_dir, NAME + ".model")
    with open(path, "wb") as f:
        f.write(proto.SerializeToString())
    return path


def app_encode(proto_path):
    """The app's tokenizer (GemmaTokenizer: BPE by piece score, leftmost on a
    tie, bytes for what no piece spells), one word at a time as Metaspace
    with `split` cuts the text."""
    proto = spm_pb.ModelProto()
    proto.ParseFromString(open(proto_path, "rb").read())
    T = spm_pb.ModelProto.SentencePiece
    ids, score, mergeable = {}, {}, set()
    for i, p in enumerate(proto.pieces):
        ids.setdefault(p.piece, i)
        score[i] = p.score
        if p.type in (T.NORMAL, T.USER_DEFINED, T.UNUSED):
            mergeable.add(i)

    def word(w):
        syms = list(w)
        while True:
            best = None
            for k in range(len(syms) - 1):
                i = ids.get(syms[k] + syms[k + 1])
                if i is None or i not in mergeable:
                    continue
                if best is None or score[i] > best[0]:
                    best = (score[i], k)
            if best is None:
                break
            k = best[1]
            syms[k:k + 2] = [syms[k] + syms[k + 1]]
        out = []
        for s in syms:
            if s in ids:
                out.append(ids[s])
            else:
                out += [ids["<0x%02X>" % b] for b in s.encode()]
        return out

    return lambda text: [i for w in text.split(" ") for i in word("▁" + w)]


def labels(session, ids):
    x = np.array([[2] + ids + [1]], dtype=np.int64)
    return session.run(None, {"input_ids": x, "attention_mask": np.ones_like(x)})[0][0].argmax(-1).tolist()


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "models/punctuation"
    os.makedirs(out_dir, exist_ok=True)
    with tempfile.TemporaryDirectory() as work:
        fp32 = fetch("onnx/model.onnx", work)
        tok_json = fetch("tokenizer.json", work)
        config = json.load(open(fetch("config.json", work)))
        assert config["id2label"] == {"0": "0", "1": ".", "2": ",", "3": "?", "4": "-", "5": ":"}, config["id2label"]
        graph = quantise(fp32, work, out_dir)
        spm = write_tokenizer(tok_json, out_dir)

        hf = Tokenizer.from_file(tok_json)
        encode = app_encode(spm)
        reference = ort.InferenceSession(fp32, providers=["CPUExecutionProvider"])
        shipped = ort.InferenceSession(graph, providers=["CPUExecutionProvider"])
        same = total = 0
        for text in SAMPLES:
            ids = hf.encode(text, add_special_tokens=False).ids
            assert encode(text) == ids, f"tokenizer differs on {text!r}"
            a, b = labels(reference, ids), labels(shipped, ids)
            same += sum(x == y for x, y in zip(a, b))
            total += len(a)
        print(f"tokenizer matches on {len(SAMPLES)} texts; labels {same}/{total} as fp32")
    for name in sorted(os.listdir(out_dir)):
        if name.startswith(NAME):
            path = os.path.join(out_dir, name)
            print(f"{name}  {os.path.getsize(path):,} bytes  sha256 {digest(path)}")


if __name__ == "__main__":
    main()
