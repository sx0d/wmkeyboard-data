#!/usr/bin/env python3
"""Embeds WM Keyboard's emoji catalogue with EmbeddingGemma 2, once, so the
phone does not have to: every text the model reads costs a full encoder pass,
and some 1,700 emoji names would keep a phone busy for ten minutes.

Reads the files scripts/repack_embeddinggemma2.py wrote and the app's
`app/src/main/assets/emoji/catalog.tsv`, and writes `eg2-emoji-vectors.bin`:
`WMEV`, int32 version (1), int32 count, int32 width (768), then each emoji as
a uint16 byte length and its UTF-8, then every vector as float16, unit
length, in the same order. Little-endian throughout.

Each emoji is read as a stored text, the way the app reads its clips:
`title: none | text: <name>: <keywords>`, its English name and up to ten of
its English keywords. Skin-tone variants (rows with a parent) are left out;
the keyboard searches their base emoji.

    venv/bin/python scripts/embed_emoji_catalog.py --models models/embeddings \
        --catalog ../WMKeyboard/app/src/main/assets/emoji/catalog.tsv
"""

import argparse
import hashlib
import os
import struct

import numpy as np
import sentencepiece as spm
from ai_edge_litert.interpreter import Interpreter

PREFIX = "title: none | text: "
MAX_KEYWORDS = 10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", required=True)
    parser.add_argument("--catalog", required=True)
    args = parser.parse_args()
    folder = args.models
    sp = spm.SentencePieceProcessor(model_file=os.path.join(folder, "eg2-tokenizer.model"))
    raw = open(os.path.join(folder, "eg2-tokens.bin"), "rb").read()
    vocab, width, multiplier = struct.unpack_from("<iif", raw, 4)
    scales = np.frombuffer(raw, "<f4", vocab, 16)
    table = np.frombuffer(raw, np.uint8, offset=16 + 4 * vocab)

    def row(token):
        packed = table[token * width // 2:(token + 1) * width // 2]
        low = (packed & 15).astype(np.int8)
        high = (packed >> 4).astype(np.int8)
        out = np.empty(width, np.float32)
        out[0::2] = np.where(low > 7, low - 16, low)
        out[1::2] = np.where(high > 7, high - 16, high)
        return out * np.float32(scales[token]) * np.float32(multiplier)

    encoder = Interpreter(model_path=os.path.join(folder, "eg2-text-encoder.tflite"), num_threads=os.cpu_count())
    run = encoder.get_signature_runner("encoder_1x128")

    def embed(text):
        ids = [sp.bos_id()] + sp.encode(PREFIX + text)[:126] + [sp.eos_id()]
        x = np.zeros((1, 128, width), np.float32)
        mask = np.zeros((1, 128), np.float32)
        for k, token in enumerate(ids):
            x[0, k] = row(token)
        mask[0, :len(ids)] = 1
        v = run(embeddings=x, input_mask=mask)["encodings"][0]
        return v / np.linalg.norm(v)

    entries = []
    for line in open(args.catalog, encoding="utf-8"):
        if line.startswith("#") or not line.strip():
            continue
        cols = line.rstrip("\n").split("\t")
        emoji, keywords, parent, name = cols[0], cols[2], cols[4], cols[5]
        if parent:
            continue
        words = [k for k in keywords.split(",") if k and k != name][:MAX_KEYWORDS]
        entries.append((emoji, f"{name}: {', '.join(words)}" if words else name))

    vectors = []
    for k, (emoji, text) in enumerate(entries):
        vectors.append(embed(text).astype("<f2"))
        if k % 200 == 0:
            print(k, emoji, text, flush=True)
    out = os.path.join(folder, "eg2-emoji-vectors.bin")
    with open(out, "wb") as f:
        f.write(b"WMEV" + struct.pack("<iii", 1, len(entries), 768))
        for emoji, _ in entries:
            data = emoji.encode("utf-8")
            f.write(struct.pack("<H", len(data)) + data)
        for vector in vectors:
            f.write(vector.tobytes())
    data = open(out, "rb").read()
    print(f"eg2-emoji-vectors.bin\t{len(data)}\t{hashlib.sha256(data).hexdigest()}\t{len(entries)} emoji")


if __name__ == "__main__":
    main()
