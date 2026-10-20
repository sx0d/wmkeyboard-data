#!/usr/bin/env python3
"""Repacks Google's EmbeddingGemma 2 LiteRT-LM bundles into the plain files
WM Keyboard's meaning-based search loads with the classic LiteRT interpreter.

The bundles on Hugging Face (litert-community, Apache-2.0) are `.litertlm`
containers made for the LiteRT-LM runtime: a SentencePiece model, a few
TFLite graphs, and the graphs' weights in separate sections the graphs point
into as an "external buffer group". The classic interpreter cannot follow
such a pointer, so this script:

* copies the tokenizer out unchanged;
* turns the token embedder (one int4 lookup table and a multiply) into a flat
  table the app reads itself, with no interpreter: a 16-byte header
  (`WMTE`, vocabulary size, width, multiplier), one float32 scale per row,
  then the rows, two signed int4 values per byte, low nibble first, exactly
  as the bundle stores them;
* gives the text encoder, the vision encoder and the vision adapter ordinary
  inline buffers for their weights (one buffer per distinct weight, since
  the six encoder signatures share theirs);
* runs the end-of-image graph, which takes no input, once and keeps its
  512 floats.

Nothing is retrained or requantised: every weight is the bundle's own bytes.
The outputs are deterministic for a given flatbuffers version, so the sizes
and digests the app pins can be checked by running this again.

    python3 -m venv venv && venv/bin/pip install ai-edge-litert==2.3.0 \
        flatbuffers==25.12.19 numpy
    venv/bin/python scripts/repack_embeddinggemma2.py --out models/embeddings

The files are too large for this repository's tree; they are assets of its
`models` release, named `<sha256>-<name>` (see models/README.md), and
scripts/upload_embedding_models.sh puts them there.
"""

import argparse
import hashlib
import os
import struct
import urllib.request

import flatbuffers
import numpy as np
from ai_edge_litert import schema_py_generated as schema
from ai_edge_litert.interpreter import Interpreter

TEXT_REPO = "litert-community/embeddinggemma-2-text-270m-litert-lm"
TEXT_REV = "9be6e8b90982095dc05c2bd162e4b954ee4dbac7"
TEXT_FILE = "embeddinggemma-2-text-270m.litertlm"
TEXT_SHA = "2d079ee2f6f066b1f368e8d7c819f55214eaef1d0513b312321901f30ab286fb"

VISION_REPO = "litert-community/embeddinggemma-2-text-vision-440m-litert-lm"
VISION_REV = "e301f74d5551b0c2641bd5cb4652a76239d5c5f8"
VISION_FILE = "embeddinggemma-2-text-vision-440m.litertlm"
VISION_SHA = "92dcbea108899e5d6e30d919b0744f90d9967e80c67a4ab5503ac16d54f62eb0"

SECTION_TOKENIZER = 4
SECTION_TFLITE = 3
SECTION_WEIGHTS = 7


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(repo, rev, name, sha, cache):
    path = os.path.join(cache, name)
    if not os.path.exists(path) or sha256(path) != sha:
        url = f"https://huggingface.co/{repo}/resolve/{rev}/{name}"
        print("fetching", url)
        urllib.request.urlretrieve(url, path)
    if sha256(path) != sha:
        raise SystemExit(f"{name}: digest mismatch")
    return path


# --- The .litertlm container -------------------------------------------------
# "LITERTLM", three uint32 version numbers, padding, the uint64 end of the
# header, then a flatbuffer LiteRTLMMetaData at byte 32 whose second field
# lists the sections: key/value items, begin and end offsets, a data type.

def _u16(b, o): return struct.unpack_from("<H", b, o)[0]
def _i32(b, o): return struct.unpack_from("<i", b, o)[0]
def _u32(b, o): return struct.unpack_from("<I", b, o)[0]
def _u64(b, o): return struct.unpack_from("<Q", b, o)[0]


def _field(b, table, index):
    vtable = table - _i32(b, table)
    if 4 + 2 * index >= _u16(b, vtable):
        return None
    offset = _u16(b, vtable + 4 + 2 * index)
    return table + offset if offset else None


def _deref(b, p): return p + _u32(b, p)


def _vector(b, p):
    v = _deref(b, p)
    return [v + 4 + 4 * k for k in range(_u32(b, v))]


def _string(b, p):
    s = _deref(b, p)
    return b[s + 4:s + 4 + _u32(b, s)].decode()


def sections(path):
    with open(path, "rb") as f:
        head = f.read(1 << 20)
    if head[:8] != b"LITERTLM":
        raise SystemExit(f"{path}: not a LiteRT-LM bundle")
    root = 32 + _u32(head, 32)
    section_metadata = _deref(head, _field(head, root, 1))
    found = []
    for pointer in _vector(head, _field(head, section_metadata, 0)):
        obj = _deref(head, pointer)
        items = {}
        items_field = _field(head, obj, 0)
        for item_pointer in _vector(head, items_field) if items_field else []:
            item = _deref(head, item_pointer)
            key = _string(head, _field(head, item, 0))
            value = _deref(head, _field(head, item, 2))
            string_field = _field(head, value, 0)
            if string_field:
                try:
                    items[key] = _string(head, string_field)
                except UnicodeDecodeError:
                    pass
        found.append({
            "type": head[_field(head, obj, 3)],
            "begin": _u64(head, _field(head, obj, 1)),
            "end": _u64(head, _field(head, obj, 2)),
            "model": items.get("model_type"),
        })
    return found


def read_section(path, section):
    with open(path, "rb") as f:
        f.seek(section["begin"])
        return f.read(section["end"] - section["begin"])


def section_of(all_sections, kind, model=None):
    hits = [s for s in all_sections if s["type"] == kind and s["model"] == model]
    if len(hits) != 1:
        raise SystemExit(f"expected one section {kind} {model}, found {len(hits)}")
    return hits[0]


# --- Rewrites ------------------------------------------------------------------

def inline_weights(graph_bytes, weights_bytes):
    model = schema.ModelT.InitFromPackedBuf(graph_bytes, 0)
    external = {e.id: e for e in (model.externalBuffers or [])}
    made = {}
    for subgraph in model.subgraphs:
        for tensor in subgraph.tensors:
            ext_id = getattr(tensor, "externalBuffer", 0)
            if not ext_id:
                continue
            ext = external[ext_id]
            if ext.packing not in (b"unpacked", None):
                raise SystemExit(f"unsupported packing {ext.packing}")
            key = (ext.group, ext.offset, ext.length)
            if key not in made:
                buffer = schema.BufferT()
                buffer.data = np.frombuffer(weights_bytes, dtype=np.uint8, count=ext.length, offset=ext.offset)
                model.buffers.append(buffer)
                made[key] = len(model.buffers) - 1
            tensor.buffer = made[key]
            tensor.externalBuffer = 0
    model.externalBuffers = None
    model.externalBufferGroups = None
    builder = flatbuffers.Builder(len(weights_bytes) + (64 << 20))
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    return bytes(builder.Output())


def token_table(graph_bytes, weights_bytes):
    model = schema.ModelT.InitFromPackedBuf(graph_bytes, 0)
    graph = model.subgraphs[0]
    table = next(t for t in graph.tensors if t.type == schema.TensorType.INT4)
    vocab, width = (int(x) for x in table.shape)
    scales = np.asarray(table.quantization.scale, dtype="<f4")
    if any(table.quantization.zeroPoint):
        raise SystemExit("the token table is not symmetric")
    # The one scalar float constant: the multiply after the lookup.
    multiplier = None
    for tensor in graph.tensors:
        data = model.buffers[tensor.buffer].data
        if tensor.type == schema.TensorType.FLOAT32 and data is not None and len(data) == 4:
            multiplier = float(np.frombuffer(bytes(data), dtype="<f4")[0])
    if multiplier is None or len(weights_bytes) != vocab * width // 2:
        raise SystemExit("unexpected embedder graph")
    header = b"WMTE" + struct.pack("<iif", vocab, width, multiplier)
    return header + scales.tobytes() + weights_bytes


def end_of_image(graph_path):
    runner = Interpreter(model_path=graph_path).get_signature_runner("eoi")
    return np.asarray(runner()["eoi_embedding"], dtype="<f4").reshape(-1).tobytes()


def write(out, name, data, manifest):
    path = os.path.join(out, name)
    with open(path, "wb") as f:
        f.write(data)
    manifest.append((name, len(data), hashlib.sha256(data).hexdigest()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache", default=".cache-embeddinggemma2")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.cache, exist_ok=True)
    manifest = []

    text = fetch(TEXT_REPO, TEXT_REV, TEXT_FILE, TEXT_SHA, args.cache)
    found = sections(text)
    write(args.out, "eg2-tokenizer.model", read_section(text, section_of(found, SECTION_TOKENIZER)), manifest)
    write(args.out, "eg2-tokens.bin", token_table(
        read_section(text, section_of(found, SECTION_TFLITE, "tf_lite_embedder")),
        read_section(text, section_of(found, SECTION_WEIGHTS, "tf_lite_embedder")),
    ), manifest)
    write(args.out, "eg2-text-encoder.tflite", inline_weights(
        read_section(text, section_of(found, SECTION_TFLITE, "tf_lite_text_encoder")),
        read_section(text, section_of(found, SECTION_WEIGHTS, "tf_lite_text_encoder")),
    ), manifest)

    vision = fetch(VISION_REPO, VISION_REV, VISION_FILE, VISION_SHA, args.cache)
    found = sections(vision)
    write(args.out, "eg2-vision-encoder.tflite", inline_weights(
        read_section(vision, section_of(found, SECTION_TFLITE, "tf_lite_vision_encoder")),
        read_section(vision, section_of(found, SECTION_WEIGHTS, "tf_lite_vision_encoder")),
    ), manifest)
    write(args.out, "eg2-vision-adapter.tflite", inline_weights(
        read_section(vision, section_of(found, SECTION_TFLITE, "tf_lite_vision_adapter")),
        read_section(vision, section_of(found, SECTION_WEIGHTS, "tf_lite_vision_adapter")),
    ), manifest)
    eoi_path = os.path.join(args.cache, "end_of_vision.tflite")
    with open(eoi_path, "wb") as f:
        f.write(read_section(vision, section_of(found, SECTION_TFLITE, "tf_lite_end_of_vision")))
    write(args.out, "eg2-image-end.bin", end_of_image(eoi_path), manifest)

    for name, size, digest in manifest:
        print(f"{name}\t{size}\t{digest}")


if __name__ == "__main__":
    main()
