# models

Small neural networks WM Keyboard downloads on demand. The app pins each file
by SHA-256 and refuses one that does not match, so a file here is never
replaced in place: a new model gets a new name.

## `cutout/u2netp.tflite`

The sticker editor's **Remove the background**, on a phone with no Google Play
services to supply ML Kit's subject segmentation.

| | |
|---|---|
| Network | U²-Net-P, the small variant of U²-Net (Qin et al., *U²-Net: Going Deeper with Nested U-Structure for Salient Object Detection*, Pattern Recognition 2020) |
| Weights | the authors' `u2netp` checkpoint, as the `u2netp.onnx` that [rembg](https://github.com/danielgatis/rembg) publishes (`sha256 309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8`) |
| Format | LiteRT / TensorFlow Lite flatbuffer, float32, converted from that ONNX with onnx2tf |
| Size | 4,575,464 bytes |
| SHA-256 | `558136589a47e97f53944ac3f65ba56f4efb5f6bae8e92b573e7c463d20fdb58` |
| Input | `input.1`, `[1, 320, 320, 3]` float32, RGB scaled by the picture's brightest value, then ImageNet mean/std |
| Output | signature `serving_default`, seven `[1, 320, 320, 1]` saliency maps; `1959` is the fused one the app reads |
| License | [Apache-2.0](https://github.com/xuebinqin/U-2-Net/blob/master/LICENSE), © Xuebin Qin et al. |

The conversion was taken from
[abhimanyu666/u2nettflite](https://huggingface.co/abhimanyu666/u2nettflite)
(Apache-2.0) and checked before it was mirrored here: run side by side with the
rembg ONNX on the same input, the largest difference on any of the seven
outputs is 3e-5.

## Meaning-based search: EmbeddingGemma 2

The keyboard's **Search by meaning**: clipboard, notes, snippets and emoji
found by what they mean, in any language, and pictures (own stickers, GIFs
saved to a pack, photos) found by what they show. One model family puts text
and pictures in the same space, so a typed query is compared with both.

These files are not in the tree. They are assets of the
[`models` release](https://github.com/wasi-master/wmkeyboard-data/releases/tag/models),
named `<sha256>-<name>` like the `natives` release beside it: pinned by the
app, never replaced, a new file gets a new digest.
`scripts/repack_embeddinggemma2.py` builds them and
`scripts/upload_embedding_models.sh` uploads them.

| | |
|---|---|
| Model | EmbeddingGemma 2 (Google, 2026): Text 270M, and the vision tower of Text Vision 440M |
| Source | `litert-community/embeddinggemma-2-text-270m-litert-lm` at `9be6e8b9`, `embeddinggemma-2-text-270m.litertlm` (`sha256 2d079ee2…0ab286fb`); `litert-community/embeddinggemma-2-text-vision-440m-litert-lm` at `e301f74d`, `embeddinggemma-2-text-vision-440m.litertlm` (`sha256 92dcbea1…4f62eb0`) |
| Change | Repackaged only: the LiteRT-LM bundle's external weight sections are written back into ordinary TFLite buffers, only the signatures the app runs are kept (shortest first, since the classic interpreter prepares subgraph 0 on load), and the int4 token table becomes a flat file the app reads without an interpreter. No weight is requantised; the kept signatures' outputs are bit-identical to the bundle's. |
| License | [Apache-2.0](https://huggingface.co/google/embeddinggemma-2), © Google |

Text (151,886,397 bytes in all):

| File | Bytes | SHA-256 | What it is |
|---|---:|---|---|
| `eg2-tokenizer.model` | 4,689,013 | `e594c8a90eb08d8bda498ff4747977dc827ae0c3c56b5c0d41a605a22d02ef03` | SentencePiece BPE model, byte fallback, 262,144 pieces (identical to `google/embeddinggemma-2`'s `tokenizer.model`) |
| `eg2-tokens.bin` | 68,157,456 | `3bc62fc4be4c2c9bc5c9902e8b41a19e698a6b218314390fa2411f4242ae5160` | `WMTE`, int32 vocabulary, int32 width (512), float32 multiplier (√512), one float32 scale per row, then rows of signed int4 pairs, low nibble first |
| `eg2-text-encoder.tflite` | 79,039,928 | `305850a79ab1a38a8d3862f5ac7d9500a386ac969c4f9783da6d81237ead038b` | signatures `encoder_1x{128,256,512}` (the bundle's 1024, 2048 and 8192 are left out): `embeddings` `[1, n, 512]`, `input_mask` `[1, n]` → `encodings` `[1, 768]` (L2-normalise it) |

Pictures (220,552,716 bytes more):

| File | Bytes | SHA-256 | What it is |
|---|---:|---|---|
| `eg2-vision-encoder.tflite` | 218,970,136 | `b7f069019f953e00910ad4b4b5fb029180ea3faa4fe0fa30d6fc2d37198d908b` | signature `vision_70` (the bundle's `vision_140` is left out): `images` `[1, 630, 768]` (16×16 RGB patches, 0..1), `positions_xy` `[1, 630, 2]` (-1 for padding) → `features` `[1, 70, 768]`, `mask` `[1, 70]` |
| `eg2-vision-adapter.tflite` | 1,580,532 | `fe5d26b805e24163ac495ef269490c6572255559bd17881cd0f66ce2758dfe3b` | signature `vision_adapter_70`: `soft_tokens` → `mm_embedding` `[1, 70, 512]` |
| `eg2-image-end.bin` | 2,048 | `e6599ba28ac69c35da660a0f938b589ffc94d203bcbd6797ec46df71d7c5990a` | the end-of-image embedding, 512 float32 |

A text is `<bos>`, its tokens, `<eos>`; a picture is `<bos>`, `<|image>`,
its valid soft tokens, the end-of-image embedding, `<eos>`; both go through
the text encoder. Queries take the prefix `task: search result | query: `,
stored texts `title: none | text: `. Checked through LiteRT 2.3.0 on these
exact files: English, Bengali and French texts rank as expected against each
other, and text queries pick the matching COCO photograph (cats on a sofa,
a skier, a bedroom, tennis, baseball, including a Bengali query).
