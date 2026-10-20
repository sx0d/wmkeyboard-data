# zinnia handwriting models

Models for [zinnia](https://github.com/taku910/zinnia), the offline handwriting
recogniser WM Keyboard uses for Chinese and Japanese where Google ML Kit's is
not in the build (F-Droid's) or not picked. The keyboard downloads each one as
a CJK pack the first time its language is written in, and checks it against
the SHA-256 pinned in `ZinniaData.kt`.

They are the Tegaki project's 0.3 releases, unchanged: these bytes are the
`.model` files in Tegaki's own release archives, renamed with the release
number so a later model can go out under a new name.

| File | Characters | Bytes | SHA-256 | Taken from |
|---|---|---|---|---|
| `handwriting-ja-0.3.model` | 6,420 | 26,068,668 | `d58618576d12c1ea38992606f30d19158a0a3beda3ac23bacccf1e48e651f2b2` | Debian `tegaki-zinnia-japanese_0.3.orig.tar.gz` |
| `handwriting-zh_CN-0.3.model` | 6,763 | 26,834,816 | `e16153d1ff267cd479aea260d6f71a3edda8b4ba06db2d121513adda65a4449e` | Debian `tegaki-zinnia-simplified-chinese_0.3.orig.tar.gz` |
| `handwriting-zh_TW-0.3.model` | 11,853 | 52,210,768 | `cd47f16b64e7ecaa4c2813d0f98231dfea7b411f48be1ead5c9b2eeb19330b11` | `tegaki-zinnia-traditional-chinese-0.3.zip` from pkgsrc distfiles (SHA-1 `a1683a302c607e4e3207c1cc45234f93126f02df`, the same in OpenMandriva's packaging) |

tegaki.org, where the archives were first published, no longer serves them.
Each archive also carries the training data the model was built from (Tomoe's
stroke data as Tegaki XML) and builds again with `tegaki-build`; the
Traditional Chinese one was first bootstrapped from character components with
`tegaki-bootstrap`.

## Licence

LGPL 2.1, the licence of the Tomoe handwriting dictionary the models are
trained on and of Tegaki's model releases; the text is in `COPYING`, as it is
in each archive. zinnia itself is BSD-3-Clause and ships in the keyboard, not
here.
