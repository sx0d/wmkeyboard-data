# Feature libraries

WM Keyboard's GitHub and F-Droid builds leave the native libraries behind
optional features out of the app, and download each one the first time its
feature is used: offline voice typing (whisper.cpp, ExecuTorch, Whistle),
grammar checking (Harper) and the offline text reader (Tesseract).

The files are not in this repository's tree. They are assets of its
[`natives` release](https://github.com/wasi-master/wmkeyboard-data/releases/tag/natives),
named `<sha256>-<library>`, for example
`3f1c…a9-libharper_jni.so`.

- **Pinned.** Every app build carries the size and SHA-256 of each library it
  can download (`assets/native/pins.tsv`), and refuses a file that does not
  match. This release is only a place to fetch the bytes from.
- **Never replaced.** A new build of a library has a new hash, so it is a new
  file. Old files stay for the app versions that still pin them.
- **Built from source.** The app repository's release workflow uploads them:
  the GitHub build's from its committed `jniLibs` (built by `native/*/build.sh`),
  and F-Droid's from the build F-Droid makes from source, which is
  reproducible, so both are usually the same file.

`libwmvoikko.so` (Finnish spell checking) is libvoikko 4.3.3 with a JNI
bridge, offered under MPL 1.1, GPL 2+ or LGPL 2.1+. Its source is
`native/voikko-jni` in the app repository plus corevoikko at commit
`4b838726d6e11af235b5aba50dcc056bc17c8176`, which its `build.sh` fetches.

Libraries this project may not redistribute (ML Kit, LiteRT, LiteRT-LM and
Qualcomm's QNN runtime) are not here. The app reads those straight out of the
vendor's own package on Google's Maven repository or Maven Central.
