# Voikko Finnish dictionary

The dictionary WM Keyboard checks Finnish with, through
[Voikko](https://voikko.puimula.org/) (libvoikko, the library the app
downloads on demand as `libwmvoikko.so`). The app fetches the zip from
**Language › Finnish › Dictionaries › Voikko spell checking**, checks it
against the SHA-256 pinned in `VoikkoDictionaryCatalog.kt`, and unpacks it
into its own storage only if libvoikko loads it.

| File | Bytes | SHA-256 |
|---|---|---|
| `voikko-fi-2.6-mor-standard.zip` | 1,645,402 | `9a7b8ed3f401ac46e81714775743219175c54164a4eaa3b0286c68db4825b439` |
| `voikko-fi-2.6.tar.gz` | 709,151 | `5c147c1c92323941e5847b5c46d00b35ff6b62dc3cf348516bc0fbdc87340220` |
| `voikko-fi-2.6.tar.gz.asc` | 833 | upstream's signature of the tarball |

The zip holds `5/mor-standard/index.txt`, `mor.vfst` and `autocorr.vfst`:
voikko-fi's standard vocabulary in libvoikko's dictionary format 5 (VFST).
Upstream publishes voikko-fi as source only, so the zip is built from the
tarball beside it, which is the signed release from
<https://www.puimula.org/voikko-sources/voikko-fi/>, unchanged. The build is
`native/voikko-jni/build-dict.sh` in the app repository: voikko-fi's own
`make vvfst` with foma and libvoikko's `voikkovfstc`, a fixed build date and
C-locale sort, packed with fixed timestamps. Run twice, it gives the same zip
byte for byte.

## Licence

voikko-fi is licensed under the GNU General Public License, version 2 or
later (`COPYING`, the text in the tarball). Copyright 2006-2026 Hannu
Väisänen, Harri Pitkänen, Teemu Likonen and others. The corresponding source
of the zip is `voikko-fi-2.6.tar.gz` in this folder, with the build script
named above.
