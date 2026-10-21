# Fonts

Fonts WM Keyboard fetches for a layout whose letters no phone font carries, or
whose phone font may not draw them right. The app downloads one the first time
it is needed, installs it into its font library under the name below, and draws
the keys with it.

| File | Name | For | Licence |
|---|---|---|---|
| [`tlh/klingon-piqad-hasta.ttf`](tlh/klingon-piqad-hasta.ttf) | Klingon pIqaD HaSta 2.003 | The Klingon pIqaD layout. pIqaD is encoded in the Private Use Area (U+F8D0–U+F8FF) per the [ConScript Unicode Registry](https://www.evertype.com/standards/csur/klingon.html) | [SIL OFL 1.1](tlh/OFL.txt), © 2005 Mike Neff (qa'vaj), © 2009 Michael Everson. Unmodified, from [evertype.com/fonts/tlh](https://www.evertype.com/fonts/tlh/) |
| [`my/noto-sans-myanmar-regular.ttf`](my/noto-sans-myanmar-regular.ttf) | Noto Sans Myanmar 2.107 | Burmese keys with **Unicode key font** on (Burmese's Zawgyi settings). A phone set up for Zawgyi replaces its Myanmar font with a Zawgyi one, which draws Unicode wrong; this face is the keyboard's own | [SIL OFL 1.1](my/OFL.txt), © 2022 The Noto Project Authors. Unmodified, the hinted Regular from [notofonts/myanmar](https://github.com/notofonts/myanmar) v2.107 |

SHA-256 of `tlh/klingon-piqad-hasta.ttf`:
`b71d452bdebf7f92e338d594a0a05f143e879011f8ced7191f5ce516e5ca1e09` (99,920 bytes).
The app checks it before installing, so a mirror cannot serve a different file.

SHA-256 of `my/noto-sans-myanmar-regular.ttf`:
`fafce4db400bc0b214907ccdbfb0ad2f18a57bfefd08c8a571830b84088cf2fc` (195,088 bytes).
