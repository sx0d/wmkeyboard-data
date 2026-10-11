#!/usr/bin/env python3
"""Rebuild data/bs/bs_full.txt.gz: OpenSubtitles counts filtered/extended by the
ebukva/bs_BA Hunspell dictionary (LGPL-3.0, https://github.com/ebukva/bs_BA).

Usage: import_bs_hunspell.py bs_BA.aff bs_BA.dic old_bs_full.txt new_bs_full.txt.gz

The .aff/.dic pair is expanded with a small unmunch (prefix/suffix flags only,
which is all this dictionary uses). Output rules:
  * corpus word that Hunspell accepts (case-insensitive): kept with its count
  * corpus word Hunspell does not know: kept only at count >= 100
    (informal / loan words), so typos and Serbian/Croatian-only forms drop
  * Hunspell form the corpus never saw: added at count 2 (the downloader stops
    reading at count < 2, so these are read after every counted word)
"""
import sys, gzip
AFF, DIC, OLD, NEW = sys.argv[1:5]
import re, collections

aff = open(AFF, encoding='utf-8').read().splitlines()
rules = collections.defaultdict(lambda: {'kind': None, 'cross': False, 'rows': []})
i = 0
while i < len(aff):
    p = aff[i].split('#')[0].split()
    if len(p) >= 4 and p[0] in ('SFX', 'PFX') and len(p[1]) == 1 and p[3].isdigit():
        r = rules[p[1]]; r['kind'] = p[0]; r['cross'] = p[2] == 'Y'
        for j in range(int(p[3])):
            q = aff[i + 1 + j].split('#')[0].split()
            strip = '' if q[2] == '0' else q[2]
            add = q[3].split('/')[0]; add = '' if add == '0' else add
            cond = q[4] if len(q) > 4 else '.'
            r['rows'].append((strip, add, re.compile(('.*' if q[0] == 'SFX' else '') + cond + ('' if q[0] == 'SFX' else '.*')) if False else cond))
        i += int(p[3])
    i += 1
def match(kind, cond, w):
    rx = (cond + '$') if kind == 'SFX' else ('^' + cond)
    return re.search(rx, w) is not None
def apply(kind, row, w):
    strip, add, cond = row
    if not match(kind, cond, w): return None
    if kind == 'SFX':
        if strip and not w.endswith(strip): return None
        return (w[:len(w) - len(strip)] if strip else w) + add
    if strip and not w.startswith(strip): return None
    return add + (w[len(strip):] if strip else w)
out = {}; keep = set()
for line in open(DIC, encoding='utf-8').read().splitlines()[1:]:
    if not line or line.startswith('#') or line[0] in ' \t': continue
    w, _, fl = line.partition('/'); w = w.split('\t')[0].strip(); fl = fl.split()[0] if fl else ''
    if not w: continue
    forms = {w}; sfx = []; pfx = []
    for f in fl:
        r = rules.get(f)
        if not r: continue
        for row in r['rows']:
            n = apply(r['kind'], row, w)
            if n:
                forms.add(n); (sfx if r['kind'] == 'SFX' else pfx).append((n, r['cross']))
    for pn, pc in pfx:
        if not pc: continue
    for f in fl:
        r = rules.get(f)
        if r and r['kind'] == 'PFX' and r['cross']:
            for s, sc in sfx:
                if sc:
                    for row in r['rows']:
                        n = apply('PFX', row, s)
                        if n: forms.add(n)
    for x in forms: out[x] = 1

known = {w.lower() for w in out}
rows = {}
for line in open(OLD, encoding='utf-8'):
    w, _, f = line.rstrip('\n').partition(' ')
    f = int(f)
    if w.lower() in known or f >= 100:
        rows[w] = f
for w in out:
    if w not in rows and w.lower() not in rows:
        rows[w] = 2
items = sorted(rows.items(), key=lambda kv: (-kv[1], kv[0]))
with gzip.open(NEW, 'wt', encoding='utf-8', compresslevel=9) as fh:
    for w, f in items:
        fh.write(f'{w} {f}\n')
print(len(items), 'words', file=sys.stderr)
