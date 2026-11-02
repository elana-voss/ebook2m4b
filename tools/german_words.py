"""Builds data/german-words.txt.gz from LibreOffice's de_DE_frami dictionary (GNU GPL 2 or 3; see
data/german-words-source.txt): the hunspell .dic + .aff expanded to lower-case word forms (a stem's
suffixes, one continuation, then prefixes). Usage: german_words.py <file.dic> <file.aff> <out.gz>."""
import collections
import gzip
import re
import sys

dic, aff, out = sys.argv[1:4]
rules = collections.defaultdict(list)
kinds = {}
L = open(aff, encoding='latin-1').read().splitlines()
i = 0
while i < len(L):
    p = L[i].split()
    if len(p) >= 4 and p[0] in ('SFX', 'PFX') and p[2] in ('Y', 'N') and p[3].isdigit():
        kind, flag, n = p[0], p[1], int(p[3])
        kinds[flag] = kind
        for r in L[i + 1:i + 1 + n]:
            q = r.split()
            strip = '' if q[2] == '0' else q[2]
            add = q[3].split('/')[0]
            add = '' if add == '0' else add
            cont = q[3].split('/')[1] if '/' in q[3] else ''
            cond = q[4] if len(q) > 4 else '.'
            rx = re.compile(cond + '$' if kind == 'SFX' else '^' + cond)
            rules[flag].append((strip, add, cont, rx))
        i += n
    i += 1


def apply(w, flag):
    for strip, add, cont, rx in rules.get(flag, ()):
        if not rx.search(w):
            continue
        if kinds[flag] == 'SFX':
            if strip and not w.endswith(strip):
                continue
            yield (w[:-len(strip)] if strip else w) + add, cont
        else:
            if strip and not w.startswith(strip):
                continue
            yield add + w[len(strip):], cont


forms = set()
for line in open(dic, encoding='latin-1'):
    line = line.strip()
    if not line or line.startswith('#') or line[0].isdigit():
        continue
    stem, _, flags = line.partition('/')
    flags = flags.split()[0] if flags else ''
    if set(flags) & set('doh'):  # forbidden, only in a compound, needs an affix
        continue
    got = {stem}
    for f in (f for f in flags if kinds.get(f) == 'SFX'):
        for w, cont in apply(stem, f):
            got.add(w)
            for g in cont:
                for w2, _ in apply(w, g):
                    got.add(w2)
    for f in (f for f in flags if kinds.get(f) == 'PFX'):
        for w in list(got):
            for w2, _ in apply(w, f):
                got.add(w2)
    forms.update(got)
low = sorted({w.lower() for w in forms if re.fullmatch(r"[A-Za-zÄÖÜäöüß]+", w)})
with open(out, 'wb') as f:
    f.write(gzip.compress(('\n'.join(low) + '\n').encode('utf-8'), 9, mtime=0))
print(len(low), 'forms')
