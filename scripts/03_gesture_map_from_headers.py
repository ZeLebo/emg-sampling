# %% Cell
import re
from pathlib import Path

import wfdb

ROOT = Path("../data/raw/grabmyo/1.1.0")

pattern = re.compile(
    r"session(?P<session>\d+)_participant(?P<participant>\d+)_gesture(?P<gesture>\d+)_trial(?P<trial>\d+)"
)

# возьмём по одному .hea на каждый gesture
picked = {}
for hea in ROOT.rglob("*.hea"):
    m = pattern.match(hea.stem)
    if not m:
        continue
    g = int(m.group("gesture"))
    if g not in picked:
        picked[g] = hea

print("Found gestures:", sorted(picked.keys()))
print()

for g in sorted(picked.keys()):
    hea = picked[g]
    rec = str(hea.with_suffix(""))
    hdr = wfdb.rdheader(rec)

    # В header комментарии могут быть в hdr.comments (список строк)
    comments = getattr(hdr, "comments", None) or []
    # ещё иногда что-то полезное бывает в record_name или сигналах
    print(f"gesture {g:02d} -> record: {hea.name}")
    if comments:
        for c in comments:
            print("  comment:", c)
    else:
        print("  comment: (none)")
    print()
