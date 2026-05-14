# %% Cell
import re
from pathlib import Path

import pandas as pd

ROOT = Path("../data/raw/grabmyo/1.1.0")

pattern = re.compile(
    r"session(?P<session>\d+)_participant(?P<participant>\d+)_gesture(?P<gesture>\d+)_trial(?P<trial>\d+)"
)

# Выбранные 4 класса
GESTURE_MAP = {
    11: "WF",  # wrist flexion
    12: "WE",  # wrist extension
    15: "HO",  # hand open
    16: "HC",  # hand close
}

# Кодируем метки в 0..3 (фиксированный порядок)
LABELS = {"WF": 0, "WE": 1, "HO": 2, "HC": 3}

rows = []
for hea in ROOT.rglob("*.hea"):
    m = pattern.match(hea.stem)
    if not m:
        continue
    g = int(m.group("gesture"))
    if g not in GESTURE_MAP:
        continue

    cls = GESTURE_MAP[g]
    rows.append({
        "record_path": str(hea.with_suffix("")),  # без .hea
        "session": int(m.group("session")),
        "participant": int(m.group("participant")),
        "gesture": g,
        "class": cls,
        "y": LABELS[cls],
        "trial": int(m.group("trial")),
    })

df = pd.DataFrame(rows).sort_values(["participant", "session", "gesture", "trial"]).reset_index(drop=True)

out = Path("data/processed/grabmyo_index_4classes.csv")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)

print("Saved:", out)
print("Rows:", len(df))
print(df["class"].value_counts().sort_index())
print("Participants:", df["participant"].nunique())
