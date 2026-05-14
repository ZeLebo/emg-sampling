# %% Cell
import re
from pathlib import Path

import pandas as pd

ROOT = Path("../data/raw/grabmyo/1.1.0")

pattern = re.compile(
    r"session(?P<session>\d+)_participant(?P<participant>\d+)_gesture(?P<gesture>\d+)_trial(?P<trial>\d+)"
)

rows = []

for hea in ROOT.rglob("*.hea"):
    name = hea.stem
    m = pattern.match(name)
    if not m:
        continue

    rows.append({
        "path": str(hea.with_suffix("")),
        "session": int(m.group("session")),
        "participant": int(m.group("participant")),
        "gesture": int(m.group("gesture")),
        "trial": int(m.group("trial")),
    })

df = pd.DataFrame(rows)

print("Всего записей:", len(df))
print(df.head())
print("Уникальные жесты:", sorted(df["gesture"].unique()))
