# %% cell
from pathlib import Path

import wfdb

ROOT = Path("../data/raw/grabmyo/1.1.0")

def main():
    hea_files = list(ROOT.rglob("*.hea"))
    if not hea_files:
        raise SystemExit("Не найдено .hea файлов. Проверь путь ROOT и что скачивание завершилось.")

    record = hea_files[0].with_suffix("")  # путь без расширения
    sig, fields = wfdb.rdsamp(str(record))

    print("Record:", record)
    print("Signal shape (T, C):", sig.shape)
    print("fs:", fields["fs"])
    print("channels:", fields.get("sig_name", [])[:10])

if __name__ == "__main__":
    main()
