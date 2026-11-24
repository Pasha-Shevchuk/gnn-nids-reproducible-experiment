"""Завантажує NSL-KDD з публічного дзеркала і перевіряє MD5 за файлами DVC (data/*.dvc).

Використання:  python scripts/get_data.py
Скрипт потребує лише стандартної бібліотеки Python.
"""
import hashlib
import pathlib
import re
import sys
import urllib.request

MIRROR = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/"
FILES = {"KDDTrain+.txt": "KDDTrain%2B.txt", "KDDTest+.txt": "KDDTest%2B.txt"}
DATA = pathlib.Path(__file__).resolve().parent.parent / "data"


def md5(path: pathlib.Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def expected_md5(name: str) -> str:
    text = (DATA / f"{name}.dvc").read_text(encoding="utf-8")
    return re.search(r"md5:\s*([0-9a-f]{32})", text).group(1)


def main() -> int:
    ok = True
    for name, remote in FILES.items():
        target = DATA / name
        if not target.exists():
            print(f"Завантаження {name} ...")
            urllib.request.urlretrieve(MIRROR + remote, target)
        exp, got = expected_md5(name), md5(target)
        status = "OK" if exp == got else "НЕВІДПОВІДНІСТЬ"
        ok &= exp == got
        print(f"{name}: md5 {got} (очікувалось {exp}) – {status}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
