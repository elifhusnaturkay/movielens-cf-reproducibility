"""Download the three MovieLens datasets into DATA_DIR (default ~/.surprise_data).

    python3 scripts/download_data.py            # ml-latest-small and ml-100k
    python3 scripts/download_data.py --20m      # also MovieLens 20M (about 190 MB)

Files come from https://files.grouplens.org/datasets/movielens/.
Note: GroupLens updates ml-latest-small from time to time. The results in this
repository use the September 2018 version (100,836 ratings; ratings.csv md5
325a6f09399d1c45671f187505e1a100). The script prints the md5 so you can check.
"""

import hashlib
import io
import os
import sys
import urllib.request
import zipfile

DATA_DIR = os.path.expanduser(os.environ.get("CFREPRO_DATA", "~/.surprise_data"))
BASE = "https://files.grouplens.org/datasets/movielens/"


def fetch(name, target):
    if os.path.exists(os.path.join(DATA_DIR, target)):
        print("already there:", target)
        return
    print("downloading", name, "...")
    data = urllib.request.urlopen(BASE + name).read()
    root = os.path.join(DATA_DIR, "ml-100k") if name == "ml-100k.zip" else DATA_DIR
    zipfile.ZipFile(io.BytesIO(data)).extractall(root)


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    fetch("ml-latest-small.zip", "ml-latest-small")
    fetch("ml-100k.zip", "ml-100k/ml-100k")
    if "--20m" in sys.argv:
        fetch("ml-20m.zip", "ml-20m")
    print("ml-latest-small ratings.csv md5:",
          md5(os.path.join(DATA_DIR, "ml-latest-small", "ratings.csv")))
