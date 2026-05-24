"""
download_dataset.py
───────────────────
Downloads the Garbage Classification dataset (~3.2 GB) from Kaggle.

Prerequisites
─────────────
1. Create a free Kaggle account at https://www.kaggle.com
2. Go to Account → API → Create New Token  →  kaggle.json downloaded
3. Place kaggle.json at  ~/.kaggle/kaggle.json  (Linux/Mac)
                      or  C:\\Users\\<name>\\.kaggle\\kaggle.json  (Windows)
4. Run:  python download_dataset.py
"""

import os
import subprocess
import sys
import zipfile
import shutil

DATASET_SLUG  = "asdasdasasdas/garbage-classification"   # ~3.2 GB
DOWNLOAD_DIR  = "./data"
EXTRACT_TO    = "./data/garbage_classification"


def check_kaggle():
    try:
        import kaggle  # noqa: F401
        return True
    except ImportError:
        print("📦 Installing kaggle...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle", "-q"])
        return True


def download():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    print("⬇️  Downloading Garbage Classification dataset (~3.2 GB) ...")
    subprocess.check_call([
        "kaggle", "datasets", "download",
        "-d", DATASET_SLUG,
        "-p", DOWNLOAD_DIR,
    ])
    print("✅ Download complete.")


def extract():
    zip_path = os.path.join(DOWNLOAD_DIR, "garbage-classification.zip")
    if not os.path.exists(zip_path):
        # Try alternative name
        zips = [f for f in os.listdir(DOWNLOAD_DIR) if f.endswith(".zip")]
        if zips:
            zip_path = os.path.join(DOWNLOAD_DIR, zips[0])
        else:
            print("❌ Zip file not found. Check download.")
            return

    print(f"📦 Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DOWNLOAD_DIR)
    print(f"✅ Extracted to {DOWNLOAD_DIR}")

    # Ensure standard directory structure
    if not os.path.exists(EXTRACT_TO):
        # Some Kaggle zips have a nested folder; flatten if needed
        subdirs = [d for d in os.listdir(DOWNLOAD_DIR)
                   if os.path.isdir(os.path.join(DOWNLOAD_DIR, d))]
        print(f"   Found subdirs: {subdirs}")
        if subdirs:
            src = os.path.join(DOWNLOAD_DIR, subdirs[0])
            shutil.move(src, EXTRACT_TO)
            print(f"   Moved to {EXTRACT_TO}")

    print(f"\n📂 Dataset ready at: {EXTRACT_TO}")
    size_gb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fs in os.walk(EXTRACT_TO)
        for f in fs
    ) / 1e9
    print(f"📦 Dataset size: {size_gb:.2f} GB")

    classes = [d for d in os.listdir(EXTRACT_TO)
               if os.path.isdir(os.path.join(EXTRACT_TO, d))]
    print(f"🗂️  Classes found: {classes}")


if __name__ == "__main__":
    check_kaggle()
    download()
    extract()
    print("\n✅ Ready! Now run:  python train.py")
