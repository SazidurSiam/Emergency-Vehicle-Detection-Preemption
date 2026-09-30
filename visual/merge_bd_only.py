"""merge_bd_only.py - BD-only dataset from merged2's bd_* files (SAME 34 test images)"""
from pathlib import Path
import shutil
BASE = Path(__file__).resolve().parents[2]
SRC = BASE / "02_Data" / "visual" / "local_bd" / "merged2"
OUT = BASE / "02_Data" / "visual" / "local_bd" / "merged3"
for s in ("train", "val", "test"):
    for sub in ("images", "labels"):
        (OUT / s / sub).mkdir(parents=True, exist_ok=True)
n = 0
for s in ("train", "val", "test"):
    for img in (SRC / s / "images").glob("bd_*"):
        shutil.copy2(img, OUT / s / "images" / img.name)
        lab = SRC / s / "labels" / (img.stem + ".txt")
        if lab.exists():
            shutil.copy2(lab, OUT / s / "labels" / lab.name)
            n += 1
(OUT / "data.yaml").write_text(
    "path: " + OUT.as_posix() + "\ntrain: train/images\nval: val/images\n"
    "test: test/images\nnc: 2\nnames: ['Ambulance', 'Fire Engine']\n")
print("BD-only dataset:", n, "images ->", OUT)
