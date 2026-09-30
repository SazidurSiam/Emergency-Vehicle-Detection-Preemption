"""
prep_data.py - Stage A data preparation (visual pipeline)
Run this FIRST, before training.

1. Scans 02_Data/visual/raw_roboflow/ for the Roboflow export
   (YOLOv8 format: images + .txt label files with bounding boxes)
2. Pairs every image with its label file
3. Splits 80% train / 10% val / 10% test  (proposal Section 6.1)
4. Writes 02_Data/splits/{train,val,test}/{images,labels}
5. Generates 02_Data/splits/data.yaml ready for Ultralytics YOLOv8

Usage (thesis env active, from repo root):
    python 03_Code/prep_data.py
"""
from pathlib import Path
import random, shutil, sys, re

BASE = Path(__file__).resolve().parents[2]          # repo root
RAW  = BASE / "02_Data" / "visual" / "raw_roboflow"
OUT  = BASE / "02_Data" / "splits"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42

def die(msg): sys.exit(f"\n[ERROR] {msg}\n")

if not RAW.exists():
    die(f"Folder not found: {RAW}\n"
        "Download the Roboflow 'Emergency Vehicle Detection' dataset\n"
        "(YOLOv8 format) and extract it into raw_roboflow/.")

# ---- collect labels by stem, then pair with images -------------------------
labels_by_stem = {}
for p in RAW.rglob("*.txt"):
    labels_by_stem.setdefault(p.stem, p)

images = [p for p in RAW.rglob("*") if p.suffix.lower() in IMG_EXT]
pairs, orphans = [], []
for img in images:
    lab = labels_by_stem.get(img.stem)
    (pairs if lab else orphans).append((img, lab))

if len(pairs) < 20:
    die(f"Only {len(pairs)} image/label pairs found in {RAW}.\n"
        "Check that the Roboflow export (YOLOv8 format) is extracted there.")

# ---- class names -----------------------------------------------------------
names = None
for cand in list(RAW.rglob("data.yaml")) + list(RAW.rglob("_darknet.labels")):
    try:
        txt = cand.read_text(encoding="utf-8", errors="ignore")
        if cand.suffix == ".yaml":
            m = re.search(r"names:\s*\[(.*?)\]", txt, re.S)
            if m:
                names = [n.strip().strip("'\"") for n in m.group(1).split(",") if n.strip()]
        else:
            names = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    except Exception:
        pass
if not names:
    max_id = -1
    for _, lab in pairs[:200]:
        for line in lab.read_text(errors="ignore").splitlines():
            parts = line.split()
            if parts:
                max_id = max(max_id, int(float(parts[0])))
    names = [f"class_{i}" for i in range(max_id + 1)] if max_id >= 0 else ["emergency_vehicle"]

# ---- split 80/10/10 and copy ----------------------------------------------
random.seed(SEED)
random.shuffle(pairs)
n = len(pairs)
n_train, n_val = int(n * 0.80), int(n * 0.10)
splits = {"train": pairs[:n_train],
          "val":   pairs[n_train:n_train + n_val],
          "test":  pairs[n_train + n_val:]}

for split, items in splits.items():
    for sub in ("images", "labels"):
        (OUT / split / sub).mkdir(parents=True, exist_ok=True)
    for img, lab in items:
        shutil.copy2(img, OUT / split / "images" / img.name)
        shutil.copy2(lab, OUT / split / "labels" / lab.name)

(OUT / "data.yaml").write_text(
    f"path: {OUT.as_posix()}\n"
    f"train: train/images\n"
    f"val: val/images\n"
    f"test: test/images\n"
    f"nc: {len(names)}\n"
    f"names: {names}\n")

print(f"Pairs found : {len(pairs)}  (orphan images without labels: {len(orphans)})")
for s, it in splits.items():
    print(f"  {s:5s}: {len(it)}")
print(f"Classes ({len(names)}): {names}")
print(f"\nNEXT: python 03_Code/train_yolo.py")
