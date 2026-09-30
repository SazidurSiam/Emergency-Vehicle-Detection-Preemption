"""
merge_bd_v3.py - STRICT regional merge (fixed lessons from attempt 1)
- Indian: ONLY full-vehicle boxes ('ambulance', 'fire_truck'); all sub-attribute
  boxes (ambulance_text/lamp/108, firesymbol/lamp/ladder/writing) are DROPPED.
- Indian train subsampled to MAX 900 images (rebalances: BD ~20% of pool).
- BD: all images, 70/15/15, BD-ONLY test (same 34 as attempt 1, same seed).
Output: 02_Data/visual/local_bd/merged2/
Run:  python 03_Code/visual/merge_bd_v3.py
"""
from pathlib import Path
import random, shutil, re, sys

BASE = Path(__file__).resolve().parents[2]
INDIAN = BASE / "02_Data" / "visual" / "indian_set"
BD     = BASE / "02_Data" / "visual" / "local_bd" / "bd_labeled"
OUT    = BASE / "02_Data" / "visual" / "local_bd" / "merged2"
SEED = 42
MAX_INDIAN_TRAIN = 900
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def die(m): sys.exit(f"[ERROR] {m}")
if not INDIAN.exists(): die(f"Missing: {INDIAN}")
if not BD.exists():     die(f"Missing: {BD}")
for split in ("train", "val", "test"):
    for sub in ("images", "labels"):
        (OUT / split / sub).mkdir(parents=True, exist_ok=True)

stats, dropped = {}, {"boxes": 0, "imgs": 0}

# class id -> name map from Indian yaml
raw = re.findall(r"'([^']+)'", (INDIAN / "data.yaml").read_text(errors="ignore"))
def cls_name(cid): return raw[cid].lower() if cid < len(raw) else ""

ind_train_pairs, ind_val_pairs = [], []
for split in ("train", "valid", "test"):
    sp = INDIAN / split
    if not sp.exists(): continue
    imgs = {p.stem: p for p in (sp / "images").glob("*") if p.suffix.lower() in IMG_EXT}
    for lab in (sp / "labels").glob("*.txt"):
        img = imgs.get(lab.stem)
        if img is None: continue
        keep = []
        for line in lab.read_text(errors="ignore").splitlines():
            parts = line.split()
            if len(parts) < 5: continue
            n = cls_name(int(float(parts[0])))
            if n == "ambulance":  keep.append("0 " + " ".join(parts[1:5]))
            elif n == "fire_truck": keep.append("1 " + " ".join(parts[1:5]))
            else: dropped["boxes"] += 1
        if keep:
            (ind_train_pairs if split == "train" else ind_val_pairs).append((img, keep))
        else:
            dropped["imgs"] += 1

random.seed(SEED); random.shuffle(ind_train_pairs)
ind_train_pairs = ind_train_pairs[:MAX_INDIAN_TRAIN]

def write_pair(img, lines, dest, prefix):
    shutil.copy2(img, OUT / dest / "images" / f"{prefix}{img.name}")
    (OUT / dest / "labels" / f"{prefix}{img.stem}.txt").write_text("\n".join(lines) + "\n")
    stats[dest] = stats.get(dest, 0) + 1

for img, lines in ind_train_pairs: write_pair(img, lines, "train", "ind_")
for img, lines in ind_val_pairs:   write_pair(img, lines, "val",  "ind_")

# ---- BD: 70/15/15, same seed -> same test split as attempt 1 ----
labs = {}
for p in BD.rglob("*.txt"): labs.setdefault(p.stem, p)
pairs = []
for img in BD.rglob("*"):
    if img.suffix.lower() in IMG_EXT and "labels" not in str(img).lower():
        if img.stem in labs: pairs.append((img, labs[img.stem]))
random.seed(SEED); random.shuffle(pairs)
n = len(pairs); i1, i2 = int(.70*n), int(.85*n)
for img, lab in pairs[:i1]:   write_pair(img, lab.read_text(errors="ignore").splitlines(), "train", "bd_")
for img, lab in pairs[i1:i2]: write_pair(img, lab.read_text(errors="ignore").splitlines(), "val",  "bd_")
for img, lab in pairs[i2:]:   write_pair(img, lab.read_text(errors="ignore").splitlines(), "test", "bd_")

(OUT / "data.yaml").write_text(
    f"path: {OUT.as_posix()}\ntrain: train/images\nval: val/images\ntest: test/images\n"
    f"nc: 2\nnames: ['Ambulance', 'Fire Engine']\n")
print("=== MERGE v3 COMPLETE ===")
print("Splits:", stats)
print(f"Dropped (non full-vehicle boxes): {dropped['boxes']}, empty imgs: {dropped['imgs']}")
print("Next: python 03_Code/visual/fine_tune_bd2.py")
