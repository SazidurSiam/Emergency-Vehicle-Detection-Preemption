"""
prep_audio.py - Audio Stage A: UrbanSound8K -> balanced binary siren dataset (MFCC)
Implements proposal Section 6.1 (audio) and 6.4 (MFCC feature extraction).

Positive class : siren (~873 clips)
Negative class : equal-sized random sample from car_horn, engine_idling,
                 street_music, air_conditioner + others (balanced binary set)
Splits         : 80% train / 10% val / 10% test  (SEED=42, reproducible)
Features       : 40 MFCCs, hop 512, n_fft 2048, sr 22050, 4s clips
Outputs        : 02_Data/splits/audio/{X,y}_{train,val,test}.npy
                 02_Data/splits/audio/test_waveforms.npz  (for -6dB SNR test)
Runtime on old CPU: ~30-60 minutes. Progress printed every 100 clips.

Run:  python 03_Code/audio/prep_audio.py
"""
from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
import librosa

BASE = Path(__file__).resolve().parents[2]
RAW  = BASE / "02_Data" / "audio" / "raw_urbansound8k"
OUT  = BASE / "02_Data" / "splits" / "audio"

SR, DURATION, N_MFCC, SEED = 22050, 4.0, 40, 42
POS_CLASS = "siren"
NEG_POOL = ["car_horn", "engine_idling", "street_music", "air_conditioner",
            "children_playing", "drilling", "dog_bark", "jackhammer"]

csvs = list(RAW.rglob("UrbanSound8K.csv"))
if not csvs:
    sys.exit(f"[ERROR] UrbanSound8K.csv not found under {RAW}\n"
             "Extract the UrbanSound8K archive there - it should contain "
             "fold1..fold10 and a metadata/ folder.")
meta_csv = csvs[0]

# ---- ROBUST fold discovery: works no matter how the archive is nested ------
# (e.g. raw_urbansound8k/UrbanSound8K/audio/fold1  OR  .../UrbanSound8K/fold1)
FOLDS = {}
for p in RAW.rglob("*"):
    if p.is_dir() and p.name.lower().startswith("fold"):
        FOLDS[p.name.lower()] = p
if not FOLDS:
    sys.exit(f"[ERROR] No 'fold*' folders found under {RAW}. "
             "Check that the UrbanSound8K archive extracted fully.")
print(f"Found fold folders: {sorted(FOLDS)}")

df = pd.read_csv(meta_csv)
pos = df[df["class"] == POS_CLASS]
neg = df[df["class"].isin(NEG_POOL)].sample(n=len(pos), random_state=SEED)
data = pd.concat([pos, neg]).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
print(f"Balanced dataset: {len(pos)} siren + {len(neg)} non-siren = {len(data)} clips")

rng = np.random.default_rng(SEED)
idx = rng.permutation(len(data))
i1, i2 = int(0.8 * len(data)), int(0.9 * len(data))
parts = {"train": data.iloc[idx[:i1]],
         "val":   data.iloc[idx[i1:i2]],
         "test":  data.iloc[idx[i2:]]}

TARGET = int(DURATION * SR)

def load_clip(path):
    try:
        y, _ = librosa.load(str(path), sr=SR, duration=DURATION, mono=True)
    except Exception:
        return None
    if len(y) < TARGET // 2:
        return None
    return np.pad(y, (0, TARGET - len(y)))[:TARGET] if len(y) < TARGET else y[:TARGET]

def mfcc(y):
    return librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC,
                                n_fft=2048, hop_length=512).astype(np.float32)

OUT.mkdir(parents=True, exist_ok=True)
for split, part in parts.items():
    X, Y, waves, names = [], [], [], []
    skipped = 0
    for k, (_, row) in enumerate(part.iterrows()):
        fold_dir = FOLDS.get(f"fold{int(row['fold'])}")
        clip = (fold_dir / row["slice_file_name"]) if fold_dir else None
        y = load_clip(clip) if clip else None
        if y is None:
            skipped += 1
            continue
        X.append(mfcc(y)); Y.append(1 if row["class"] == POS_CLASS else 0)
        if split == "test":
            waves.append(y.astype(np.float32)); names.append(clip.name)
        if (k + 1) % 100 == 0:
            print(f"  [{split}] {k+1}/{len(part)} processed... (skipped: {skipped})", flush=True)
    if not X:
        sys.exit(f"[ERROR] All clips failed for split '{split}'. "
                 f"Folds found: {sorted(FOLDS)}")
    X = np.stack(X); Y = np.array(Y, dtype=np.int64)
    np.save(OUT / f"X_{split}.npy", X); np.save(OUT / f"y_{split}.npy", Y)
    print(f"[{split}] saved X{X.shape} -> {OUT}  (skipped: {skipped})")
    if split == "test":
        np.savez_compressed(OUT / "test_waveforms.npz",
                            waveforms=np.stack(waves), labels=Y, names=names, sr=SR)

json.dump({"sr": SR, "duration": DURATION, "n_mfcc": N_MFCC, "seed": SEED},
          open(OUT / "audio_meta.json", "w"), indent=2)
print("\nDONE. Next: python 03_Code/audio/train_audio_cnn.py")
