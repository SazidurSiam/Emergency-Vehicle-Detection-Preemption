"""
prep_audio_v2.py - Noise-augmented retrain prep (Thesis Experiment 3 / RQ2)
---------------------------------------------------------------------------
Builds an AUGMENTED TRAIN set only. Val/test stay untouched (fair comparison).

For every train clip it saves TWO versions:
  1. clean MFCC (same as before)
  2. noisy MFCC  - mixed at random SNR in [-6, 20] dB
                   noise = 50% white Gaussian / 50% random other urban clip
Target: fix the -6dB collapse (0.48 -> expect 0.70+), Ramirez et al. 2022 style.

SPLIT IDENTICAL to prep_audio.py (same SEED + same code) -> scientifically
valid before/after comparison.

Run AFTER renaming old artifacts:
    move 04_Results\audio_cnn\best.pt 04_Results\audio_cnn\best_clean.pt
    move 04_Results\metrics\audio_metrics.csv 04_Results\metrics\audio_metrics_clean.csv
Then:
    python 03_Code/audio/prep_audio_v2.py       (~45-60 min)
    python 03_Code/audio/train_audio_cnn.py     (~15-20 min, overwrites best.pt)
    python 03_Code/audio/evaluate_audio.py      (~5 min, new audio_metrics.csv)
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
SNR_LO, SNR_HI = -6.0, 20.0

csvs = list(RAW.rglob("UrbanSound8K.csv"))
if not csvs:
    sys.exit(f"[ERROR] UrbanSound8K.csv not found under {RAW}")
meta_csv = csvs[0]

FOLDS = {}
for p in RAW.rglob("*"):
    if p.is_dir() and p.name.lower().startswith("fold"):
        FOLDS[p.name.lower()] = p
if not FOLDS:
    sys.exit(f"[ERROR] No fold folders under {RAW}")

df = pd.read_csv(meta_csv)
pos = df[df["class"] == POS_CLASS]
neg = df[df["class"].isin(NEG_POOL)].sample(n=len(pos), random_state=SEED)
data = pd.concat([pos, neg]).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

rng = np.random.default_rng(SEED)          # SAME seed -> SAME split as v1
idx = rng.permutation(len(data))
i1 = int(0.8 * len(data))
train_part = data.iloc[idx[:i1]]           # only train is rebuilt

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

def mix_at_snr(y, noise, snr_db):
    p_sig = np.mean(y ** 2) + 1e-12
    p_no  = np.mean(noise ** 2) + 1e-12
    return (y + noise * np.sqrt(p_sig / (p_no * 10 ** (snr_db / 10)))).astype(np.float32)

clip_path = lambda row: FOLDS[f"fold{int(row['fold'])}"] / row["slice_file_name"]

# materialize train waveforms once (small enough: ~1400 clips x 4s)
rows = list(train_part.iterrows())
waves = {}
for _, row in rows:
    y = load_clip(clip_path(row))
    if y is not None:
        waves[row["slice_file_name"]] = (y, 1 if row["class"] == POS_CLASS else 0)
names = list(waves)
print(f"Train clips loaded: {len(names)}  -> building clean + noisy copies...")

arng = np.random.default_rng(SEED + 999)
X, Y = [], []
for k, name in enumerate(names):
    y, lab = waves[name]
    X.append(mfcc(y)); Y.append(lab)                       # clean copy
    snr = float(arng.uniform(SNR_LO, SNR_HI))              # noisy copy
    if arng.random() < 0.5:
        noise = arng.normal(0.0, 1.0, len(y)).astype(np.float32)
    else:
        other = waves[names[arng.integers(0, len(names))]][0]
        off = int(arng.integers(0, max(1, len(other) - TARGET)))
        seg = other[off:off + TARGET]
        noise = np.pad(seg, (0, TARGET - len(seg))) if len(seg) < TARGET else seg
    X.append(mfcc(mix_at_snr(y, noise, snr))); Y.append(lab)
    if (k + 1) % 100 == 0:
        print(f"  {k+1}/{len(names)} clips augmented...", flush=True)

X = np.stack(X); Y = np.array(Y, dtype=np.int64)
np.save(OUT / "X_train.npy", X); np.save(OUT / "y_train.npy", Y)
print(f"\n[train-augmented] saved X{X.shape} -> {OUT}")
json.dump({"noise_augmentation": {"snr_range_db": [SNR_LO, SNR_HI],
                                  "noise_types": ["white", "random_urban_clip"]}},
          open(OUT / "augmentation_log.json", "w"), indent=2)
print("DONE. Next: python 03_Code/audio/train_audio_cnn.py")
