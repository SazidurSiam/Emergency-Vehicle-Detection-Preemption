"""
evaluate_audio.py - Audio Stage A evaluation (proposal Section 6.6)
Clean test metrics: Accuracy, Precision, Recall, F1, FPR
Robustness test: -6dB SNR white-noise mix on the saved test waveforms
                 (like Ramirez et al. 2022), re-extracting MFCC before classifying.
Output: 04_Results/metrics/audio_metrics.csv + confusion matrix figure

Run:  python 03_Code/audio/evaluate_audio.py
"""
from pathlib import Path
import json
import numpy as np
import librosa, torch, torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "02_Data" / "splits" / "audio"
RES  = BASE / "04_Results" / "audio_cnn"
MET  = BASE / "04_Results" / "metrics"
FIG  = BASE / "04_Results" / "figures"
MET.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)

ckpt = torch.load(RES / "best.pt", weights_only=False)
mean, std, classes = ckpt["mean"], ckpt["std"], ckpt["classes"]

class SirenCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(64, 2)
    def forward(self, x):
        return self.fc(self.net(x).flatten(1))

model = SirenCNN(); model.load_state_dict(ckpt["state"]); model.eval()
meta = json.load(open(DATA / "audio_meta.json"))
SR, N_MFCC = meta["sr"], meta["n_mfcc"]

Xte = (np.load(DATA / "X_test.npy") - mean) / std
yte = np.load(DATA / "y_test.npy")
with torch.no_grad():
    preds = model(torch.tensor(Xte).unsqueeze(1).float()).argmax(1).numpy()

acc  = accuracy_score(yte, preds)
prec = precision_score(yte, preds, zero_division=0)
rec  = recall_score(yte, preds, zero_division=0)
f1   = f1_score(yte, preds, zero_division=0)
tn, fp, fn, tp = confusion_matrix(yte, preds, labels=[0, 1]).ravel()
fpr = fp / (fp + tn)
print(f"CLEAN TEST  Acc={acc:.4f}  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}  FPR={fpr:.4f}")

cm = confusion_matrix(yte, preds, labels=[0, 1])
fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(cm, cmap="Blues")
for (i, j), v in np.ndenumerate(cm):
    ax.text(j, i, str(v), ha="center", va="center")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(classes); ax.set_yticklabels(classes)
ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Siren CNN - Test Confusion Matrix")
fig.tight_layout(); fig.savefig(FIG / "audio_confusion_matrix.png", dpi=150)

# ---- -6dB SNR robustness test (waveform domain, like Ramirez et al.) ----
acc_snr = None
npz = DATA / "test_waveforms.npz"
if npz.exists():
    d = np.load(npz); waves, labels, rng = d["waveforms"], d["labels"], np.random.default_rng(7)
    correct = 0
    for y, lab in zip(waves, labels):
        p_sig = np.mean(y ** 2) + 1e-12
        noise = rng.normal(0, np.sqrt(p_sig / (10 ** (-6 / 10))), len(y))
        m = librosa.feature.mfcc(y=(y + noise).astype(np.float32), sr=SR,
                                 n_mfcc=N_MFCC, n_fft=2048, hop_length=512)
        x = torch.tensor(((m - mean) / std)).unsqueeze(0).unsqueeze(0).float()
        with torch.no_grad():
            if model(x).argmax(1).item() == lab:
                correct += 1
    acc_snr = correct / len(labels)
    print(f"SNR -6dB TEST  Acc={acc_snr:.4f}  ({correct}/{len(labels)})")

with open(MET / "audio_metrics.csv", "w") as f:
    f.write("metric,value\n")
    for k, v in [("accuracy", acc), ("precision", prec), ("recall", rec),
                 ("f1", f1), ("fpr", fpr), ("accuracy_snr_minus6db", acc_snr)]:
        f.write(f"{k},{v if v is not None else 'NA'}\n")
print(f"\nSaved: {MET / 'audio_metrics.csv'}")
print("Figure: figures/audio_confusion_matrix.png")
